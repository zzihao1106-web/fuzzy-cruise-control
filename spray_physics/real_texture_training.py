"""用真实清晰区域构造再打光配对：直接监督去光斑结果，保留原始喷雾纹理。"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from deep_data import read_gray, crop_geometry, crop_resize, write_json
from deep_spray import runtime
from gray_background import uniform_gray_variants
from gray_scene_data import temporal_channels
from restore_model import ReferenceRestorer

VERSION = "6.0.0-alpha1"


class RealTextureRestorer(ReferenceRestorer):
    """直接拟合背景光的保留系数，不再同时学习不可辨识的前景亮度。"""
    def __init__(self, width=16, depth=3, correction_limit=0.04):
        super().__init__(width, depth, correction_limit)
        self.intro = nn.Conv2d(9, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.constant_(self.ending.bias, 0.85)

    def decompose(self, x, project=False):
        transmission = self.ending(self.features(x)).sigmoid()
        gray = x[:, 1:2]-transmission*(x[:, 3:4]-x[:, 4:5])
        return {"gray": gray, "transmission": transmission, "alpha": 1-transmission,
                "offset": transmission*0, "reconstruction": x[:, 1:2]}

    def forward(self, x):
        return self.decompose(x)["gray"]


def prepare_real(args):
    from spray_recognition import discover_sequence
    if args.size < 32:
        raise ValueError("准备尺寸至少32。")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths, _ = discover_sequence(cfg["source_dir"])
    if len(paths) < 20:
        raise ValueError("时间块留出训练至少需要20帧；更少图像可先使用已训练权重处理。")
    reference = np.median(np.stack([read_gray(p) for p in paths[:3]]), 0)
    gray, _, _, _ = uniform_gray_variants(reference, cfg, True)
    bounds, roi, _ = crop_geometry(reference.shape, cfg)
    n = args.size
    b, g = crop_resize(reference, bounds, n), crop_resize(gray, bounds, n)
    valid = (np.abs(cv2.GaussianBlur(b-g, (0, 0), 2)) < 0.055)
    valid &= crop_resize(roi, bounds, n, True) > 0.5
    yy, xx = np.indices((n, n)); x0, y0, x1, y1 = bounds
    sx, sy = cfg["source_xy"]
    valid &= np.hypot(xx-(sx-x0)/(x1-x0)*n, yy-(sy-y0)/(y1-y0)*n) > n*0.09
    valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # 只在照明接近灰底的真实区域建立参考；其余位置既不计损失也不当真值。
    pool = np.stack([np.clip(crop_resize(read_gray(p), bounds, n)-b+g, 0, 1) for p in paths])
    pool[:, ~valid] = g[~valid]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"real_texture_pool.npz", pool=pool, valid=valid, gray=g,
                        highlight=np.maximum(b-g, 0))
    total = len(paths)
    train_end, val_start, val_end, test_start = int(total*0.43), int(total*0.57)+1, int(total*0.70), int(total*0.85)+1
    info = {"version": VERSION, "size": n, "frames": total,
            "train_frames_1based": list(range(1, train_end+1)),
            "validation_frames_1based": list(range(val_start, val_end+1)),
            "test_frames_1based": list(range(test_start, total+1)),
            "unused_gap_frames_1based": list(range(train_end+1, val_start))+list(range(val_end+1, test_start)),
            "reference_background_frames": [1, 2, 3], "valid_fraction": float(valid.mean()),
            "human_labels": 0, "pristine_real_spray_ground_truth": False,
            "meaning": "真实清晰区域经小幅背景校正后的代理参考；反事实加光验证，不等于原始重合区真值"}
    write_json(out/"real_texture_manifest.json", info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


class RealRelightingPairs:
    def __init__(self, data, split, seed):
        with np.load(data/"real_texture_pool.npz") as z:
            self.pool, self.valid, self.gray, self.highlight = (z[k] for k in ("pool", "valid", "gray", "highlight"))
        self.manifest = json.loads((data/"real_texture_manifest.json").read_text(encoding="utf-8"))
        self.indices = np.array(self.manifest[split+"_frames_1based"])-1
        self.rng = np.random.default_rng(seed)
        self.n = self.gray.shape[0]
        self.yy, self.xx = np.indices((self.n, self.n), dtype=np.float32)

    def sample(self):
        rng, n = self.rng, self.n
        k = int(rng.choice(self.indices))
        angle = float(rng.uniform(-180, 180))
        matrix = cv2.getRotationMatrix2D((n/2, n/2), angle, 1)
        rotate = lambda a: cv2.warpAffine(a.astype(np.float32), matrix, (n, n), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.median(self.gray)))
        valid = cv2.warpAffine(self.valid.astype(np.uint8), matrix, (n, n), flags=cv2.INTER_NEAREST) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        g = np.full((n, n), float(np.median(self.gray)), np.float32)
        # 自然曝光与喷雾亮度扰动不改变纹理来源。
        scale = rng.uniform(0.8, 1.15)
        clean = [np.clip(g+(rotate(self.pool[j])-g)*scale, 0, 1) for j in
                 (max(int(self.indices[0]), k-1), k, min(int(self.indices[-1]), k+1))]
        highlight = np.zeros_like(g)
        if rng.random() < 0.6:
            matrix_h = cv2.getRotationMatrix2D((n/2, n/2), float(rng.uniform(-180, 180)), rng.uniform(0.8, 1.1))
            highlight = cv2.warpAffine(self.highlight, matrix_h, (n, n))*rng.uniform(0.6, 1.4)
        for _ in range(int(rng.integers(1, 4))):
            px, py = rng.uniform(0, n, 2); rx, ry = rng.uniform(0.06*n, 0.40*n, 2)
            highlight += rng.uniform(0.08, 0.35)*np.exp(-((self.xx-px)/rx)**2-((self.yy-py)/ry)**2)
        highlight = np.minimum(highlight, 0.95-g)
        if rng.random() < 0.12:
            highlight *= 0
        appearance = rng.uniform(0.62, 0.98)
        kind = rng.random()
        observed, trans = [], []
        for c in clean:
            opacity = np.clip((c-g)/max(appearance-float(g.mean()), 0.15), 0, 0.98)
            # 同时覆盖透射衰减与近似加性反光，不把一种成像假设当作事实。
            t = 1-opacity if kind < 0.75 else np.ones_like(c)
            obs = c+t*highlight+rng.normal(0, rng.uniform(0.001, 0.006), c.shape)
            observed.append(np.clip(obs, 0, 1)); trans.append(t)
        # 饱和像素不提供恢复真值，不用它们的误差粉饰结果。
        valid &= observed[1] < 0.98
        target = {"gray": clean[1][None], "transmission": trans[1][None], "valid": valid[None].astype(np.float32)}
        x = temporal_channels(observed[0], observed[1], observed[2], g+highlight, g)
        return x, target

    def batch(self, count):
        samples = [self.sample() for _ in range(count)]
        return np.stack([s[0] for s in samples]), {k: np.stack([s[1][k] for s in samples]) for k in samples[0][1]}


def pair_loss(z, target, x):
    valid = target["valid"]
    fg = (target["gray"]-x[:, 4:5]).abs() > 0.04
    def balanced(error):
        front, back = valid*fg, valid*~fg
        return 2*(error*front).sum()/front.sum().clamp_min(1)+(error*back).sum()/back.sum().clamp_min(1)
    # 喷雾与背景分别归一化，避免早期小喷雾被大面积背景淹没后退化成静态相减。
    mae = balanced((z["gray"]-target["gray"]).abs())
    t_loss = balanced((z["transmission"]-target["transmission"]).abs())
    return mae*4+t_loss*0.25


def covariance_gray(x):
    outputs = []
    for s in x:
        i, b, g = s[1], s[3], s[4]
        blur = lambda a: cv2.GaussianBlur(a, (0, 0), max(2, a.shape[0]*0.045))
        mb, mi = blur(b), blur(i)
        t = np.clip((blur(b*i)-mb*mi+0.00002)/(blur(b*b)-mb*mb+0.00002), 0, 1)
        outputs.append((i-t*(b-g))[None])
    return np.stack(outputs)


def evaluate_pairs(model, data, device, split="validation", batches=6, seed=713905, baseline=None):
    synth = RealRelightingPairs(data, split, seed)
    sums = {}
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, y = synth.batch(4); inputs = torch.from_numpy(x).to(device)
            outputs = {"real_texture_network": model.decompose(inputs)["gray"].cpu().numpy(),
                       "covariance": covariance_gray(x), "static_subtraction": x[:, 1:2]-(x[:, 3:4]-x[:, 4:5]),
                       "flat_field": x[:, 1:2]*x[:, 4:5]/np.maximum(x[:, 3:4], 1/255)}
            if baseline is not None:
                outputs["v5"] = baseline.decompose(inputs)["gray"].cpu().numpy()
            valid = y["valid"] > 0.5
            fg = valid & (np.abs(y["gray"]-x[:, 4:5]) > 0.04)
            bg = valid & (np.abs(y["gray"]-x[:, 4:5]) < 0.02)
            affected = fg & ((x[:, 3:4]-x[:, 4:5]) > 0.1)
            for name, prediction in outputs.items():
                counters = sums.setdefault(name, {})
                for key, mask in (("fg_mae", fg), ("bg_mae", bg), ("glare_overlap_fg_mae", affected)):
                    pair = counters.setdefault(key, [0., 0])
                    pair[0] += float(np.abs(prediction-y["gray"])[mask].sum()); pair[1] += int(mask.sum())
    scores = {name: {k: a/max(n, 1) for k, (a, n) in counters.items()} for name, counters in sums.items()}
    return {"split": split, "samples": batches*4, "scores": scores,
            "proxy_relighting_test_only": True, "original_glare_region_ground_truth_available": False}


def train_real(args):
    torch, device = runtime(args.device)
    if min(args.steps, args.batch) < 1:
        raise ValueError("训练步数与批次必须为正。")
    data, out = Path(args.data), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = RealTextureRestorer(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=0.00015)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    synth = RealRelightingPairs(data, "train", args.seed)
    best, history, start = float("inf"), [], time.perf_counter()
    for step in range(1, args.steps+1):
        model.train(); x, y = synth.batch(args.batch)
        inputs = torch.from_numpy(x).to(device)
        target = {k: torch.from_numpy(v).to(device) for k, v in y.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = pair_loss(model.decompose(inputs), target, inputs)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(opt); scaler.update(); schedule.step()
        if step == 1 or step % 100 == 0:
            print(f"真实纹理再打光 {step}/{args.steps}，损失{loss.item():.4f}，耗时{time.perf_counter()-start:.0f}秒", flush=True)
        if step % 200 == 0 or step == args.steps:
            result = evaluate_pairs(model, data, device)
            score = result["scores"]["real_texture_network"]
            rank = score["fg_mae"]+score["bg_mae"]+score["glare_overlap_fg_mae"]
            state = {"version": VERSION, "kind": "counterfactual_real_texture", "spec": model.spec,
                     "model": model.state_dict(), "step": step, "validation": result, "human_labels": 0}
            torch.save(state, out/"real_texture_last.pt")
            if rank < best:
                best = rank; torch.save(state, out/"real_texture_best.pt")
            history.append({"step": step, **result}); write_json(out/"history.json", history)
            print(json.dumps({"step": step, **result}, ensure_ascii=False), flush=True)
    state = torch.load(out/"real_texture_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    baseline = None
    if args.baseline:
        from gray_scene_model import TemporalGrayRestorer
        old = torch.load(args.baseline, map_location="cpu", weights_only=True)
        baseline = TemporalGrayRestorer(**old["spec"]).to(device).eval(); baseline.load_state_dict(old["model"])
    test = evaluate_pairs(model, data, device, "test", 16, 984205, baseline)
    write_json(out/"heldout_comparison.json", test)
    write_json(out/"training_summary.json", {"version": VERSION, "steps": args.steps, "best_step": state["step"],
               "size": synth.n, "seed": args.seed, "seconds": time.perf_counter()-start,
               "human_labels": 0, "real_pristine_accuracy_available": False, "test": test})


def main():
    p = argparse.ArgumentParser(description="真实清晰喷雾区域的反事实再打光训练与对照")
    p.add_argument("operation", choices=["prepare", "train"])
    p.add_argument("--config"); p.add_argument("--data"); p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=256); p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--width", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261032); p.add_argument("--device", default="auto")
    p.add_argument("--baseline")
    args = p.parse_args()
    (prepare_real if args.operation == "prepare" else train_real)(args)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
