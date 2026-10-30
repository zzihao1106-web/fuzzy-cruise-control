"""分层前景恢复的配对训练与独立测试，保留旧差分恢复器作对照。"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from deep_data import SpraySynthesizer, write_json
from deep_spray import runtime, score_batch


def optical_batch(synth, batch):
    samples = [synth.sample(optical=True) for _ in range(batch)]
    return (np.stack([s[0] for s in samples]), np.stack([s[1] for s in samples]),
            {k: np.stack([s[2][k] for s in samples]) for k in samples[0][2]})


def optical_metrics(prediction, alpha, target, x, semantic):
    fg = target["alpha"] > 0.05; bg = ~fg
    truth = target["foreground"]
    weak = fg & (truth > 0.025) & (np.abs(x[:, :1]-x[:, 1:2]) < 0.025)
    mean = lambda a, m: float(a[m].mean()) if m.any() else 0.0
    fmae, bmae = mean(np.abs(prediction-truth), fg), mean(prediction, bg)
    return {"foreground_mae": fmae, "background_mae": bmae,
            "low_contrast_foreground_mae": mean(np.abs(prediction-truth), weak),
            "low_contrast_recall_above_0_025": mean((prediction > 0.025).astype(float), weak),
            "low_contrast_pixels": int(weak.sum()), "foreground_energy_ratio": float(prediction[fg].sum()/max(1e-8, truth[fg].sum())),
            "alpha_mae": float(np.abs(alpha-target["alpha"]).mean()),
            "contrast_baseline_low_contrast_mae": mean(np.abs(np.abs(x[:, :1]-x[:, 1:2])-truth), weak),
            "selection_score_lower_is_better": fmae+bmae*3+mean(np.abs(prediction-truth), weak),
            "synthetic_only": True, "target_definition": "已知合成成像模型中的alpha乘前景层"}


def evaluate_optical(model, data, source, device, seed=348901, batches=6, baseline=None):
    import torch
    synth = SpraySynthesizer(data/"backgrounds.npz", source, seed, True, curriculum=True)
    pp, aa, xx, ss, tt, old, reconstructions = [], [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, semantic, target = optical_batch(synth, 4)
            inputs = torch.from_numpy(x).to(device)
            z = model.decompose(inputs)
            effective = z["projected_foreground"] if model.spec.get("physical_projection") else z["foreground"]
            pp.append(effective.cpu().numpy()); aa.append(z["alpha"].cpu().numpy())
            reconstructions.append(z["reconstruction"].cpu().numpy())
            xx.append(x); ss.append(semantic); tt.append(target)
            if baseline is not None:
                r, _ = baseline(inputs); old.append(r.abs().cpu().numpy())
    target = {k: np.concatenate([t[k] for t in tt]) for k in tt[0]}
    result = optical_metrics(np.concatenate(pp), np.concatenate(aa), target, np.concatenate(xx), np.concatenate(ss))
    result["reconstruction_mae"] = float(np.abs(np.concatenate(reconstructions)-np.concatenate(xx)[:, :1]).mean())
    result["physical_projection"] = bool(model.spec.get("physical_projection"))
    if baseline is not None:
        a = np.concatenate(old); truth = target["foreground"]
        weak = (target["alpha"] > 0.05) & (truth > 0.025) & (np.abs(np.concatenate(xx)[:, :1]-np.concatenate(xx)[:, 1:2]) < 0.025)
        result["old_residual_model_low_contrast_mae"] = float(np.abs(a[weak]-truth[weak]).mean()) if weak.any() else None
        result["old_residual_model_low_contrast_energy_ratio"] = float(a[weak].sum()/max(1e-8, truth[weak].sum()))
    return result


def train_optical(args):
    import torch
    from restore_model import OpticalRestorer, RESTORE_VERSION, optical_loss
    from deep_model import BackgroundUNet, supervised_loss
    torch, device = runtime(args.device)
    data, out = Path(args.data), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    resume = getattr(args, "resume_cleaner", None)
    if getattr(args, "foreground_gain", None) is not None and args.foreground_gain <= 0:
        raise ValueError("固定显示增益必须为正。")
    if min(args.boundary_steps, args.batch) < 1 or (not resume and args.steps < 1):
        raise ValueError("训练步数与batch必须为正。")
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    source = manifest["source_normalized"]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats()
    parent_state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if parent_state and parent_state.get("restore_kind") != "optical":
        raise ValueError("此热启动入口只接受分层恢复器权重。")
    restorer = OpticalRestorer(**parent_state["restorer_spec"]).to(device) if parent_state else OpticalRestorer(width=args.width).to(device)
    if parent_state:
        restorer.load_state_dict(parent_state["restorer"])
    if getattr(args, "physical_projection", False):
        restorer.spec["physical_projection"] = True
    opt = torch.optim.AdamW(restorer.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    synth = SpraySynthesizer(data/"backgrounds.npz", source, args.seed, curriculum=True)
    start, best, history = time.perf_counter(), float("inf"), []
    for step in range(1, (0 if resume else args.steps)+1):
        restorer.train(); x, semantic, target = optical_batch(synth, args.batch)
        inputs = torch.from_numpy(x).to(device)
        truth = {k: torch.from_numpy(v).to(device) for k, v in target.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            z = restorer.decompose(inputs); loss = optical_loss(z, truth, inputs)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(restorer.parameters(), 5); scaler.step(opt); scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"分层恢复训练 {step}/{args.steps}，损失{loss.item():.4f}", flush=True)
        if step % 100 == 0 or step == args.steps:
            metrics = evaluate_optical(restorer, data, source, device)
            history.append({"step": step, **metrics})
            state = {"version": RESTORE_VERSION, "restore_kind": "optical", "restorer_spec": restorer.spec,
                     "restorer": restorer.state_dict(), "cleaner_step": step, "gain": 2.0,
                     "seed": args.seed, "size": synth.size, "cleaner_validation": metrics,
                     "training_kind": "synthetic_optical_decomposition", "human_labels": 0}
            torch.save(state, out/"cleaner_last.pt")
            if metrics["selection_score_lower_is_better"] < best:
                best = metrics["selection_score_lower_is_better"]; torch.save(state, out/"cleaner_best.pt")
            write_json(out/"cleaner_history.json", history)
            print(f"分层验证：前景误差{metrics['foreground_mae']:.4f}，背景{metrics['background_mae']:.4f}，弱区误差{metrics['low_contrast_foreground_mae']:.4f}", flush=True)
    cleaner_seconds = time.perf_counter()-start
    state = parent_state or torch.load(out/"cleaner_best.pt", map_location="cpu", weights_only=True)
    state["gain"] = getattr(args, "foreground_gain", None) or state["gain"]
    state["restorer_spec"] = dict(restorer.spec)
    restorer.load_state_dict(state["restorer"]); restorer.eval()
    boundary = BackgroundUNet(width=16, iterations=1).to(device)
    opt = torch.optim.AdamW(boundary.parameters(), lr=0.001)
    synth = SpraySynthesizer(data/"backgrounds.npz", source, args.seed+4000, curriculum=True)
    best_iou, logs = -1, []
    for step in range(1, args.boundary_steps+1):
        x, semantic, _ = optical_batch(synth, args.batch)
        with torch.no_grad():
            r, _ = restorer(torch.from_numpy(x).to(device))
            c = ((r*state["gain"]).clamp(0, 1)*255).round()/255
            inputs = torch.cat([c, torch.zeros_like(c), c, torch.zeros_like(c)], 1)
        boundary.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = supervised_loss(boundary(inputs), torch.from_numpy(semantic).to(device))
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(boundary.parameters(), 5); scaler.step(opt); scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"分层去白图边界训练 {step}/{args.boundary_steps}，损失{loss.item():.4f}", flush=True)
        if step % 100 == 0 or step == args.boundary_steps:
            validation = SpraySynthesizer(data/"backgrounds.npz", source, args.seed+90000, True, curriculum=True)
            ps, ys = [], []; boundary.eval()
            with torch.inference_mode():
                for _ in range(4):
                    xx, yy, _ = optical_batch(validation, 4)
                    rr, _ = restorer(torch.from_numpy(xx).to(device))
                    cc = ((rr*state["gain"]).clamp(0, 1)*255).round()/255
                    p = boundary(torch.cat([cc, torch.zeros_like(cc), cc, torch.zeros_like(cc)], 1))[-1][:, 0].sigmoid()
                    ps.append(p.cpu().numpy()); ys.append(yy[:, 0])
            scores = score_batch(np.concatenate(ps), np.concatenate(ys)); logs.append({"step": step, **scores})
            bundle = {**state, "boundary_spec": boundary.spec, "boundary": boundary.state_dict(),
                      "boundary_step": step, "boundary_validation": scores,
                      "boundary_input_contract": "saved_cleaned_grayscale_png_only"}
            torch.save(bundle, out/"pipeline_last.pt")
            if scores["iou"] > best_iou:
                best_iou = scores["iou"]; torch.save(bundle, out/"pipeline_best.pt")
            write_json(out/"boundary_history.json", logs)
    baseline = None
    if getattr(args, "baseline", None):
        from restore_spray import load_bundle
        baseline, _, _ = load_bundle(args.baseline, device)
    heldout = evaluate_optical(restorer, data, source, device, seed=684203, batches=12, baseline=baseline)
    write_json(out/"optical_heldout_test.json", heldout)
    summary = {"version": RESTORE_VERSION, "restore_kind": "optical", "cleaner_steps": 0 if resume else args.steps,
               "inherited_cleaner_step": state["cleaner_step"] if resume else None,
               "cleaner_parent_sha256": hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
               "boundary_steps": args.boundary_steps, "cleaner_seconds": cleaner_seconds,
               "total_seconds": time.perf_counter()-start, "best_cleaner": state["cleaner_validation"],
               "best_synthetic_boundary_iou": best_iou, "parameters": sum(p.numel() for p in restorer.parameters()),
               "seed": args.seed, "size": synth.size, "batch": args.batch, "learning_rate": args.lr,
               "gain": state["gain"], "human_labels": 0, "real_accuracy_available": False,
               "boundary_uses_raw_input": False,
               "peak_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None,
               "manifest_sha256": hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest(),
               "assumption": "简化线性背景透过与前景混合，真实喷雾适用性未标注验证"}
    write_json(out/"training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
