"""中文桌面入口：一次标定后批量识别整个图像序列。"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from spray_recognition import auto_geometry, discover_sequence


HERE = Path(__file__).resolve().parent


class SprayTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("喷雾识别工具 V2 / V3深度学习试验")
        self.geometry(f"{min(1220, self.winfo_screenwidth()-80)}x{min(880, self.winfo_screenheight()-100)}")
        self.minsize(960, 680)
        self.config_data = json.loads((HERE / ("config.json" if (HERE / "config.json").exists() else "config.example.json")).read_text(encoding="utf-8"))
        self.config_data.update(source_xy=None, chamber_xy=None, chamber_axes_px=None, radial_bins=None, mm_per_pixel=None)
        self.variables = {key: tk.StringVar(value=value) for key, value in {
            "source_dir": "", "output_dir": "", "liquid_name": "柴油", "background_frames": "5", "fps": "10000",
            "known_diameter_mm": "60", "ambient_pressure_bar": "40", "threshold_gray": "10",
            "physics_weight": "0.14", "liquid_density_kg_m3": "", "injection_pressure_MPa_label": "140", "velocity_coefficient": "",
            "deep_checkpoint": str(HERE / "models" / "deep_spray_v3_alpha2.pt") if (HERE / "models" / "deep_spray_v3_alpha2.pt").is_file() else "",
        }.items()}
        self.pressure_confirmed = tk.BooleanVar(value=False)
        self.consecutive = tk.BooleanVar(value=True)
        self.polarity = tk.StringVar(value="自动")
        self.engine = tk.StringVar(value="V2图像证据")
        self.loaded_source = None
        self.status = tk.StringVar(value="选择一个图像序列文件夹，加载预览并检查一次标定。")
        self.geometry_info = tk.StringVar(value="尚未加载图像")
        self.process = None
        self.events = queue.Queue()
        self.preview_image = None
        self.image_reference = None
        self.click_mode = None
        self.circle_points = []
        self._build()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        ttk.Style().configure("TLabel", font=("Microsoft YaHei UI", 10))
        ttk.Style().configure("TButton", font=("Microsoft YaHei UI", 10))
        outer = ttk.Frame(self, padding=15); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="喷雾物理约束识别", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="选择序列 → 一次标定 → 自动识别全序列 → 导出轮廓、掩膜与物理量").pack(anchor="w", pady=(4, 12))
        paths = ttk.Frame(outer); paths.pack(fill="x")
        for row, (key, label) in enumerate([("source_dir", "输入图像文件夹"), ("output_dir", "结果保存位置")]):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(paths, textvariable=self.variables[key]).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(paths, text="选择…", command=lambda k=key: self.browse(k)).grid(row=row, column=2)
        paths.columnconfigure(1, weight=1)
        quick = ttk.Frame(outer); quick.pack(fill="x", pady=(8, 0))
        ttk.Button(quick, text="载入本机60mm示例", command=lambda: self.load_settings(HERE / "config.json"),
                   state="normal" if (HERE / "config.json").exists() else "disabled").pack(side="left")
        ttk.Button(quick, text="载入本机V3示例", command=lambda: self.load_settings(HERE / "config.deep.local.json"),
                   state="normal" if (HERE / "config.deep.local.json").exists() else "disabled").pack(side="left", padx=6)
        ttk.Button(quick, text="加载设置…", command=self.load_settings).pack(side="left", padx=6)
        ttk.Button(quick, text="保存设置…", command=self.save_settings).pack(side="left")
        deep_row = ttk.Frame(outer); deep_row.pack(fill="x", pady=(8, 0))
        ttk.Label(deep_row, text="识别算法：").pack(side="left")
        ttk.Combobox(deep_row, textvariable=self.engine, values=["V2图像证据", "V3深度学习试验"], state="readonly", width=19).pack(side="left")
        ttk.Label(deep_row, text="  V3权重：").pack(side="left")
        ttk.Entry(deep_row, textvariable=self.variables["deep_checkpoint"]).pack(side="left", fill="x", expand=True)
        ttk.Button(deep_row, text="选择权重…", command=self.browse_checkpoint).pack(side="left", padx=(6, 0))
        middle = ttk.Frame(outer); middle.pack(fill="both", expand=True, pady=10)
        # 参数较多时只滚动左侧设置，保留运行按钮和图像预览可见。
        sidebar = ttk.Frame(middle); sidebar.pack(side="left", fill="y", padx=(0, 12))
        setting_canvas = tk.Canvas(sidebar, width=290, height=330, highlightthickness=0)
        setting_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=setting_canvas.yview)
        setting_canvas.configure(yscrollcommand=setting_scroll.set)
        setting_canvas.pack(side="left", fill="both", expand=True); setting_scroll.pack(side="right", fill="y")
        settings = ttk.LabelFrame(setting_canvas, text="整段序列共用的设置", padding=10)
        settings_window = setting_canvas.create_window((0, 0), window=settings, anchor="nw", width=285)
        settings.bind("<Configure>", lambda event: setting_canvas.configure(scrollregion=setting_canvas.bbox("all")))
        setting_canvas.bind("<Configure>", lambda event: setting_canvas.itemconfigure(settings_window, width=event.width))
        fields = [("liquid_name", "喷射液体（工况说明）"), ("background_frames", "起喷前背景帧数"), ("fps", "采样率（帧/秒）"),
                  ("known_diameter_mm", "标定圆直径（mm，可留空）"), ("ambient_pressure_bar", "环境背压（bar）"),
                  ("threshold_gray", "图像变化阈值（灰度）"), ("physics_weight", "增长模型权重"),
                  ("liquid_density_kg_m3", "液体密度（kg/m³，可留空）"),
                  ("injection_pressure_MPa_label", "喷射压力（MPa）"), ("velocity_coefficient", "速度系数Cv（可留空）")]
        for row, (key, label) in enumerate(fields):
            ttk.Label(settings, text=label).grid(row=row*2, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(settings, textvariable=self.variables[key], width=30).grid(row=row*2+1, column=0, sticky="ew")
        ttk.Checkbutton(settings, text="喷射压力及压力基准已确认", variable=self.pressure_confirmed).grid(row=2*len(fields), column=0, sticky="w", pady=7)
        ttk.Label(settings, text="密度、尺寸、Cv与压力齐全时，\n启用伯努利速度上界。\n绿色=证据；橙色=包络；紫色=未知。", foreground="#555555").grid(row=2*len(fields)+1, column=0, sticky="w", pady=6)
        right = ttk.Frame(middle); right.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(right); controls.pack(fill="x")
        ttk.Button(controls, text="加载 / 自动估计", command=self.load_preview).pack(side="left", padx=(0, 5))
        ttk.Button(controls, text="标喷嘴（1点）", command=lambda: self.set_mode("source")).pack(side="left", padx=3)
        ttk.Button(controls, text="标定圆周（3点）", command=lambda: self.set_mode("circle")).pack(side="left", padx=3)
        options = ttk.Frame(right); options.pack(fill="x", pady=(8, 0))
        ttk.Label(options, text="喷雾相对背景：").pack(side="left")
        ttk.Combobox(options, textvariable=self.polarity, values=["自动", "亮喷雾", "暗喷雾"], state="readonly", width=8).pack(side="left")
        ttk.Checkbutton(options, text="图像为连续导出帧（用于速度计算）", variable=self.consecutive).pack(side="left", padx=12)
        self.canvas = tk.Canvas(right, background="#20252b", highlightthickness=0, height=300)
        self.canvas.pack(fill="both", expand=True, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda event: self.draw_preview())
        ttk.Label(right, textvariable=self.geometry_info, wraplength=760).pack(anchor="w")
        buttons = ttk.Frame(outer); buttons.pack(fill="x", pady=5)
        self.run_button = ttk.Button(buttons, text="自动识别整个序列", command=lambda: self.run(False)); self.run_button.pack(side="left")
        self.batch_button = ttk.Button(buttons, text="批量处理子文件夹", command=lambda: self.run(True)); self.batch_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="取消运行", command=self.cancel).pack(side="left")
        ttk.Button(buttons, text="打开结果文件夹", command=self.open_results).pack(side="right")
        ttk.Label(outer, textvariable=self.status, foreground="#1e5d87").pack(anchor="w", pady=(4, 6))
        self.log = tk.Text(outer, height=5, wrap="word", font=("Consolas", 9), background="#f5f6f7")
        self.log.pack(fill="x")

    def browse(self, key):
        path = filedialog.askdirectory(title="选择文件夹")
        if path:
            self.variables[key].set(path)
            if key == "source_dir":
                self.config_data.update(source_xy=None, chamber_xy=None, chamber_axes_px=None, mm_per_pixel=None)
                self.loaded_source = None; self.preview_image = None
                self.canvas.delete("all"); self.geometry_info.set("请重新加载新序列的预览与标定。")
                if not self.variables["output_dir"].get():
                    self.variables["output_dir"].set(str(Path(path).parent / (Path(path).name + "_识别结果")))

    def get_config(self):
        cfg = dict(self.config_data)
        for key, var in self.variables.items():
            value = var.get().strip()
            if key in {"source_dir", "output_dir", "liquid_name", "deep_checkpoint"}:
                cfg[key] = value
            else:
                cfg[key] = float(value) if value else None
        cfg["background_frames"] = int(cfg["background_frames"] or 5)
        cfg["injection_pressure_confirmed"] = self.pressure_confirmed.get()
        cfg["consecutive_frames_assumed"] = self.consecutive.get()
        cfg["signal_polarity"] = {"自动": "auto", "亮喷雾": "bright", "暗喷雾": "dark"}[self.polarity.get()]
        cfg["engine"] = "deep" if self.engine.get() == "V3深度学习试验" else "evidence"
        cfg["radial_bins"] = None
        cfg["selected_frames"] = [1, 6, 10, 15, 20, 30, 50, 70]
        if cfg["fps"] is None or cfg["fps"] <= 0 or cfg["background_frames"] < 2:
            raise ValueError("采样率必须大于0；背景帧数至少为2。")
        if cfg["threshold_gray"] is None or cfg["threshold_gray"] <= 0 or cfg["physics_weight"] is None or cfg["physics_weight"] < 0:
            raise ValueError("灰度阈值必须大于0；模型权重不能为负。")
        if cfg.get("known_diameter_mm") is not None and cfg["known_diameter_mm"] <= 0:
            raise ValueError("标定直径必须大于0，或留空。")
        if cfg.get("velocity_coefficient") is not None and not 0 < cfg["velocity_coefficient"] <= 1:
            raise ValueError("速度系数Cv必须大于0且不超过1，或留空。")
        if cfg.get("liquid_density_kg_m3") is not None and cfg["liquid_density_kg_m3"] <= 0:
            raise ValueError("液体密度必须大于0，或留空。")
        if cfg.get("known_diameter_mm") and cfg.get("chamber_axes_px"):
            cfg["mm_per_pixel"] = cfg["known_diameter_mm"] / (2 * np.sqrt(np.prod(cfg["chamber_axes_px"])))
        else:
            cfg["mm_per_pixel"] = None
        return cfg

    def load_preview(self, reestimate=True):
        try:
            cfg = self.get_config()
            files, excluded = discover_sequence(cfg["source_dir"])
            if len(files) < cfg["background_frames"] + 10:
                raise ValueError("图像过少，需要足够的起喷前图像及早期发展图像。")
            if reestimate or not all(cfg.get(key) for key in ["source_xy", "chamber_xy", "chamber_axes_px"]):
                geometry = auto_geometry(files, cfg["background_frames"])
                self.config_data.update(geometry)
                self.config_data["calibration_source"] = "界面自动估计视窗；需检查预览或一次修正"
            with Image.open(files[min(len(files)-1, 19)]) as image:
                self.preview_image = image.convert("RGB")
            self.loaded_source = Path(cfg["source_dir"]).resolve()
            self.status.set(f"已读取 {len(files)} 帧；排除 {len(excluded)} 张不同分辨率附件。可一次修正喷嘴和标定圆。")
            self.draw_preview()
        except Exception as exc:
            messagebox.showerror("无法加载", str(exc))

    def draw_preview(self):
        if self.preview_image is None:
            return
        try:
            cfg = self.get_config()
        except (ValueError, TypeError):
            return
        im = self.preview_image.copy(); draw = ImageDraw.Draw(im)
        sx, sy = cfg["source_xy"]; cx, cy = cfg["chamber_xy"]; ax, ay = cfg["chamber_axes_px"]
        draw.ellipse((cx-ax, cy-ay, cx+ax, cy+ay), outline="#ff5b5b", width=3)
        draw.line((sx-10, sy, sx+10, sy), fill="#56bfff", width=3)
        draw.line((sx, sy-10, sx, sy+10), fill="#56bfff", width=3)
        for x, y in self.circle_points:
            draw.ellipse((x-5, y-5, x+5, y+5), fill="#ffe16e")
        cw, ch = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        self.display_scale = min(cw / im.width, ch / im.height)
        resized = im.resize((int(im.width*self.display_scale), int(im.height*self.display_scale)))
        self.display_offset = ((cw-resized.width)/2, (ch-resized.height)/2)
        self.image_reference = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw/2, ch/2, image=self.image_reference)
        scale_text = f"{cfg['mm_per_pixel']:.5f} mm/px" if cfg.get("mm_per_pixel") else "未输入尺寸，输出像素单位"
        self.geometry_info.set(f"喷嘴中心 ({sx:.1f}, {sy:.1f})；标定圆等效直径 {2*np.sqrt(ax*ay):.1f} px；{scale_text}。红圈/蓝十字需与图像吻合。")

    def set_mode(self, mode):
        if self.preview_image is None:
            self.load_preview()
        if self.preview_image is not None:
            self.click_mode = mode; self.circle_points = []
            self.status.set("在预览上点击喷嘴中心一次。" if mode == "source" else "在同一标定圆周上点击相隔较远的三个点。")

    def on_click(self, event):
        if self.preview_image is None or not self.click_mode:
            return
        x = (event.x-self.display_offset[0]) / self.display_scale
        y = (event.y-self.display_offset[1]) / self.display_scale
        if not (0 <= x < self.preview_image.width and 0 <= y < self.preview_image.height):
            return
        if self.click_mode == "source":
            self.config_data["source_xy"] = [x, y]; self.click_mode = None
            self.status.set("喷嘴中心已设置；整段序列共用。")
        else:
            self.circle_points.append([x, y])
            if len(self.circle_points) == 3:
                p = np.asarray(self.circle_points)
                try:
                    center = np.linalg.solve(2*(p[1:]-p[0]), (p[1:]**2-p[0]**2).sum(1))
                    radius = float(np.linalg.norm(p[0]-center))
                    if radius < 20 or radius > max(self.preview_image.size):
                        raise ValueError("拟合圆尺寸异常")
                    self.config_data.update(chamber_xy=center.tolist(), chamber_axes_px=[radius, radius], calibration_source="界面原图圆周三点标定")
                    self.status.set("圆周标定已完成；整段序列共用。")
                except (np.linalg.LinAlgError, ValueError):
                    messagebox.showerror("标定失败", "三点接近共线或标定圆异常，请重新选取。")
                self.circle_points = []; self.click_mode = None
        self.draw_preview()

    def save_settings(self):
        try:
            cfg = self.get_config()
            target = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("配置文件", "*.json")])
            if target:
                Path(target).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("无法保存", str(exc))

    def load_settings(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("配置文件", "*.json")])
        if not path:
            return
        try:
            path = Path(path)
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if cfg.get("output_dir"):
                cfg["output_dir"] = str((path.parent / cfg["output_dir"]).resolve())
            self.config_data.update(cfg)
            for key, var in self.variables.items():
                value = cfg.get(key)
                var.set("" if value is None else str(value))
            self.pressure_confirmed.set(cfg.get("injection_pressure_confirmed", False))
            self.consecutive.set(cfg.get("consecutive_frames_assumed", True))
            self.polarity.set({"auto": "自动", "bright": "亮喷雾", "dark": "暗喷雾"}.get(cfg.get("signal_polarity"), "自动"))
            self.engine.set("V3深度学习试验" if cfg.get("engine") == "deep" else "V2图像证据")
            self.load_preview(False)
        except Exception as exc:
            messagebox.showerror("无法加载配置", str(exc))

    def browse_checkpoint(self):
        path = filedialog.askopenfilename(title="选择已训练喷雾模型", filetypes=[("PyTorch权重", "*.pt")])
        if path:
            self.variables["deep_checkpoint"].set(path)

    def run(self, batch=False):
        if self.process and self.process.poll() is None:
            return
        try:
            cfg = self.get_config()
            if not cfg["source_dir"] or not cfg["output_dir"]:
                raise ValueError("请选择输入与输出文件夹。")
            if Path(cfg["source_dir"]).resolve() == Path(cfg["output_dir"]).resolve():
                raise ValueError("输入和输出文件夹不能相同。")
            if batch:
                cfg.update(source_xy=None, chamber_xy=None, chamber_axes_px=None, mm_per_pixel=None)
            elif self.loaded_source != Path(cfg["source_dir"]).resolve():
                # 手工在路径框输入新序列时，也不误用旧序列的几何标定。
                cfg.update(source_xy=None, chamber_xy=None, chamber_axes_px=None, mm_per_pixel=None)
            out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
            config_path = out / "工具运行设置.json"
            config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [sys.executable, "-u", str(HERE / "spray_recognition.py"), "--config", str(config_path)]
            if cfg["engine"] == "deep":
                if not Path(cfg["deep_checkpoint"]).is_file():
                    raise ValueError("请先选择V3已训练权重（.pt）。V2无需权重。")
                local_python = HERE.parent.parent / ".venv-spray-v3" / "Scripts" / "python.exe"
                interpreter = str(local_python) if local_python.is_file() else sys.executable
                command = [interpreter, "-u", str(HERE / "deep_spray.py"), "predict", "--config", str(config_path),
                           "--checkpoint", cfg["deep_checkpoint"]]
            if batch:
                command += ["--batch", "--input", cfg["source_dir"], "--output", cfg["output_dir"]]
            env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.run_button.config(state="disabled"); self.batch_button.config(state="disabled")
            self.status.set("正在自动处理整段序列…")
            self.log.delete("1.0", "end")
            threading.Thread(target=self._read_output, daemon=True).start()
        except Exception as exc:
            messagebox.showerror("无法运行", str(exc))

    def _read_output(self):
        for line in self.process.stdout:
            self.events.put(("log", line))
        self.events.put(("done", self.process.wait()))

    def _poll(self):
        while not self.events.empty():
            kind, value = self.events.get()
            if kind == "log":
                self.log.insert("end", value); self.log.see("end")
            else:
                self.run_button.config(state="normal"); self.batch_button.config(state="normal")
                self.status.set("处理完成，可打开结果文件夹。" if value == 0 else "处理未完成，请查看日志中的具体原因。")
        self.after(100, self._poll)

    def cancel(self):
        if self.process and self.process.poll() is None:
            self.process.terminate(); self.status.set("已请求停止，已经完成的结果保留。")

    def open_results(self):
        path = self.variables["output_dir"].get()
        if path and Path(path).exists():
            os.startfile(path)

    def close(self):
        self.cancel(); self.destroy()


if __name__ == "__main__":
    SprayTool().mainloop()
