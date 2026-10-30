"""本地三类稀疏标注：确定喷雾、确定背景与未知，不修改原始图像。"""

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

DEFAULT_FRAMES = [6, 10, 12, 14, 16, 18, 19, 20, 54, 70]


class Annotator(tk.Tk):
    def __init__(self, data, output):
        super().__init__()
        self.title("喷雾标注：1喷雾 / 2背景 / 3未知；左右切帧；S保存")
        self.data, self.output = Path(data), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.data/"manifest.json").read_text(encoding="utf-8"))
        self.frames = [i for i in DEFAULT_FRAMES if i <= len(self.manifest["files"])]
        self.index, self.category, self.previous = 0, 1, None
        self.brush = tk.IntVar(value=9)
        self.state_text = tk.StringVar()
        self.dirty = False
        buttons = ttk.Frame(self, padding=8); buttons.pack(fill="x")
        for text, command in [("喷雾 [1]", lambda: self.set_category(1)), ("背景 [2]", lambda: self.set_category(0)),
                              ("未知/擦除 [3]", lambda: self.set_category(255)), ("保存 [S]", self.save),
                              ("上一帧 [←]", lambda: self.move(-1)), ("下一帧 [→]", lambda: self.move(1))]:
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=3)
        ttk.Label(buttons, text="笔刷原图像素：").pack(side="left")
        ttk.Spinbox(buttons, from_=2, to=80, textvariable=self.brush, width=5).pack(side="left")
        ttk.Label(self, text="仅涂确定区域；看不清的边界保留未知。左侧可画，右侧为空背景参考。").pack(anchor="w", padx=10)
        self.canvas = tk.Canvas(self, width=1100, height=560, background="#20252b")
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(self, textvariable=self.state_text).pack(anchor="w", padx=10, pady=8)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda event: setattr(self, "previous", None))
        self.bind("1", lambda event: self.set_category(1)); self.bind("2", lambda event: self.set_category(0))
        self.bind("3", lambda event: self.set_category(255)); self.bind("s", lambda event: self.save())
        self.bind("<Left>", lambda event: self.move(-1)); self.bind("<Right>", lambda event: self.move(1))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.load()

    def label_path(self):
        entry = self.manifest["files"][self.frames[self.index]-1]
        return self.output/(Path(entry["name"]).stem+".png")

    def set_category(self, value):
        self.category = value
        self.draw()

    def load(self):
        entry = self.manifest["files"][self.frames[self.index]-1]
        self.image = Image.open(Path(self.manifest["source_dir"])/entry["name"]).convert("RGB")
        self.mask = np.asarray(Image.open(self.label_path()).convert("L")).copy() if self.label_path().exists() else np.full((self.image.height, self.image.width), 255, np.uint8)
        self.dirty = False
        self.previous = None
        self.scale = min(650/self.image.width, 500/self.image.height)
        self.draw()

    def draw(self):
        image = np.asarray(self.image).copy()
        for category, color in [(0, [40, 100, 255]), (1, [40, 240, 80])]:
            use = self.mask == category
            image[use] = (image[use]*0.60+np.array(color)*0.40).astype(np.uint8)
        size = (round(self.image.width*self.scale), round(self.image.height*self.scale))
        self.photo = ImageTk.PhotoImage(Image.fromarray(image).resize(size))
        self.background = ImageTk.PhotoImage(Image.open(self.data/"原始背景.png").convert("RGB").resize((400, round(400*self.image.height/self.image.width))))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.create_image(size[0]+15, 0, anchor="nw", image=self.background)
        text = {0: "确定背景", 1: "确定喷雾", 255: "未知/擦除"}[self.category]
        self.state_text.set(f"帧{self.frames[self.index]}；当前：{text}；已标喷雾{int((self.mask==1).sum())}px，背景{int((self.mask==0).sum())}px；{'未保存' if self.dirty else '已保存或尚无标注'}")

    def paint(self, event):
        x, y = int(event.x/self.scale), int(event.y/self.scale)
        if not (0 <= x < self.image.width and 0 <= y < self.image.height):
            self.previous = None
            return
        previous = self.previous or (x, y)
        cv2.line(self.mask, previous, (x, y), self.category, max(2, self.brush.get()))
        self.previous, self.dirty = (x, y), True
        self.draw()

    def save(self):
        Image.fromarray(self.mask).save(self.label_path())
        self.dirty = False
        self.draw()

    def move(self, amount):
        if self.dirty:
            self.save()
        self.index = (self.index+amount) % len(self.frames)
        self.load()

    def close(self):
        if self.dirty:
            self.save()
        self.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地喷雾三类标注，不需要深度学习环境")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    Annotator(options.data, options.output).mainloop()
