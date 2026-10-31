"""三帧合成配对监督；灰底模板不是有喷雾图的人工真值。"""

import cv2
import numpy as np

from deep_data import SpraySynthesizer


def temporal_channels(previous, image, following, background, gray):
    return np.stack([previous, image, following, background, gray,
                     (image-background)*4, (image-previous)*4,
                     (following-image)*4, (background-gray)*2]).astype(np.float32)


class GraySceneSynthesizer:
    def __init__(self, data, source, seed, procedural_only=False):
        self.synth = SpraySynthesizer(data/"backgrounds.npz", source, seed,
                                      procedural_only=procedural_only, curriculum=True)
        with np.load(data/"gray_reference.npz") as z:
            self.gray = z["gray"].astype(np.float32)

    def sample(self):
        rng = self.synth.rng
        x, _, layers = self.synth.sample(optical=True)
        alpha = layers["alpha"][0]
        background = x[1]
        offset = layers["background_offset"][0]
        current_bg = np.clip(background+offset, 0, 1)
        # 已知层构造灰底目标。噪声随原图保留，不做生成式纹理填补。
        gray = self.gray.copy()
        if self.synth.procedural_only or rng.random() < 0.3:
            gray = np.full_like(gray, rng.uniform(0.18, 0.40))
        appearance = np.where(alpha > 1e-5, layers["foreground"][0]/np.maximum(alpha, 1e-5), 0.5)
        kernel = np.ones((3, 3), np.uint8)
        prior_alpha = cv2.erode(alpha, kernel)*rng.uniform(0.95, 1.03)
        later_alpha = cv2.dilate(alpha, kernel)*rng.uniform(0.97, 1.04)
        # 允许后期消散及运动，不把整段强行限定为单调增长。
        if rng.random() < 0.25:
            prior_alpha, later_alpha = later_alpha, prior_alpha
        neighbors = []
        for a in (prior_alpha, later_alpha):
            dx, dy = rng.uniform(-1.2, 1.2, 2)
            a = cv2.warpAffine(a, np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                              (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_CONSTANT)
            a = np.clip(a, 0, 0.98)
            f = (1-a)*current_bg+a*appearance+rng.normal(0, 0.004, a.shape)
            neighbors.append(np.clip(f, 0, 1))
        image = x[0]
        target = image-(1-alpha)*(current_bg-gray)
        return temporal_channels(neighbors[0], image, neighbors[1], background, gray), {
            "alpha": alpha[None].astype(np.float32), "gray": target[None].astype(np.float32),
            "offset": offset[None].astype(np.float32)}

    def batch(self, count):
        samples = [self.sample() for _ in range(count)]
        return np.stack([s[0] for s in samples]), {
            key: np.stack([s[1][key] for s in samples]) for key in samples[0][1]}
