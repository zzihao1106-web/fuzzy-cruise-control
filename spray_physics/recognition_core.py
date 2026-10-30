"""V2：先保留图像证据，模型仅提出有局部支持的边界修正。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RecognitionResult:
    evidence: np.ndarray
    envelope: np.ndarray
    inferred: np.ndarray
    unknown: np.ndarray
    model_added: np.ndarray
    boundary: np.ndarray
    prediction: np.ndarray
    reliable: np.ndarray
    status: str
    measurement_valid: bool
    model_conflict_ratio: float


class EvidenceTracker:
    def __init__(self, roi, source_xy, threshold, angular_bins, radial_bins, max_step, config):
        self.roi = roi
        self.source = tuple(source_xy)
        self.threshold = threshold
        self.na = angular_bins
        self.nr = radial_bins
        self.max_step = max_step
        self.config = config
        yy, xx = np.indices(roi.shape)
        self.radius = np.hypot(xx-source_xy[0], yy-source_xy[1])
        self.angle = ((np.arctan2(yy-source_xy[1], xx-source_xy[0]) % (2*np.pi)) * angular_bins/(2*np.pi)).astype(int)
        self.seed = (self.radius < max(12, radial_bins*0.18)) & roi
        self.previous_evidence = np.zeros_like(roi)
        self.previous_boundary = np.zeros(angular_bins)
        self.seen_spray = False
        self.observed_count = 0
        self.min_pixels = max(24, int(roi.sum()*0.00012))

    def _polar(self, image):
        return cv2.warpPolar(image.astype(np.float32), (self.nr, self.na), self.source, self.nr,
                             cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR)

    def _mask(self, boundary):
        return (self.radius <= boundary[self.angle]) & (boundary[self.angle] > 0) & self.roi

    def process(self, residual, wall, predicted=None, model_strength=None, glare_fraction=None, before_onset=False):
        zero = np.zeros_like(self.roi)
        no_boundary = np.zeros(self.na)
        predicted = np.zeros(self.na) if predicted is None else np.minimum(np.maximum(predicted, 0), wall)
        if before_onset:
            return RecognitionResult(zero, zero, zero, zero, zero, no_boundary, predicted,
                                     np.zeros(self.na, bool), "background", True, 0.)

        weak = (residual > np.maximum(4, self.threshold*0.65)) & self.roi
        strong = (residual > self.threshold+4) & self.roi
        size = max(3, int(self.nr/120) | 1)
        connected = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        # 窗框反光常表现为薄环和细桥；先去细结构，再恢复附近的原始证据。
        # 喷嘴附近不做这项厚度筛选，避免删掉刚形成的短细喷束。
        thickness = max(3, int(self.config.get("min_structure_px", 9)*self.nr/360) | 1)
        structural = cv2.morphologyEx(connected, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness)))
        # 临近窗框的薄环采用更强的厚度检查，防止薄环连上某一喷束后沿圆周扩散。
        edge_size = max(thickness, int(self.config.get("rim_structure_px", 21)*self.nr/360) | 1)
        edge_structure = cv2.morphologyEx(connected, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_size, edge_size)))
        near_rim = self.radius > wall[self.angle]*0.85
        structural[near_rim] = edge_structure[near_rim]
        structural[self.seed] = connected[self.seed]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(structural, 8)
        previous_seed = cv2.dilate(self.previous_evidence.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        seed_ids = np.unique(labels[(self.seed | previous_seed) & strong])
        selected = np.zeros(count, bool)
        selected[seed_ids[seed_ids > 0]] = True
        selected &= stats[:, cv2.CC_STAT_AREA] >= max(6, int(self.nr/60))
        linked = selected[labels]
        restored = cv2.dilate(linked.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        evidence = restored & weak & (residual > self.threshold)
        # 视窗边缘的低幅残差易来自反光漂移；须有附近较强信号支持才能测量。
        outer_strong = (residual > self.threshold+self.config.get("rim_threshold_margin", 14)).astype(np.uint8)
        outer_strong = cv2.morphologyEx(outer_strong, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        outer_supported = cv2.dilate(outer_strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        ambiguous_rim = evidence & near_rim & ~outer_supported
        evidence &= ~near_rim | outer_supported

        # 没有当前证据时不产生测量，旧模型只能进入独立的未知输出。
        if int((evidence & strong).sum()) < self.min_pixels:
            unknown_boundary = np.maximum(self.previous_boundary, predicted) if self.seen_spray else no_boundary
            unknown = self._mask(unknown_boundary)
            status = "unknown_no_observation" if self.seen_spray else "background"
            return RecognitionResult(zero, zero, zero, unknown, zero, no_boundary, unknown_boundary,
                                     np.zeros(self.na, bool), status, not self.seen_spray, 0.)

        self.seen_spray = True
        self.observed_count += 1
        pol_mask = self._polar(evidence)
        rr = np.arange(self.nr)[None, :]
        # 前沿来自当前证据，不再忽略喷嘴附近固定半径，也不由幂律先验裁掉。
        data_boundary = np.max(np.where(pol_mask > 0.25, rr, 0), axis=1).astype(float)
        data_boundary = np.minimum(data_boundary+np.where(data_boundary > 0, 1, 0), wall)
        boundary = data_boundary.copy()
        pol_residual = self._polar(residual)
        pol_threshold = self._polar(self.threshold)
        model_strength = np.zeros(self.na) if model_strength is None else model_strength
        glare_fraction = np.zeros(self.na) if glare_fraction is None else glare_fraction
        max_gap = self.config.get("max_completion_px", 14)*self.nr/360
        if self.config.get("mm_per_pixel"):
            max_gap = min(max_gap, self.config.get("max_completion_mm", 1.5)/self.config["mm_per_pixel"])
        unknown_boundary = boundary.copy()
        model_proposals = np.zeros(self.na, bool)
        local_width = max(2, int(self.na*8/360))
        # 仅在有足够当前观测后启用模型，固定帧号不再决定角向形状。
        if self.observed_count >= 4:
            for angle in range(self.na):
                base = data_boundary[angle]
                if base < self.nr*0.12 or model_strength[angle] <= 0:
                    continue
                offsets = np.r_[np.arange(-local_width, -1), np.arange(2, local_width+1)]
                neighbors = data_boundary[(angle+offsets) % self.na]
                neighbors = neighbors[neighbors > self.nr*0.1]
                if len(neighbors) < local_width:
                    continue
                neighbor_limit = float(np.median(neighbors))
                # 当前邻角与增长模型必须同时允许更远的位置。
                target = min(predicted[angle], neighbor_limit, base+max_gap, wall[angle])
                if self.previous_boundary[angle] > 0:
                    target = min(target, self.previous_boundary[angle]+self.max_step)
                if target <= base+1:
                    continue
                start, stop = int(base)+1, int(target)+1
                band = pol_residual[angle, start:stop]
                band_threshold = pol_threshold[angle, start:stop]
                if not len(band):
                    continue
                weak_support = float(np.mean(band > np.maximum(4, band_threshold*0.65)))
                # 模型权重为零时保留纯图像基线；全无信号的缺口只输出未知。
                if self.config.get("physics_weight", 0) > 0 and weak_support >= 0.5:
                    strength = min(1., self.config["physics_weight"]/0.14) * model_strength[angle]
                    boundary[angle] = base+(target-base)*strength
                    model_proposals[angle] = boundary[angle] > base+0.5
                elif glare_fraction[angle] > 0.25:
                    unknown_boundary[angle] = target

        base_envelope = self._mask(data_boundary) | evidence
        envelope = self._mask(boundary) | evidence
        added = envelope & ~base_envelope
        unknown = (self._mask(np.maximum(unknown_boundary, boundary)) | ambiguous_rim) & ~envelope
        inferred = envelope & ~evidence
        reliable = np.zeros(self.na, bool)
        for angle, value in enumerate(boundary.astype(int)):
            if value <= 0:
                continue
            samples = pol_residual[angle, max(0, value-5):value+1]
            reliable[angle] = samples.mean() >= pol_threshold[angle, min(value, self.nr-1)] and not model_proposals[angle]
        observed = data_boundary > self.nr*0.1
        conflict = observed & (model_strength > 0) & (np.abs(data_boundary-predicted) > max_gap*2)
        self.previous_evidence = evidence.copy()
        self.previous_boundary = boundary.copy()
        return RecognitionResult(evidence, envelope, inferred, unknown, added, boundary, predicted, reliable,
                                 "observed", True, float(conflict.sum()/max(observed.sum(), 1)))
