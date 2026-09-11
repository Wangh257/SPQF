from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .data import FeatureSpec, full_image_features
from .models import PhaseToDepthMLP
from .normalization import NormalizationStats
from .utils import choose_device


class DirectMLPPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(choose_device(device))
        state: dict[str, Any] = torch.load(
            self.checkpoint_path, map_location=self.device
        )
        if state.get("method") != "direct_mlp":
            raise ValueError(f"{self.checkpoint_path} 不是 Direct MLP checkpoint")
        self.state = state
        self.stats = NormalizationStats.from_dict(state["normalization"])
        self.feature_spec = FeatureSpec(**state["feature_spec"])
        self.image_size = tuple(int(x) for x in state["image_size"])
        self.camera = state["camera"]
        self.phase_type = str(state["phase_type"])
        self.model = PhaseToDepthMLP(**state["model_config"])
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        phase: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        amplitude: Optional[np.ndarray] = None,
        chunk_size: int = 262_144,
        warn_out_of_range: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = np.asarray(phase, dtype=np.float32).squeeze()
        if phase.shape != self.image_size:
            raise ValueError(
                f"输入相位尺寸 {phase.shape} 与训练尺寸 {self.image_size} 不一致。"
                "不能在未同步修改内参的情况下 resize/crop。"
            )
        if self.feature_spec.use_amplitude and amplitude is None:
            raise ValueError("该 checkpoint 训练时使用了 amplitude，推理时必须提供")
        if amplitude is not None:
            amplitude = np.asarray(amplitude, dtype=np.float32).squeeze()
            if amplitude.shape != phase.shape:
                raise ValueError("amplitude 尺寸必须与 phase 一致")
        if valid_mask is None:
            valid = np.isfinite(phase)
        else:
            valid = np.asarray(valid_mask).squeeze().astype(bool) & np.isfinite(phase)
            if valid.shape != phase.shape:
                raise ValueError("valid_mask 尺寸必须与 phase 一致")
        if amplitude is not None:
            valid &= np.isfinite(amplitude)
        if not np.any(valid):
            raise ValueError("输入中没有有效相位像素")

        if warn_out_of_range:
            valid_phase = phase[valid]
            outside = (valid_phase < self.stats.phase_min) | (
                valid_phase > self.stats.phase_max
            )
            ratio = float(outside.mean())
            if ratio > 0:
                warnings.warn(
                    f"{ratio:.2%} 的有效相位超出训练范围 "
                    f"[{self.stats.phase_min:.6g}, {self.stats.phase_max:.6g}]，这些像素属于外推。",
                    stacklevel=2,
                )

        features = full_image_features(
            phase,
            self.stats,
            self.feature_spec,
            amplitude,
        )
        valid_flat = np.flatnonzero(valid.ravel())
        zn_flat = np.full(phase.size, np.nan, dtype=np.float32)
        for start in range(0, valid_flat.size, chunk_size):
            indices = valid_flat[start : start + chunk_size]
            tensor = torch.from_numpy(features[indices]).to(self.device)
            zn_flat[indices] = self.model(tensor).float().cpu().numpy()
        zn = zn_flat.reshape(phase.shape)
        depth = self.stats.denormalize_depth(zn)
        return zn, depth, valid
