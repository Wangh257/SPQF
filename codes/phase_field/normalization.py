from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    """只用训练平面计算的归一化参数。"""

    phase_mean: float
    phase_std: float
    phase_min: float
    phase_max: float
    depth_min_mm: float
    depth_max_mm: float
    amplitude_mean: float = 0.0
    amplitude_std: float = 1.0

    def normalize_phase(self, phase: np.ndarray) -> np.ndarray:
        return (np.asarray(phase, dtype=np.float32) - self.phase_mean) / self.phase_std

    def normalize_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        scale = self.depth_max_mm - self.depth_min_mm
        return (np.asarray(depth_mm, dtype=np.float32) - self.depth_min_mm) / scale

    def denormalize_depth(self, zn: np.ndarray) -> np.ndarray:
        scale = self.depth_max_mm - self.depth_min_mm
        return np.asarray(zn, dtype=np.float32) * scale + self.depth_min_mm

    def normalize_amplitude(self, amplitude: np.ndarray) -> np.ndarray:
        return (
            np.asarray(amplitude, dtype=np.float32) - self.amplitude_mean
        ) / self.amplitude_std

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizationStats":
        return cls(**{key: float(item) for key, item in value.items()})


class RunningStats:
    """可流式累积的标量均值、方差和范围。"""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_sq += float(np.square(array).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def finish(self) -> tuple[float, float, float, float]:
        if self.count == 0:
            raise ValueError("无有可用数据，无法计算归一化参数")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        std = max(float(np.sqrt(variance)), 1e-8)
        return mean, std, float(self.minimum), float(self.maximum)


def validate_stats(stats: NormalizationStats) -> None:
    values: Iterable[float] = stats.to_dict().values()
    if not all(np.isfinite(list(values))):
        raise ValueError("归一化参数包含 NaN/Inf")
    if stats.phase_std <= 0:
        raise ValueError("phase_std 必须大于 0")
    if stats.depth_max_mm <= stats.depth_min_mm:
        raise ValueError("depth_max_mm 必须大于 depth_min_mm")
    if stats.amplitude_std <= 0:
        raise ValueError("amplitude_std 必须大于 0")
