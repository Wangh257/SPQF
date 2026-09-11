from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class PolynomialLUT:
    """每像素的可选次数 ``phase -> ZC`` 多项式。"""

    coefficients_z: np.ndarray  # H,W,degree+1; 从常数项到最高次项
    phase_center: float
    phase_scale: float
    metadata: dict[str, Any]

    @property
    def degree(self) -> int:
        return int(self.coefficients_z.shape[-1] - 1)

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.coefficients_z.shape[:2])

    def predict_z(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=np.float32).squeeze()
        if phase.ndim == 0 and self.image_size == (1, 1):
            phase = phase.reshape(1, 1)
        if phase.shape != self.image_size:
            raise ValueError(f"相位尺寸 {phase.shape} 与 LUT 尺寸 {self.image_size} 不一致")
        q = (phase - self.phase_center) / self.phase_scale
        result = np.asarray(self.coefficients_z[..., -1], dtype=np.float64)
        for power in range(self.degree - 1, -1, -1):
            result = result * q + self.coefficients_z[..., power]
        result = result.astype(np.float32)
        result[~np.isfinite(phase)] = np.nan
        return result

    def save(
        self,
        path: str | Path,
        xn: np.ndarray | None = None,
        yn: np.ndarray | None = None,
    ) -> Path:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "coefficients_z": self.coefficients_z.astype(np.float32),
            "phase_center": np.float64(self.phase_center),
            "phase_scale": np.float64(self.phase_scale),
            "metadata_json": np.array(json.dumps(self.metadata, ensure_ascii=False)),
        }
        if xn is not None and yn is not None:
            coeff = self.coefficients_z.astype(np.float32)
            payload["coefficients_xyz"] = np.stack(
                (xn[..., None] * coeff, yn[..., None] * coeff, coeff), axis=0
            ).astype(np.float32)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PolynomialLUT":
        with np.load(Path(path).resolve(), allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            return cls(
                coefficients_z=np.asarray(source["coefficients_z"], dtype=np.float32),
                phase_center=float(source["phase_center"]),
                phase_scale=float(source["phase_scale"]),
                metadata=metadata,
            )


def fit_polynomial_lut(
    phase_stack: np.ndarray,
    depth_stack_mm: np.ndarray,
    degree: int = 3,
    ridge: float = 1e-8,
    chunk_pixels: int = 100_000,
    metadata: dict[str, Any] | None = None,
    valid_stack: np.ndarray | None = None,
    uses_spatial_phase_fitting: bool = False,
) -> PolynomialLUT:
    """
    逐像素拟合多项式，不对 phase 做空间拟合。

    ``valid_stack=None`` 时为完全无 mask 拟合；传入与相位同尺寸的
    bool 数组时，只使用 mask 为 True 的标定观测。

    为改善高次数值条件，实际基函数为
    ``q=(phase-phase_center)/phase_scale`` 的幂，与原相位多项式等价。
    """
    phase = np.asarray(phase_stack, dtype=np.float32)
    depth = np.asarray(depth_stack_mm, dtype=np.float32)
    if phase.ndim != 3 or phase.shape != depth.shape:
        raise ValueError("phase_stack 和 depth_stack_mm 必须是相同的 (N,H,W)")
    if valid_stack is None:
        supplied_valid = np.ones(phase.shape, dtype=bool)
        uses_mask_for_fit = False
    else:
        supplied_valid = np.asarray(valid_stack, dtype=bool)
        if supplied_valid.shape != phase.shape:
            raise ValueError("valid_stack 必须与 phase_stack 尺寸相同")
        uses_mask_for_fit = True
    plane_count, height, width = phase.shape
    if degree < 1 or degree >= plane_count:
        raise ValueError(f"多项式次数必须在 [1,{plane_count - 1}] 内")
    observation_valid = supplied_valid & np.isfinite(phase) & np.isfinite(depth)
    finite_phase = phase[observation_valid]
    if finite_phase.size == 0:
        raise ValueError("没有有效相位")
    phase_center = float(finite_phase.mean(dtype=np.float64))
    phase_scale = max(float(finite_phase.std(dtype=np.float64)), 1e-8)
    phase_flat = phase.reshape(plane_count, -1)
    depth_flat = depth.reshape(plane_count, -1)
    supplied_valid_flat = supplied_valid.reshape(plane_count, -1)
    coefficient_count = degree + 1
    coefficients = np.full(
        (height * width, coefficient_count), np.nan, dtype=np.float32
    )
    identity = np.eye(coefficient_count, dtype=np.float64)

    for start in range(0, height * width, chunk_pixels):
        end = min(start + chunk_pixels, height * width)
        phi = phase_flat[:, start:end].T.astype(np.float64)
        target = depth_flat[:, start:end].T.astype(np.float64)
        valid = (
            supplied_valid_flat[:, start:end].T & np.isfinite(phi) & np.isfinite(target)
        )
        q = np.where(valid, (phi - phase_center) / phase_scale, 0.0)
        powers = np.stack([q**power for power in range(coefficient_count)], axis=-1)
        weighted_powers = np.where(valid[..., None], powers, 0.0)
        safe_target = np.where(valid, target, 0.0)
        gram = np.einsum("cnp,cnq->cpq", weighted_powers, weighted_powers)
        rhs = np.einsum("cnp,cn->cp", weighted_powers, safe_target)
        gram += ridge * identity[None, ...]
        solved = np.linalg.solve(gram, rhs)
        counts = valid.sum(axis=1)
        solved[counts < coefficient_count] = np.nan
        coefficients[start:end] = solved.astype(np.float32)
        print(f"fit pixels: {end}/{height * width}")

    info = dict(metadata or {})
    info.update(
        {
            "method": "per_pixel_polynomial_lut",
            "degree": degree,
            "plane_count": plane_count,
            "image_size": [height, width],
            "uses_mask_for_fit": uses_mask_for_fit,
            "uses_spatial_phase_fitting": uses_spatial_phase_fitting,
            "phase_basis": "q=(phase-phase_center)/phase_scale",
            "depth_definition": "camera_axis_Z",
            "depth_unit": "mm",
        }
    )
    return PolynomialLUT(
        coefficients.reshape(height, width, coefficient_count),
        phase_center,
        phase_scale,
        info,
    )
