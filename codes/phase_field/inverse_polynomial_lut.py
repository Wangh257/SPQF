from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InversePolynomialLUT:
    """每像素逆多项式：``ZC = 1 / P(q)``。"""

    denominator_coefficients: np.ndarray  # H,W,degree+1，常数项到最高次项
    phase_center: float
    phase_scale: float
    metadata: dict[str, Any]
    denominator_epsilon: float = 1e-12

    @property
    def degree(self) -> int:
        return int(self.denominator_coefficients.shape[-1] - 1)

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.denominator_coefficients.shape[:2])

    def predict_z(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=np.float32).squeeze()
        if phase.ndim == 0 and self.image_size == (1, 1):
            phase = phase.reshape(1, 1)
        if phase.shape != self.image_size:
            raise ValueError(f"相位尺寸 {phase.shape} 与 LUT 尺寸 {self.image_size} 不一致")
        q = (phase - self.phase_center) / self.phase_scale
        denominator = np.asarray(
            self.denominator_coefficients[..., -1], dtype=np.float64
        )
        for power in range(self.degree - 1, -1, -1):
            denominator = (
                denominator * q + self.denominator_coefficients[..., power]
            )
        valid = (
            np.isfinite(phase)
            & np.isfinite(denominator)
            & (denominator > float(self.denominator_epsilon))
        )
        depth = np.full(phase.shape, np.nan, dtype=np.float32)
        depth[valid] = (1.0 / denominator[valid]).astype(np.float32)
        return depth

    def save(self, path: str | Path) -> Path:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            denominator_coefficients=self.denominator_coefficients.astype(np.float32),
            phase_center=np.float64(self.phase_center),
            phase_scale=np.float64(self.phase_scale),
            denominator_epsilon=np.float64(self.denominator_epsilon),
            metadata_json=np.array(json.dumps(self.metadata, ensure_ascii=False)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "InversePolynomialLUT":
        with np.load(Path(path).resolve(), allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            epsilon = (
                float(source["denominator_epsilon"])
                if "denominator_epsilon" in source
                else 1e-12
            )
            return cls(
                denominator_coefficients=np.asarray(
                    source["denominator_coefficients"], dtype=np.float32
                ),
                phase_center=float(source["phase_center"]),
                phase_scale=float(source["phase_scale"]),
                metadata=metadata,
                denominator_epsilon=epsilon,
            )


def fit_inverse_polynomial_lut(
    phase_stack: np.ndarray,
    depth_stack_mm: np.ndarray,
    degree: int = 3,
    ridge: float = 1e-8,
    chunk_pixels: int = 100_000,
    metadata: dict[str, Any] | None = None,
    valid_stack: np.ndarray | None = None,
) -> InversePolynomialLUT:
    """逐像素最小二乘拟合 ``1/ZC = sum(a_k*q^k)``。"""
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
    observation_valid = (
        supplied_valid & np.isfinite(phase) & np.isfinite(depth) & (depth > 0)
    )
    finite_phase = phase[observation_valid]
    if finite_phase.size == 0:
        raise ValueError("没有相位和正深度均有效的观测")
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
        target_depth = depth_flat[:, start:end].T.astype(np.float64)
        valid = (
            supplied_valid_flat[:, start:end].T
            & np.isfinite(phi)
            & np.isfinite(target_depth)
            & (target_depth > 0)
        )
        q = np.where(valid, (phi - phase_center) / phase_scale, 0.0)
        powers = np.stack(
            [q**power for power in range(coefficient_count)], axis=-1
        )
        weighted_powers = np.where(valid[..., None], powers, 0.0)
        reciprocal_depth = np.zeros_like(target_depth, dtype=np.float64)
        np.divide(1.0, target_depth, out=reciprocal_depth, where=valid)
        gram = np.einsum("cnp,cnq->cpq", weighted_powers, weighted_powers)
        rhs = np.einsum("cnp,cn->cp", weighted_powers, reciprocal_depth)
        gram += float(ridge) * identity[None, ...]
        solved = np.linalg.solve(gram, rhs)
        solved[valid.sum(axis=1) < coefficient_count] = np.nan
        coefficients[start:end] = solved.astype(np.float32)
        print(f"fit inverse polynomial pixels: {end}/{height * width}")

    info = dict(metadata or {})
    info.update(
        {
            "method": "per_pixel_inverse_polynomial_lut",
            "polynomial_form": "ZC=1/(a0+a1*q+...+a_degree*q^degree)",
            "degree": int(degree),
            "plane_count": int(plane_count),
            "image_size": [int(height), int(width)],
            "uses_mask_for_fit": uses_mask_for_fit,
            "uses_spatial_phase_fitting": False,
            "phase_basis": "q=(phase-phase_center)/phase_scale",
            "depth_definition": "camera_axis_Z",
            "depth_unit": "mm",
        }
    )
    return InversePolynomialLUT(
        coefficients.reshape(height, width, coefficient_count),
        phase_center,
        phase_scale,
        info,
    )
