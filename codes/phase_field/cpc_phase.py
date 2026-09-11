from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .data import load_array


def _surface_basis(u: np.ndarray, v: np.ndarray, order: int) -> np.ndarray:
    """Complete two-dimensional polynomial basis with total degree <= order."""
    columns: list[np.ndarray] = []
    for u_power in range(order + 1):
        for v_power in range(order + 1 - u_power):
            columns.append((u**u_power) * (v**v_power))
    return np.stack(columns, axis=1)


def cpc_fit_phase(
    phase: np.ndarray,
    mask: np.ndarray,
    order: int = 2,
    ridge: float = 1e-12,
    chunk_pixels: int = 200_000,
) -> np.ndarray:
    """
    Fit the unwrapped phase surface from mask-valid (white) pixels.

    This is the Python equivalent of the traditional CPC step. Pixel coordinates
    are normalized before fitting for numerical stability; the represented
    polynomial surface is unchanged.
    """
    phase = np.asarray(phase, dtype=np.float64).squeeze()
    mask = np.asarray(mask).squeeze().astype(bool)
    if phase.ndim != 2 or mask.shape != phase.shape:
        raise ValueError("phase 和 mask 必须是尺寸相同的 HxW 数组")
    if order < 1 or order > 5:
        raise ValueError("CPC 曲面次数必须在 [1,5] 内")
    if chunk_pixels < 1:
        raise ValueError("chunk_pixels 必须大于 0")

    height, width = phase.shape
    coefficient_count = (order + 1) * (order + 2) // 2
    valid = mask.ravel() & np.isfinite(phase.ravel())
    if int(valid.sum()) < coefficient_count:
        raise ValueError(
            f"mask 内有效像素 {int(valid.sum())} 少于 CPC 系数数 {coefficient_count}"
        )

    gram = np.zeros((coefficient_count, coefficient_count), dtype=np.float64)
    rhs = np.zeros(coefficient_count, dtype=np.float64)
    phase_flat = phase.ravel()
    pixel_count = phase.size
    u_scale = max((width - 1) / 2.0, 1.0)
    v_scale = max((height - 1) / 2.0, 1.0)

    for start in range(0, pixel_count, chunk_pixels):
        end = min(start + chunk_pixels, pixel_count)
        indices = np.arange(start, end, dtype=np.int64)
        u = ((indices % width) - (width - 1) / 2.0) / u_scale
        v = ((indices // width) - (height - 1) / 2.0) / v_scale
        basis = _surface_basis(u, v, order)
        selected = valid[start:end]
        selected_basis = basis[selected]
        selected_phase = phase_flat[start:end][selected]
        gram += selected_basis.T @ selected_basis
        rhs += selected_basis.T @ selected_phase

    gram += float(ridge) * np.eye(coefficient_count, dtype=np.float64)
    coefficients = np.linalg.solve(gram, rhs)
    fitted = np.empty(pixel_count, dtype=np.float32)
    for start in range(0, pixel_count, chunk_pixels):
        end = min(start + chunk_pixels, pixel_count)
        indices = np.arange(start, end, dtype=np.int64)
        u = ((indices % width) - (width - 1) / 2.0) / u_scale
        v = ((indices // width) - (height - 1) / 2.0) / v_scale
        fitted[start:end] = (_surface_basis(u, v, order) @ coefficients).astype(
            np.float32
        )
    return fitted.reshape(height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 mask 对原始展开相位进行 CPC 曲面拟合")
    parser.add_argument("--phase", required=True, help="原始展开相位 .mat/.npy")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-12)
    parser.add_argument("--chunk-pixels", type=int, default=200_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    phase = load_array(args.phase, "phase")
    mask = load_array(args.mask)
    fitted = cpc_fit_phase(
        phase,
        mask,
        order=args.order,
        ridge=args.ridge,
        chunk_pixels=args.chunk_pixels,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "CPC polynomial surface fit",
        "phase": str(Path(args.phase).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "order": args.order,
    }
    savemat(
        output,
        {
            "phase": fitted,
            "cpc_metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        },
        do_compression=True,
    )
    print(f"CPC 拟合相位已保存：{output}")


if __name__ == "__main__":
    main()
