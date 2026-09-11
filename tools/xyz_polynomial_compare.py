#!/usr/bin/env python3
"""独立的 XC/YC/ZC 多项式 LUT 对比程序。

不修改、不调用 train.sh/test.sh 的模型逻辑；只读取已经准备好的相位。
三次时每个像素拟合：

    XC = ax0 + ax1*q + ax2*q^2 + ax3*q^3
    YC = ay0 + ay1*q + ay2*q^2 + ay3*q^3
    ZC = az0 + az1*q + az2*q^2 + az3*q^3

共 3 x 4 = 12 个参数。q 是相位的统一中心化/缩放，只用于数值稳定，
不改变多项式次数或表达能力。
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "codes"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from phase_field.data import CalibrationDataset, load_array  # noqa: E402


# ======================== 只需修改这里 ========================
# 直接指定提前处理好的相位。两种示例：
phase_dir = "data/raw_phase_no_mask"
# phase_dir = "data/raw_phase_cpc_mask_order2"
phase_pattern = "PSP_{index}.mat"

manifest = "codes/phase_field/data/calibration_manifest_raw_no_mask.json"
degree = 5  # degree=3 时每像素共 12 个 XC/YC/ZC 参数
output_dir = "results/xyz_polynomial_no_mask_degree5"

# 固定测试 16、17、19；其余 manifest 中 split=train 的标定板用于拟合。
test_indices = (16, 17, 19)  # 1-based

# True：MAE/RMSE 只统计 mask=True 区域；False：统计全部有效像素。
# mask 只影响评价，不参与多项式拟合。
use_mask = False
mask_dir = "cal/mask"
mask_pattern = "mask_{index}.bmp"

ridge = 1e-8
chunk_pixels = 20_000
# =============================================================


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _regression_metrics(error: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("没有可评价的有效数值")
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mae_mm": float(absolute.mean()),
        "rmse_mm": float(sqrt(np.square(values).mean())),
        "median_ae_mm": float(np.median(absolute)),
        "p95_ae_mm": float(np.percentile(absolute, 95)),
        "max_ae_mm": float(absolute.max()),
        "bias_mm": float(values.mean()),
    }


def _xyz_metrics(error_xyz: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error_xyz, dtype=np.float64).reshape(-1, 3)
    error = error[np.all(np.isfinite(error), axis=1)]
    if error.size == 0:
        raise ValueError("没有可评价的三维点")
    distance = np.linalg.norm(error, axis=1)
    return {
        "point_count": int(error.shape[0]),
        "XC": _regression_metrics(error[:, 0]),
        "YC": _regression_metrics(error[:, 1]),
        "ZC": _regression_metrics(error[:, 2]),
        "XYZ_components": _regression_metrics(error),
        "euclidean_3d": _regression_metrics(distance),
    }


def _predict_xyz(
    coefficients_xyz: np.ndarray,
    phase: np.ndarray,
    phase_center: float,
    phase_scale: float,
) -> np.ndarray:
    image_phase = np.asarray(phase, dtype=np.float64).squeeze()
    if image_phase.shape != tuple(coefficients_xyz.shape[1:3]):
        raise ValueError(
            f"相位尺寸 {image_phase.shape} 与模型尺寸 "
            f"{tuple(coefficients_xyz.shape[1:3])} 不一致"
        )
    q = (image_phase - phase_center) / phase_scale
    result = np.asarray(coefficients_xyz[..., -1], dtype=np.float64)
    for power in range(coefficients_xyz.shape[-1] - 2, -1, -1):
        result = result * q[None, ...] + coefficients_xyz[..., power]
    xyz = np.moveaxis(result, 0, -1).astype(np.float32)
    xyz[~np.isfinite(image_phase)] = np.nan
    return xyz


def _save_ascii_ply(path: Path, xyz: np.ndarray, valid: np.ndarray) -> int:
    points = np.asarray(xyz, dtype=np.float32)[valid]
    points = points[np.all(np.isfinite(points), axis=1)]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        np.savetxt(handle, points, fmt="%.7g %.7g %.7g")
    return int(points.shape[0])


def _load_training_stacks(
    dataset: CalibrationDataset,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    indices = dataset.indices("train")
    if not indices:
        raise ValueError("manifest 中没有 split=train 的标定板")
    phases: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    for index in indices:
        arrays = dataset.load(index)
        phases.append(np.asarray(arrays.phase, dtype=np.float32))
        depths.append(np.asarray(arrays.depth_mm, dtype=np.float32))
        print(f"load train PSP_{index + 1}")
    return np.stack(phases), np.stack(depths), indices


def fit_xyz_polynomial(
    phase_stack: np.ndarray,
    depth_stack: np.ndarray,
    xn: np.ndarray,
    yn: np.ndarray,
    polynomial_degree: int,
    ridge_value: float,
    chunk_size: int,
) -> tuple[np.ndarray, float, float]:
    """每像素独立最小二乘拟合 XC/YC/ZC；三次时为12参数。"""
    phase = np.asarray(phase_stack, dtype=np.float32)
    depth = np.asarray(depth_stack, dtype=np.float32)
    if phase.ndim != 3 or phase.shape != depth.shape:
        raise ValueError("phase_stack 和 depth_stack 必须是相同的 (N,H,W)")
    plane_count, height, width = phase.shape
    if polynomial_degree < 1 or polynomial_degree >= plane_count:
        raise ValueError(
            f"多项式次数必须在 [1,{plane_count - 1}] 之间"
        )
    if xn.shape != (height, width) or yn.shape != (height, width):
        raise ValueError("相机射线尺寸与相位不一致")

    observation_valid = np.isfinite(phase) & np.isfinite(depth) & (depth > 0)
    valid_phase = phase[observation_valid]
    if valid_phase.size == 0:
        raise ValueError("没有有效训练相位")
    phase_center = float(valid_phase.mean(dtype=np.float64))
    phase_scale = max(float(valid_phase.std(dtype=np.float64)), 1e-8)

    pixel_count = height * width
    coefficient_count = polynomial_degree + 1
    coefficients = np.full(
        (3, pixel_count, coefficient_count), np.nan, dtype=np.float32
    )
    phase_flat = phase.reshape(plane_count, pixel_count)
    depth_flat = depth.reshape(plane_count, pixel_count)
    xn_flat = np.asarray(xn, dtype=np.float64).reshape(pixel_count)
    yn_flat = np.asarray(yn, dtype=np.float64).reshape(pixel_count)
    identity = np.eye(coefficient_count, dtype=np.float64)

    for start in range(0, pixel_count, chunk_size):
        end = min(start + chunk_size, pixel_count)
        phi = phase_flat[:, start:end].T.astype(np.float64)
        zc = depth_flat[:, start:end].T.astype(np.float64)
        valid = np.isfinite(phi) & np.isfinite(zc) & (zc > 0)
        q = np.where(valid, (phi - phase_center) / phase_scale, 0.0)
        powers = np.stack(
            [q**power for power in range(coefficient_count)], axis=-1
        )
        weighted_powers = np.where(valid[..., None], powers, 0.0)

        xc = zc * xn_flat[start:end, None]
        yc = zc * yn_flat[start:end, None]
        targets = np.stack((xc, yc, zc), axis=-1)
        weighted_targets = np.where(valid[..., None], targets, 0.0)

        gram = np.einsum("cnp,cnq->cpq", weighted_powers, weighted_powers)
        rhs = np.einsum("cnp,cnk->cpk", weighted_powers, weighted_targets)
        gram += float(ridge_value) * identity[None, ...]
        solved = np.linalg.solve(gram, rhs)  # chunk, coefficient, XYZ
        solved[valid.sum(axis=1) < coefficient_count] = np.nan
        coefficients[:, start:end, :] = np.moveaxis(solved, -1, 0).astype(
            np.float32
        )
        print(f"fit XYZ polynomial pixels: {end}/{pixel_count}")

    return (
        coefficients.reshape(3, height, width, coefficient_count),
        phase_center,
        phase_scale,
    )


def _mask_for_index(index_1based: int, shape: tuple[int, int]) -> np.ndarray:
    if not use_mask:
        return np.ones(shape, dtype=bool)
    path = _resolve(mask_dir) / mask_pattern.format(index=index_1based)
    if not path.is_file():
        raise FileNotFoundError(f"测试 mask 不存在：{path}")
    mask = np.asarray(load_array(path)).squeeze().astype(bool)
    if mask.shape != shape:
        raise ValueError(f"{path} 的 mask 尺寸为 {mask.shape}，期望 {shape}")
    return mask


def main() -> None:
    phase_root = _resolve(phase_dir)
    manifest_path = _resolve(manifest)
    output_root = _resolve(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = CalibrationDataset(
        manifest_path,
        cache_size=0,
        phase_dir=phase_root,
        phase_pattern=phase_pattern,
    )
    selected_test_indices = [value - 1 for value in test_indices]
    for index in selected_test_indices:
        if index < 0 or index >= len(dataset.records):
            raise ValueError(f"测试编号 {index + 1} 超出 manifest 范围")

    phase_stack, depth_stack, train_record_indices = _load_training_stacks(dataset)
    coefficients, phase_center, phase_scale = fit_xyz_polynomial(
        phase_stack,
        depth_stack,
        dataset.xn,
        dataset.yn,
        degree,
        ridge,
        chunk_pixels,
    )

    model_path = output_root / f"xyz_polynomial_degree_{degree}.npz"
    model_metadata = {
        "method": "per_pixel_direct_XC_YC_ZC_polynomial",
        "formula": "[XC,YC,ZC] = sum_k coefficients_xyz[...,k] * q^k",
        "parameter_count_per_pixel": 3 * (degree + 1),
        "degree": degree,
        "phase_dir": str(phase_root),
        "phase_pattern": phase_pattern,
        "manifest": str(manifest_path),
        "train_indices": [index + 1 for index in train_record_indices],
        "test_indices": list(test_indices),
        "phase_normalization": "q=(phase-phase_center)/phase_scale",
        "coordinate_unit": "mm",
    }
    np.savez_compressed(
        model_path,
        coefficients_xyz=coefficients,
        phase_center=np.float64(phase_center),
        phase_scale=np.float64(phase_scale),
        metadata_json=np.array(json.dumps(model_metadata, ensure_ascii=False)),
    )
    print(
        f"{3 * (degree + 1)}参数 XYZ 多项式模型已保存："
        f"{model_path}"
    )

    per_sample: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []
    for record_index in selected_test_indices:
        index_1based = record_index + 1
        arrays = dataset.load(record_index)
        predicted_xyz = _predict_xyz(
            coefficients, arrays.phase, phase_center, phase_scale
        )
        true_xyz = np.stack(
            (
                dataset.xn * arrays.depth_mm,
                dataset.yn * arrays.depth_mm,
                arrays.depth_mm,
            ),
            axis=-1,
        ).astype(np.float32)
        evaluation_mask = _mask_for_index(
            index_1based, (dataset.height, dataset.width)
        )
        valid = (
            evaluation_mask
            & np.all(np.isfinite(predicted_xyz), axis=-1)
            & np.all(np.isfinite(true_xyz), axis=-1)
            & (true_xyz[..., 2] > 0)
        )
        error_xyz = predicted_xyz[valid] - true_xyz[valid]
        metrics = _xyz_metrics(error_xyz)
        metrics.update(
            {
                "index": index_1based,
                "sample_id": dataset.records[record_index].sample_id,
                "phase": str(dataset.records[record_index].phase_path),
            }
        )
        per_sample.append(metrics)
        all_errors.append(error_xyz)

        sample_output = output_root / f"PSP_{index_1based}"
        sample_output.mkdir(parents=True, exist_ok=True)
        np.save(sample_output / "XC.npy", predicted_xyz[..., 0])
        np.save(sample_output / "YC.npy", predicted_xyz[..., 1])
        np.save(sample_output / "ZC.npy", predicted_xyz[..., 2])
        np.save(sample_output / "point_map.npy", predicted_xyz)
        np.save(sample_output / "evaluation_mask.npy", evaluation_mask)
        _save_ascii_ply(
            sample_output / "point_cloud.ply",
            predicted_xyz,
            np.all(np.isfinite(predicted_xyz), axis=-1),
        )
        print(
            f"PSP_{index_1based} | "
            # f"XC_MAE={metrics['XC']['mae_mm']:.6g} mm | "
            # f"YC_MAE={metrics['YC']['mae_mm']:.6g} mm | "
            f"ZC_MAE={metrics['ZC']['mae_mm']:.6g} mm | "
            # f"3D_MAE={metrics['euclidean_3d']['mae_mm']:.6g} mm"
        )

    summary = _xyz_metrics(np.concatenate(all_errors, axis=0))
    result = {
        "method": "per_pixel_direct_XC_YC_ZC_polynomial",
        "degree": degree,
        "parameter_count_per_pixel": 3 * (degree + 1),
        "model": str(model_path),
        "phase_dir": str(phase_root),
        "manifest": str(manifest_path),
        "train_indices": [index + 1 for index in train_record_indices],
        "test_indices": list(test_indices),
        "use_mask_for_metrics": use_mask,
        "mask_dir": str(_resolve(mask_dir)) if use_mask else None,
        "metric_region": "mask_true_reliable_pixels" if use_mask else "all_pixels",
        "summary": summary,
        "per_sample": per_sample,
    }
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        # f"总体 | XC_MAE={summary['XC']['mae_mm']:.6g} mm | "
        # f"YC_MAE={summary['YC']['mae_mm']:.6g} mm | "
        f"ZC_MAE={summary['ZC']['mae_mm']:.6g} mm | "
        # f"3D_MAE={summary['euclidean_3d']['mae_mm']:.6g} mm | "
        # f"3D_RMSE={summary['euclidean_3d']['rmse_mm']:.6g} mm"
    )
    print(f"详细指标：{metrics_path}")


if __name__ == "__main__":
    main()
