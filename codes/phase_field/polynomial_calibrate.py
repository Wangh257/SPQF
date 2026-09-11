from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import CalibrationDataset, SampleRecord, load_array
from .geometry import intersect_camera_rays_with_plane
from .polynomial_lut import PolynomialLUT, fit_polynomial_lut
from .utils import camera_to_dict, regression_metrics, save_json


def _indices_for_split(dataset: CalibrationDataset, split: str) -> list[int]:
    if split == "all":
        return list(range(len(dataset.records)))
    indices = dataset.indices(split)
    if not indices:
        raise ValueError(f"manifest 中没有 split={split!r}")
    return indices


def _phase_path(
    record: SampleRecord,
    record_index: int,
    phase_dir: Path | None,
    phase_pattern: str,
) -> Path:
    if phase_dir is None:
        return record.phase_path
    return phase_dir / phase_pattern.format(
        index=record_index + 1,
        id=record.sample_id,
        plane_id=record.plane_id,
    )


def _unmasked_depth(dataset: CalibrationDataset, record: SampleRecord) -> np.ndarray:
    if record.depth_path is not None:
        depth = np.asarray(load_array(record.depth_path, "depth"), dtype=np.float32)
    elif record.z_mm is not None:
        depth = np.full((dataset.height, dataset.width), record.z_mm, dtype=np.float32)
    else:
        depth = intersect_camera_rays_with_plane(
            dataset.xn, dataset.yn, record.plane or ()
        )
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.shape != (dataset.height, dataset.width):
        raise ValueError(f"{record.sample_id} 的深度尺寸错误：{depth.shape}")
    return depth


def load_unmasked_stacks(
    dataset: CalibrationDataset,
    indices: Iterable[int],
    phase_dir: str | Path | None,
    phase_pattern: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = None if phase_dir is None else Path(phase_dir).resolve()
    phases: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    ids: list[str] = []
    for index in indices:
        record = dataset.records[index]
        path = _phase_path(record, index, root, phase_pattern)
        phase = np.asarray(load_array(path, "phase"), dtype=np.float32).squeeze()
        if phase.shape != (dataset.height, dataset.width):
            raise ValueError(f"{path} 的相位尺寸错误：{phase.shape}")
        phases.append(phase)
        depths.append(_unmasked_depth(dataset, record))
        ids.append(record.sample_id)
        print(
            f"load {record.sample_id}: phase=[{np.nanmin(phase):.4f},"
            f" {np.nanmax(phase):.4f}] (mask not used)"
        )
    return np.stack(phases), np.stack(depths), ids


def _evaluation_mask(
    record: SampleRecord,
    record_index: int,
    shape: tuple[int, int],
    mask_dir: Path | None,
    mask_pattern: str,
) -> np.ndarray | None:
    mask_path = record.mask_path
    if mask_dir is not None:
        mask_path = mask_dir / mask_pattern.format(
            index=record_index + 1,
            id=record.sample_id,
            sample_id=record.sample_id,
            plane_id=record.plane_id,
        )
    if mask_path is None:
        return None
    if not mask_path.is_file():
        raise FileNotFoundError(f"评价 mask 不存在：{mask_path}")
    mask = np.asarray(load_array(mask_path)).squeeze().astype(bool)
    if mask.shape != shape:
        raise ValueError(f"{mask_path} 的 mask 尺寸错误：{mask.shape}")
    return mask


def evaluate_polynomial_lut(
    lut: PolynomialLUT,
    dataset: CalibrationDataset,
    indices: Iterable[int],
    phase_dir: str | Path | None,
    phase_pattern: str,
    eval_mask_dir: str | Path | None = None,
    eval_mask_pattern: str = "mask_{index}.bmp",
) -> dict[str, Any]:
    root = None if phase_dir is None else Path(phase_dir).resolve()
    mask_root = None if eval_mask_dir is None else Path(eval_mask_dir).resolve()
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    reliable_predictions: list[np.ndarray] = []
    reliable_targets: list[np.ndarray] = []
    black_predictions: list[np.ndarray] = []
    black_targets: list[np.ndarray] = []
    per_plane: list[dict[str, Any]] = []
    for index in indices:
        record = dataset.records[index]
        phase = np.asarray(
            load_array(_phase_path(record, index, root, phase_pattern), "phase"),
            dtype=np.float32,
        ).squeeze()
        target = _unmasked_depth(dataset, record)
        prediction = lut.predict_z(phase)
        finite = np.isfinite(phase) & np.isfinite(target) & np.isfinite(prediction)
        row: dict[str, Any] = {
            "sample_id": record.sample_id,
            "all_pixels": regression_metrics(prediction[finite], target[finite]),
        }
        all_predictions.append(prediction[finite])
        all_targets.append(target[finite])
        mask = _evaluation_mask(
            record, index, phase.shape, mask_root, eval_mask_pattern
        )
        if mask is not None:
            reliable = finite & mask
            black = finite & ~mask
            if np.any(reliable):
                row["mask_valid_pixels"] = regression_metrics(
                    prediction[reliable], target[reliable]
                )
                reliable_predictions.append(prediction[reliable])
                reliable_targets.append(target[reliable])
            if np.any(black):
                row["mask_invalid_black_pixels"] = regression_metrics(
                    prediction[black], target[black]
                )
                black_predictions.append(prediction[black])
                black_targets.append(target[black])
        per_plane.append(row)

    all_summary = regression_metrics(
        np.concatenate(all_predictions), np.concatenate(all_targets)
    )
    result: dict[str, Any] = {
        "summary": all_summary,
        "summary_all_pixels": all_summary,
        "per_plane": per_plane,
        "mask_usage": "mask was not used for phase extraction, calibration, fitting, or prediction; only for post-hoc stratified metrics",
    }
    if reliable_predictions:
        reliable_summary = regression_metrics(
            np.concatenate(reliable_predictions), np.concatenate(reliable_targets)
        )
        result["summary"] = reliable_summary
        result["summary_mask_valid_pixels"] = reliable_summary
    if black_predictions:
        result["summary_mask_invalid_black_pixels"] = regression_metrics(
            np.concatenate(black_predictions), np.concatenate(black_targets)
        )
    return result


def calibrate(
    manifest_path: str | Path,
    output_dir: str | Path,
    degree: int = 3,
    fit_split: str = "train",
    eval_split: str = "test",
    phase_dir: str | Path | None = None,
    phase_pattern: str = "PSP_{index}.mat",
    ridge: float = 1e-8,
    chunk_pixels: int = 100_000,
    eval_mask_dir: str | Path | None = None,
    eval_mask_pattern: str = "mask_{index}.bmp",
) -> Path:
    dataset = CalibrationDataset(manifest_path, cache_size=0)
    fit_indices = _indices_for_split(dataset, fit_split)
    phase_stack, depth_stack, fit_ids = load_unmasked_stacks(
        dataset, fit_indices, phase_dir, phase_pattern
    )
    metadata = {
        "manifest": str(Path(manifest_path).resolve()),
        "fit_split": fit_split,
        "fit_ids": fit_ids,
        "phase_dir": None if phase_dir is None else str(Path(phase_dir).resolve()),
        "phase_pattern": phase_pattern,
        "camera": camera_to_dict(dataset.camera),
    }
    lut = fit_polynomial_lut(
        phase_stack,
        depth_stack,
        degree,
        ridge,
        chunk_pixels,
        metadata,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = lut.save(
        output / f"polynomial_degree_{degree}.npz", dataset.xn, dataset.yn
    )
    if eval_split.lower() != "none":
        eval_indices = _indices_for_split(dataset, eval_split)
        metrics = evaluate_polynomial_lut(
            lut,
            dataset,
            eval_indices,
            phase_dir,
            phase_pattern,
            eval_mask_dir,
            eval_mask_pattern,
        )
        metrics.update(
            {
                "degree": degree,
                "fit_split": fit_split,
                "eval_split": eval_split,
                "model": str(model_path),
            }
        )
        save_json(output / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="无 mask、无相位空间拟合的可选次数逐像素多项式标定")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase-dir", help="原始未拟合相位目录；不给则使用 manifest 路径")
    parser.add_argument("--phase-pattern", default="PSP_{index}.mat")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--fit-split", default="train", choices=["train", "val", "test", "all"]
    )
    parser.add_argument(
        "--eval-split", default="test", choices=["train", "val", "test", "all", "none"]
    )
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--chunk-pixels", type=int, default=100000)
    parser.add_argument("--eval-mask-dir", help="仅用于评价指标的 mask 目录")
    parser.add_argument("--eval-mask-pattern", default="mask_{index}.bmp")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = calibrate(
        args.manifest,
        args.output,
        args.degree,
        args.fit_split,
        args.eval_split,
        args.phase_dir,
        args.phase_pattern,
        args.ridge,
        args.chunk_pixels,
        args.eval_mask_dir,
        args.eval_mask_pattern,
    )
    print(f"多项式 LUT 已保存：{model}")


if __name__ == "__main__":
    main()
