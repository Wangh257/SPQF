from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import savemat

from .cpc_phase import cpc_fit_phase
from .data import CalibrationDataset, load_array
from .polynomial_calibrate import (
    _indices_for_split,
    _phase_path,
    _unmasked_depth,
)
from .polynomial_lut import PolynomialLUT, fit_polynomial_lut
from .utils import camera_to_dict, regression_metrics, save_json


def load_cpc_phase_stacks(
    dataset: CalibrationDataset,
    indices: Iterable[int],
    raw_phase_dir: str | Path | None,
    phase_pattern: str,
    cpc_order: int,
    cpc_output_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    mask_pattern: str = "mask_{index}.bmp",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Use each mask to CPC-fit raw phase, then return the filled phase stack."""
    phase_root = None if raw_phase_dir is None else Path(raw_phase_dir).resolve()
    mask_root = None if mask_dir is None else Path(mask_dir).resolve()
    cpc_root = None if cpc_output_dir is None else Path(cpc_output_dir).resolve()
    if cpc_root is not None:
        cpc_root.mkdir(parents=True, exist_ok=True)
    phases: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    sample_ids: list[str] = []
    expected_shape = (dataset.height, dataset.width)

    for index in indices:
        record = dataset.records[index]
        current_mask_path = (
            record.mask_path
            if mask_root is None
            else mask_root
            / mask_pattern.format(
                index=index + 1,
                id=record.sample_id,
                plane_id=record.plane_id,
            )
        )
        if current_mask_path is None:
            raise ValueError(f"{record.sample_id} 没有 mask，无法执行 CPC")
        phase_path = _phase_path(record, index, phase_root, phase_pattern)
        raw_phase = np.asarray(
            load_array(phase_path, "phase"), dtype=np.float32
        ).squeeze()
        mask = np.asarray(load_array(current_mask_path)).squeeze().astype(bool)
        if raw_phase.shape != expected_shape:
            raise ValueError(f"{phase_path} 的相位尺寸错误：{raw_phase.shape}")
        if mask.shape != expected_shape:
            raise ValueError(f"{current_mask_path} 的 mask 尺寸错误：{mask.shape}")

        fitted_phase = cpc_fit_phase(raw_phase, mask, order=cpc_order)
        phases.append(fitted_phase)
        depths.append(_unmasked_depth(dataset, record))
        sample_ids.append(record.sample_id)
        if cpc_root is not None:
            savemat(
                cpc_root / f"PSP_{index + 1}.mat",
                {"phase": fitted_phase},
                do_compression=True,
            )
        print(
            f"load {record.sample_id}: raw phase -> CPC(order={cpc_order}), "
            f"mask valid={int(mask.sum())}/{mask.size}"
        )
    return np.stack(phases), np.stack(depths), sample_ids


def evaluate_cpc_polynomial_lut(
    lut: PolynomialLUT,
    dataset: CalibrationDataset,
    indices: Iterable[int],
    raw_phase_dir: str | Path | None,
    phase_pattern: str,
    cpc_order: int,
    cpc_output_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    mask_pattern: str = "mask_{index}.bmp",
) -> dict[str, Any]:
    selected_indices = list(indices)
    cpc_phases, targets, sample_ids = load_cpc_phase_stacks(
        dataset,
        selected_indices,
        raw_phase_dir,
        phase_pattern,
        cpc_order,
        cpc_output_dir,
        mask_dir,
        mask_pattern,
    )
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    per_plane: list[dict[str, Any]] = []
    mask_root = None if mask_dir is None else Path(mask_dir).resolve()
    for record_index, sample_id, phase, target in zip(
        selected_indices, sample_ids, cpc_phases, targets
    ):
        record = dataset.records[record_index]
        evaluation_mask_path = (
            record.mask_path
            if mask_root is None
            else mask_root
            / mask_pattern.format(
                index=record_index + 1,
                id=record.sample_id,
                sample_id=record.sample_id,
                plane_id=record.plane_id,
            )
        )
        if evaluation_mask_path is None or not evaluation_mask_path.is_file():
            raise FileNotFoundError(
                f"{sample_id} 的评价 mask 不存在：{evaluation_mask_path}"
            )
        evaluation_mask = (
            np.asarray(load_array(evaluation_mask_path)).squeeze().astype(bool)
        )
        if evaluation_mask.shape != phase.shape:
            raise ValueError(
                f"{evaluation_mask_path} 的 mask 尺寸错误："
                f"{evaluation_mask.shape}"
            )
        prediction = lut.predict_z(phase)
        valid = (
            evaluation_mask
            & np.isfinite(phase)
            & np.isfinite(target)
            & np.isfinite(prediction)
        )
        plane_prediction = prediction[valid]
        plane_target = target[valid]
        predictions.append(plane_prediction)
        truths.append(plane_target)
        per_plane.append(
            {
                "sample_id": sample_id,
                "all_finite_pixels": regression_metrics(plane_prediction, plane_target),
            }
        )
    summary = regression_metrics(np.concatenate(predictions), np.concatenate(truths))
    return {
        "summary": summary,
        "summary_mask_valid_pixels": summary,
        "per_plane": per_plane,
        "metric_region": "mask_true_reliable_pixels",
        "mask_usage": "mask is used by CPC to fit/fill phase and to exclude black-dot pixels from evaluation metrics",
    }


def calibrate_masked_cpc(
    manifest_path: str | Path,
    output_dir: str | Path,
    degree: int = 3,
    cpc_order: int = 2,
    fit_split: str = "train",
    eval_split: str = "test",
    raw_phase_dir: str | Path | None = None,
    phase_pattern: str = "PSP_{index}.mat",
    ridge: float = 1e-8,
    chunk_pixels: int = 100_000,
    mask_dir: str | Path | None = None,
    mask_pattern: str = "mask_{index}.bmp",
) -> Path:
    dataset = CalibrationDataset(manifest_path, cache_size=0)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase_stack, depth_stack, fit_ids = load_cpc_phase_stacks(
        dataset,
        _indices_for_split(dataset, fit_split),
        raw_phase_dir,
        phase_pattern,
        cpc_order,
        output / "cpc_phases_fit",
        mask_dir,
        mask_pattern,
    )
    metadata = {
        "manifest": str(Path(manifest_path).resolve()),
        "fit_split": fit_split,
        "fit_ids": fit_ids,
        "raw_phase_dir": (
            None if raw_phase_dir is None else str(Path(raw_phase_dir).resolve())
        ),
        "phase_pattern": phase_pattern,
        "mask_dir": None if mask_dir is None else str(Path(mask_dir).resolve()),
        "mask_pattern": mask_pattern,
        "cpc_order": cpc_order,
        "camera": camera_to_dict(dataset.camera),
        "mask_role": "fit the CPC phase surface from white/reliable pixels",
    }
    lut = fit_polynomial_lut(
        phase_stack,
        depth_stack,
        degree=degree,
        ridge=ridge,
        chunk_pixels=chunk_pixels,
        metadata=metadata,
        uses_spatial_phase_fitting=True,
    )
    model_path = lut.save(
        output / f"polynomial_masked_cpc_degree_{degree}.npz",
        dataset.xn,
        dataset.yn,
    )
    if eval_split.lower() != "none":
        metrics = evaluate_cpc_polynomial_lut(
            lut,
            dataset,
            _indices_for_split(dataset, eval_split),
            raw_phase_dir,
            phase_pattern,
            cpc_order,
            output / f"cpc_phases_{eval_split}",
            mask_dir,
            mask_pattern,
        )
        metrics.update(
            {
                "degree": degree,
                "cpc_order": cpc_order,
                "fit_split": fit_split,
                "eval_split": eval_split,
                "model": str(model_path),
            }
        )
        save_json(output / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="mask+CPC 拟合相位后的可选次数逐像素多项式标定")
    parser.add_argument("--manifest", required=True, help="manifest 必须包含 mask")
    parser.add_argument("--raw-phase-dir", help="原始未 CPC 拟合的展开相位目录")
    parser.add_argument("--phase-pattern", default="PSP_{index}.mat")
    parser.add_argument("--mask-dir", help="mask 目录；不给时使用 manifest 中的路径")
    parser.add_argument("--mask-pattern", default="mask_{index}.bmp")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--cpc-order", type=int, default=2)
    parser.add_argument(
        "--fit-split", default="train", choices=["train", "val", "test", "all"]
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "val", "test", "all", "none"],
    )
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--chunk-pixels", type=int, default=100_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = calibrate_masked_cpc(
        manifest_path=args.manifest,
        output_dir=args.output,
        degree=args.degree,
        cpc_order=args.cpc_order,
        fit_split=args.fit_split,
        eval_split=args.eval_split,
        raw_phase_dir=args.raw_phase_dir,
        phase_pattern=args.phase_pattern,
        ridge=args.ridge,
        chunk_pixels=args.chunk_pixels,
        mask_dir=args.mask_dir,
        mask_pattern=args.mask_pattern,
    )
    print(f"mask+CPC 多项式 LUT 已保存：{model}")


if __name__ == "__main__":
    main()
