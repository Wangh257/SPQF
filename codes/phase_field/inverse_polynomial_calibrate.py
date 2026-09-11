from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import CalibrationDataset
from .inverse_polynomial_lut import fit_inverse_polynomial_lut
from .polynomial_calibrate import (
    _indices_for_split,
    evaluate_polynomial_lut,
    load_unmasked_stacks,
)
from .utils import camera_to_dict, save_json


def calibrate_inverse_polynomial(
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
    """标定 ``ZC=1/P_degree(phase)`` 的逐像素逆多项式 LUT。"""
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
    lut = fit_inverse_polynomial_lut(
        phase_stack=phase_stack,
        depth_stack_mm=depth_stack,
        degree=degree,
        ridge=ridge,
        chunk_pixels=chunk_pixels,
        metadata=metadata,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = lut.save(output / f"inverse_polynomial_degree_{degree}.npz")

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
                "polynomial_form": "ZC=1/(a0+a1*q+...+a_degree*q^degree)",
                "fit_split": fit_split,
                "eval_split": eval_split,
                "model": str(model_path),
            }
        )
        save_json(output / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐像素逆多项式标定：ZC=1/(a0+a1*q+...+a_degree*q^degree)"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase-dir", help="相位目录；不给则使用 manifest 路径")
    parser.add_argument("--phase-pattern", default="PSP_{index}.mat")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--fit-split", default="train", choices=["train", "val", "test", "all"]
    )
    parser.add_argument(
        "--eval-split", default="test", choices=["train", "val", "test", "all", "none"]
    )
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--chunk-pixels", type=int, default=100_000)
    parser.add_argument("--eval-mask-dir", help="仅用于评价指标的 mask 目录")
    parser.add_argument("--eval-mask-pattern", default="mask_{index}.bmp")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = calibrate_inverse_polynomial(
        manifest_path=args.manifest,
        output_dir=args.output,
        degree=args.degree,
        fit_split=args.fit_split,
        eval_split=args.eval_split,
        phase_dir=args.phase_dir,
        phase_pattern=args.phase_pattern,
        ridge=args.ridge,
        chunk_pixels=args.chunk_pixels,
        eval_mask_dir=args.eval_mask_dir,
        eval_mask_pattern=args.eval_mask_pattern,
    )
    print(f"逆多项式 LUT 已保存：{model}")


if __name__ == "__main__":
    main()
