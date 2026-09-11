from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data import CalibrationDataset, load_array
from .infer import run_inference
from .polynomial_infer import infer_polynomial
from .utils import regression_metrics, save_json


def _external_mask_path(
    mask_dir: Path | None,
    mask_pattern: str,
    index: int,
    sample_id: str,
    plane_id: str,
) -> Path | None:
    if mask_dir is None:
        return None
    path = mask_dir / mask_pattern.format(
        index=index + 1,
        id=sample_id,
        sample_id=sample_id,
        plane_id=plane_id,
    )
    if not path.is_file():
        raise FileNotFoundError(f"第 {index + 1} 张标定板的 mask 不存在：{path}")
    return path


def batch_test(
    model_path: str | Path,
    manifest_path: str | Path,
    phase_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    phase_pattern: str = "PSP_{index}.mat",
    mask_dir: str | Path | None = None,
    mask_pattern: str = "mask_{index}.bmp",
    device: str = "auto",
    chunk_size: int = 262_144,
) -> dict[str, Any]:
    """重建并评价 manifest 某一 split 中的全部标定板。"""
    model = Path(model_path).resolve()
    if model.suffix.lower() not in {".pt", ".npz"}:
        raise ValueError("模型必须是 Direct MLP .pt 或多项式 .npz")
    dataset = CalibrationDataset(
        manifest_path,
        cache_size=1,
        phase_dir=phase_dir,
        phase_pattern=phase_pattern,
    )
    if split == "all":
        indices = list(range(len(dataset.records)))
    else:
        indices = dataset.indices(split)
    if not indices:
        raise ValueError(f"manifest 中没有 split={split!r} 的样本")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_mask_dir = None if mask_dir is None else Path(mask_dir).resolve()
    if resolved_mask_dir is not None and not resolved_mask_dir.is_dir():
        raise FileNotFoundError(f"mask 目录不存在：{resolved_mask_dir}")

    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []
    for index in indices:
        record = dataset.records[index]
        arrays = dataset.load(index)
        current_mask = _external_mask_path(
            resolved_mask_dir,
            mask_pattern,
            index,
            record.sample_id,
            record.plane_id,
        )
        evaluation_mask: np.ndarray | None = None
        if current_mask is not None:
            evaluation_mask = np.asarray(load_array(current_mask)).squeeze().astype(bool)
            expected_shape = (dataset.height, dataset.width)
            if evaluation_mask.shape != expected_shape:
                raise ValueError(
                    f"{current_mask} 的 mask 尺寸为 {evaluation_mask.shape}，"
                    f"期望 {expected_shape}"
                )
        sample_output = output / f"PSP_{index + 1}"
        if model.suffix.lower() == ".pt":
            run_inference(
                checkpoint=model,
                phase_path=record.phase_path,
                output_dir=sample_output,
                mask_path=None,
                amplitude_path=record.amplitude_path,
                device=device,
                chunk_size=chunk_size,
            )
        else:
            infer_polynomial(
                model_path=model,
                phase_path=record.phase_path,
                output_dir=sample_output,
                mask_path=None,
            )

        prediction = np.asarray(np.load(sample_output / "ZC.npy"), dtype=np.float32)
        inference_valid = np.asarray(
            np.load(sample_output / "valid_mask.npy"), dtype=bool
        )
        valid = arrays.valid & inference_valid & np.isfinite(prediction)
        if evaluation_mask is not None:
            valid &= evaluation_mask
        if not np.any(valid):
            raise ValueError(f"{record.sample_id} 没有可评价的有效像素")
        pred_values = prediction[valid]
        target_values = arrays.depth_mm[valid]
        metrics = regression_metrics(pred_values, target_values)
        metrics.update(
            {
                "index": index + 1,
                "sample_id": record.sample_id,
                "plane_id": record.plane_id,
                "phase": str(record.phase_path),
                "evaluation_mask": (
                    None if current_mask is None else str(current_mask)
                ),
                "output": str(sample_output),
            }
        )
        per_sample.append(metrics)
        all_predictions.append(pred_values)
        all_targets.append(target_values)
        print(
            f"PSP_{index + 1} ({record.sample_id}) | "
            f"MAE={metrics['mae_mm']:.6g} mm | "
            f"RMSE={metrics['rmse_mm']:.6g} mm"
        )

    result: dict[str, Any] = {
        "model": str(model),
        "model_type": "direct_mlp" if model.suffix.lower() == ".pt" else "polynomial",
        "manifest": str(Path(manifest_path).resolve()),
        "phase_dir": str(Path(phase_dir).resolve()),
        "phase_pattern": phase_pattern,
        "evaluation_mask_dir": (
            None if resolved_mask_dir is None else str(resolved_mask_dir)
        ),
        "evaluation_mask_pattern": (
            None if resolved_mask_dir is None else mask_pattern
        ),
        "metric_region": (
            "all_valid_pixels"
            if resolved_mask_dir is None
            else "mask_true_reliable_pixels"
        ),
        "split": split,
        "sample_count": len(indices),
        "sample_indices": [index + 1 for index in indices],
        "summary": regression_metrics(
            np.concatenate(all_predictions), np.concatenate(all_targets)
        ),
        "per_sample": per_sample,
    }
    save_json(output / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按训练 manifest 的 split 批量重建并评价标定板"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--phase-pattern", default="PSP_{index}.mat")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test", "all"]
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-dir")
    parser.add_argument("--mask-pattern", default="mask_{index}.bmp")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=262_144)
    args = parser.parse_args()
    result = batch_test(
        model_path=args.model,
        manifest_path=args.manifest,
        phase_dir=args.phase_dir,
        output_dir=args.output,
        split=args.split,
        phase_pattern=args.phase_pattern,
        mask_dir=args.mask_dir,
        mask_pattern=args.mask_pattern,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    summary = result["summary"]
    print(
        f"完成：split={result['split']}，样本={result['sample_indices']}，"
        f"MAE={summary['mae_mm']:.6g} mm，RMSE={summary['rmse_mm']:.6g} mm，"
        f"输出={Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
