from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import CalibrationDataset
from .geometry import depth_to_point_map, fit_plane
from .predictor import DirectMLPPredictor
from .utils import regression_metrics, save_json


def evaluate(
    checkpoint: str | Path,
    manifest: str | Path,
    split: str,
    output_dir: str | Path,
    max_pixels_per_plane: int = 0,
    chunk_size: int = 262_144,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    predictor = DirectMLPPredictor(checkpoint, device)
    dataset = CalibrationDataset(manifest)
    if tuple(predictor.image_size) != (dataset.height, dataset.width):
        raise ValueError("checkpoint 和评估数据的图像尺寸不一致")
    indices = dataset.indices(split)
    if not indices:
        raise ValueError(f"manifest 中没有 split={split!r}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    per_plane: list[dict[str, Any]] = []
    for index in indices:
        record = dataset.records[index]
        arrays = dataset.load(index)
        zn, prediction, predicted_valid = predictor.predict(
            arrays.phase,
            arrays.valid,
            arrays.amplitude,
            chunk_size,
            warn_out_of_range=False,
        )
        flat = np.flatnonzero(arrays.valid & predicted_valid)
        if max_pixels_per_plane > 0 and flat.size > max_pixels_per_plane:
            flat = rng.choice(flat, size=max_pixels_per_plane, replace=False)
        pred_values = prediction.ravel()[flat]
        target_values = arrays.depth_mm.ravel()[flat]
        metrics = regression_metrics(pred_values, target_values)
        metrics.update({"sample_id": record.sample_id, "plane_id": record.plane_id})

        point_map = depth_to_point_map(
            prediction,
            dataset.camera.intrinsics,
            dataset.camera.radial_distortion,
            dataset.camera.tangential_distortion,
            dataset.camera.distortion_model,
        )
        points = point_map.reshape(-1, 3)[flat]
        points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0] < 3:
            metrics["predicted_plane_fit_rmse_mm"] = float("nan")
            per_plane.append(metrics)
            all_predictions.append(pred_values)
            all_targets.append(target_values)
            continue
        if points.shape[0] > 100_000:
            points = points[rng.choice(points.shape[0], 100_000, replace=False)]
        plane = fit_plane(points)
        distances = np.abs(points @ plane[:3] + plane[3])
        metrics["predicted_plane_fit_rmse_mm"] = float(
            np.sqrt(np.square(distances).mean())
        )
        if record.plane is not None:
            gt_plane = np.asarray(record.plane, dtype=np.float64)
            gt_plane /= np.linalg.norm(gt_plane[:3])
            signed = points @ gt_plane[:3] + gt_plane[3]
            metrics["distance_to_gt_plane_bias_mm"] = float(signed.mean())
            metrics["distance_to_gt_plane_rmse_mm"] = float(
                np.sqrt(np.square(signed).mean())
            )
        per_plane.append(metrics)
        all_predictions.append(pred_values)
        all_targets.append(target_values)

    summary = regression_metrics(
        np.concatenate(all_predictions), np.concatenate(all_targets)
    )
    result = {"split": split, "summary": summary, "per_plane": per_plane}
    save_json(output / "metrics.json", result)
    fieldnames = list(dict.fromkeys(key for row in per_plane for key in row.keys()))
    with (output / "per_plane_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_plane)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="按完整标定平面评估 Direct MLP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pixels-per-plane", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint,
        args.manifest,
        args.split,
        args.output,
        args.max_pixels_per_plane,
        args.chunk_size,
        args.device,
    )
    summary = result["summary"]
    print(
        f"{args.split}: MAE={summary['mae_mm']:.4f} mm, "
        f"RMSE={summary['rmse_mm']:.4f} mm, P95={summary['p95_ae_mm']:.4f} mm"
    )


if __name__ == "__main__":
    main()
