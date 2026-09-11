from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .data import load_array
from .geometry import depth_to_point_map
from .predictor import DirectMLPPredictor


def _save_depth_preview(path: Path, depth: np.ndarray, valid: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        masked = np.ma.masked_where(~valid, depth)
        figure, axis = plt.subplots(figsize=(10, 7))
        image = axis.imshow(masked, cmap="turbo")
        axis.set_title("Predicted camera Z depth (mm)")
        axis.axis("off")
        figure.colorbar(image, ax=axis, label="ZC / mm")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
    except ImportError:
        warnings.warn("未安装 matplotlib，跳过 predicted_depth.png", stacklevel=2)


def _save_ascii_ply(path: Path, point_map: np.ndarray) -> int:
    points = np.asarray(point_map, dtype=np.float32).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write(
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        np.savetxt(handle, points, fmt="%.7g %.7g %.7g")
    return int(points.shape[0])


def run_inference(
    checkpoint: str | Path,
    phase_path: str | Path,
    output_dir: str | Path,
    mask_path: Optional[str | Path] = None,
    amplitude_path: Optional[str | Path] = None,
    device: str = "auto",
    chunk_size: int = 262_144,
    save_ply: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictor = DirectMLPPredictor(checkpoint, device)
    phase = load_array(phase_path, "phase")
    mask = None if mask_path is None else load_array(mask_path)
    amplitude = None if amplitude_path is None else load_array(amplitude_path)
    zn, zc, valid = predictor.predict(phase, mask, amplitude, chunk_size)
    camera = predictor.camera
    point_map = depth_to_point_map(
        zc,
        np.asarray(camera["intrinsics"], dtype=np.float64),
        camera.get("radial_distortion", []),
        camera.get("tangential_distortion", []),
        camera.get("distortion_model", "none"),
    )
    point_map[~valid] = np.nan
    xc, yc, zc_from_points = (
        point_map[..., 0],
        point_map[..., 1],
        point_map[..., 2],
    )
    np.save(output / "predicted_zn.npy", zn)
    np.save(output / "predicted_depth.npy", zc)
    np.save(output / "valid_mask.npy", valid)
    np.save(output / "point_map.npy", point_map)
    np.save(output / "XC.npy", xc)
    np.save(output / "YC.npy", yc)
    np.save(output / "ZC.npy", zc_from_points)
    _save_depth_preview(output / "predicted_depth.png", zc, valid)
    point_count = (
        _save_ascii_ply(output / "point_cloud.ply", point_map) if save_ply else 0
    )
    valid_zn = zn[valid]
    outside_depth = (valid_zn < 0.0) | (valid_zn > 1.0)
    metadata = {
        "method": "direct_mlp",
        "checkpoint": str(Path(checkpoint).resolve()),
        "phase": str(Path(phase_path).resolve()),
        "image_size": list(zn.shape),
        "depth_definition": "camera_axis_Z",
        "depth_unit": "mm",
        "zn_definition": "(ZC-depth_min_mm)/(depth_max_mm-depth_min_mm)",
        "valid_pixels": int(valid.sum()),
        "point_cloud_vertices": point_count,
        "zn_outside_0_1_ratio": float(outside_depth.mean()),
        "normalization": predictor.stats.to_dict(),
        "camera": camera,
        "formulas": {
            "ZC": "zn*(depth_max_mm-depth_min_mm)+depth_min_mm",
            "XC": "xn*ZC",
            "YC": "yn*ZC",
            "ray": "[xn,yn,1] from undistorted camera pixel",
        },
    }
    with (output / "inference_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    if metadata["zn_outside_0_1_ratio"] > 0:
        warnings.warn(
            f"{metadata['zn_outside_0_1_ratio']:.2%} 的预测 zn 位于 [0,1] 之外，"
            "结果已原样保存，未静默 clamp。",
            stacklevel=2,
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct MLP 单张相位图推理")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", required=True, help=".mat/.npy/.npz，可用 file::key")
    parser.add_argument("--mask")
    parser.add_argument("--amplitude")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--no-ply", action="store_true")
    args = parser.parse_args()
    metadata = run_inference(
        args.checkpoint,
        args.phase,
        args.output,
        args.mask,
        args.amplitude,
        args.device,
        args.chunk_size,
        not args.no_ply,
    )
    print(f"推理完成：{metadata['valid_pixels']} 个有效像素，输出位于 {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
