from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_array
from .geometry import depth_to_point_map
from .infer import _save_ascii_ply, _save_depth_preview
from .polynomial_model import load_polynomial_model


def infer_polynomial(
    model_path: str | Path,
    phase_path: str | Path,
    output_dir: str | Path,
    mask_path: str | Path | None = None,
) -> dict[str, object]:
    lut = load_polynomial_model(model_path)
    phase = np.asarray(load_array(phase_path, "phase"), dtype=np.float32).squeeze()
    zc = lut.predict_z(phase)
    camera = lut.metadata["camera"]
    point_map = depth_to_point_map(
        zc,
        np.asarray(camera["intrinsics"], dtype=np.float64),
        camera.get("radial_distortion", []),
        camera.get("tangential_distortion", []),
        camera.get("distortion_model", "none"),
    )
    valid = np.isfinite(phase) & np.isfinite(zc) & (zc > 0)
    if mask_path is not None:
        mask = np.asarray(load_array(mask_path)).squeeze().astype(bool)
        if mask.shape != phase.shape:
            raise ValueError(f"mask 尺寸 {mask.shape} 与相位尺寸 {phase.shape} 不一致")
        valid &= mask
    zc = zc.copy()
    zc[~valid] = np.nan
    point_map[~valid] = np.nan
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "ZC.npy", zc)
    np.save(output / "XC.npy", point_map[..., 0])
    np.save(output / "YC.npy", point_map[..., 1])
    np.save(output / "point_map.npy", point_map)
    np.save(output / "valid_mask.npy", valid)
    _save_depth_preview(output / "predicted_depth.png", zc, valid)
    point_count = _save_ascii_ply(output / "point_cloud.ply", point_map)
    metadata: dict[str, object] = {
        "method": str(lut.metadata.get("method", "per_pixel_polynomial_lut")),
        "polynomial_form": str(lut.metadata.get("polynomial_form", "ZC=P(phase)")),
        "degree": lut.degree,
        "uses_mask_for_calibration": bool(
            lut.metadata.get("uses_mask_for_fit", False)
            or lut.metadata.get("mask_role")
        ),
        "uses_mask_for_inference": mask_path is not None,
        "uses_spatial_phase_fitting": bool(
            lut.metadata.get("uses_spatial_phase_fitting", False)
        ),
        "model": str(Path(model_path).resolve()),
        "phase": str(Path(phase_path).resolve()),
        "inference_mask": (
            None if mask_path is None else str(Path(mask_path).resolve())
        ),
        "valid_pixels": int(valid.sum()),
        "point_cloud_vertices": point_count,
    }
    with (output / "inference_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="用逐像素多项式 LUT 生成 XC/YC/ZC")
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = infer_polynomial(args.model, args.phase, args.output, args.mask)
    print(
        f"推理完成：degree={result['degree']}, "
        f"valid_pixels={result['valid_pixels']}, output={Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
