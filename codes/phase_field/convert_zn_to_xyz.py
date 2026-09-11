from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import load_array
from .geometry import normalized_depth_to_camera_points
from .normalization import NormalizationStats


def convert(
    zn_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    mask_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用 checkpoint 中的深度范围、K 和畸变参数将 zn 转换为相机坐标。"""
    state = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
    stats = NormalizationStats.from_dict(state["normalization"])
    camera = state["camera"]
    zn = np.asarray(load_array(zn_path, "zn"), dtype=np.float32).squeeze()
    expected = tuple(int(value) for value in state["image_size"])
    if zn.shape != expected:
        raise ValueError(f"zn 尺寸 {zn.shape} 与 checkpoint 尺寸 {expected} 不一致")
    if mask_path is None:
        valid = np.isfinite(zn)
    else:
        valid = np.asarray(load_array(mask_path)).squeeze().astype(bool) & np.isfinite(
            zn
        )
        if valid.shape != zn.shape:
            raise ValueError("mask 尺寸必须与 zn 一致")
    zc, point_map = normalized_depth_to_camera_points(
        zn,
        stats.depth_min_mm,
        stats.depth_max_mm,
        np.asarray(camera["intrinsics"], dtype=np.float64),
        camera.get("radial_distortion", []),
        camera.get("tangential_distortion", []),
        camera.get("distortion_model", "none"),
    )
    zc[~valid] = np.nan
    point_map[~valid] = np.nan
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "ZC.npy", zc)
    np.save(output / "XC.npy", point_map[..., 0])
    np.save(output / "YC.npy", point_map[..., 1])
    np.save(output / "point_map.npy", point_map)
    return zc, point_map


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Direct MLP 预测的 zn 转换为 XC/YC/ZC")
    parser.add_argument("--zn", required=True, help="predicted_zn.npy 或其他 HxW zn 数组")
    parser.add_argument("--checkpoint", required=True, help="提供 Z 归一化范围和相机参数")
    parser.add_argument("--mask")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    zc, points = convert(args.zn, args.checkpoint, args.output, args.mask)
    print(
        f"转换完成：ZC {zc.shape}, point_map {points.shape}，"
        f"输出位于 {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
