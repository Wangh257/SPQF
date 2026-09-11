from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .data import load_array
from .geometry import fit_plane


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def _field(container: Any, *names: str) -> Any:
    candidates = {name.lower() for name in names}
    if isinstance(container, dict):
        for key, value in container.items():
            if str(key).lower() in candidates:
                return value
    if isinstance(container, np.ndarray) and container.dtype.names:
        for key in container.dtype.names:
            if key.lower() in candidates:
                return container[key]
    for name in names:
        if hasattr(container, name):
            return getattr(container, name)
    raise KeyError(f"找不到字段 {names}")


def _optional_field(container: Any, *names: str, default: Any = None) -> Any:
    try:
        return _field(container, *names)
    except KeyError:
        return default


def load_camera_parameters(path: str | Path) -> dict[str, np.ndarray]:
    """读取 MATLAB cameraParams 或导出的纯数值 npz/mat。"""
    path = Path(path).resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            root: Any = dict(source.items())
    elif path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("读取 cameraParams.mat 需要 scipy") from exc
        loaded = loadmat(path, simplify_cells=True)
        root = loaded.get("cameraParams", loaded)
    else:
        raise ValueError("camera parameters 必须是 .mat 或 .npz")
    try:
        intrinsic_raw = np.asarray(
            _field(root, "K", "IntrinsicMatrix", "intrinsics"), dtype=np.float64
        ).squeeze()
        rotations = np.asarray(
            _field(root, "R", "RotationMatrices", "rotations"), dtype=np.float64
        ).squeeze()
        translations = np.asarray(
            _field(root, "T", "TranslationVectors", "translations"), dtype=np.float64
        ).squeeze()
        world_points = np.asarray(
            _field(root, "WorldPoints", "world_points"), dtype=np.float64
        ).squeeze()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"无法从 {path} 解析 cameraParams 对象。请按 README 中的代码将 "
            "K/R/T/WorldPoints 导出成纯数值 camera_numeric.mat 或 npz。"
        ) from exc
    if intrinsic_raw.shape != (3, 3):
        raise ValueError(f"内参尺寸错误：{intrinsic_raw.shape}")
    # MATLAB cameraParameters.IntrinsicMatrix 是转置存储；已导出的 K 通常是标准形式。
    if abs(intrinsic_raw[2, 0]) + abs(intrinsic_raw[2, 1]) > abs(
        intrinsic_raw[0, 2]
    ) + abs(intrinsic_raw[1, 2]):
        intrinsics = intrinsic_raw.T
    else:
        intrinsics = intrinsic_raw
    if rotations.ndim != 3:
        raise ValueError(f"RotationMatrices 应为 3D 数组，实际 {rotations.shape}")
    if rotations.shape[:2] == (3, 3):
        rotations = np.moveaxis(rotations, -1, 0)
    elif rotations.shape[1:] != (3, 3):
        raise ValueError(f"RotationMatrices 尺寸错误：{rotations.shape}")
    translations = translations.reshape(-1, 3)
    world_points = world_points.reshape(-1, world_points.shape[-1])
    if world_points.shape[1] == 2:
        world_points = np.column_stack((world_points, np.zeros(world_points.shape[0])))
    if world_points.shape[1] != 3:
        raise ValueError("WorldPoints 应为 (M,2) 或 (M,3)")
    if rotations.shape[0] != translations.shape[0]:
        raise ValueError("RotationMatrices 与 TranslationVectors 数量不一致")
    radial = np.asarray(
        _optional_field(root, "RadialDistortion", "radial_distortion", default=[]),
        dtype=np.float64,
    ).reshape(-1)
    tangential = np.asarray(
        _optional_field(
            root, "TangentialDistortion", "tangential_distortion", default=[]
        ),
        dtype=np.float64,
    ).reshape(-1)
    return {
        "K": intrinsics,
        "R": rotations,
        "T": translations,
        "world_points": world_points,
        "radial": radial,
        "tangential": tangential,
    }


def _split_names(
    count: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    val_count: Optional[int] = None,
    test_count: Optional[int] = None,
) -> list[str]:
    if count < 2:
        raise ValueError("至少需要 2 个标定平面才能划分训练和测试集")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    val_count = (
        max(1, int(round(count * val_fraction)))
        if val_count is None
        else int(val_count)
    )
    test_count = (
        max(1, int(round(count * test_fraction)))
        if test_count is None
        else int(test_count)
    )
    if val_count < 0 or test_count < 0:
        raise ValueError("val_count/test_count 不能为负数")
    if val_count + test_count >= count:
        raise ValueError("val_count + test_count 必须小于标定平面总数")
    names = np.full(count, "train", dtype=object)
    names[order[:val_count]] = "val"
    names[order[val_count : val_count + test_count]] = "test"
    return [str(value) for value in names]


def create_manifest(
    camera_params: str | Path,
    phase_dir: str | Path,
    output_path: str | Path,
    phase_glob: str = "*.mat",
    mask_dir: Optional[str | Path] = None,
    mask_pattern: str = "mask_{index}.bmp",
    distortion_model: str = "matlab_legacy",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    val_count: Optional[int] = None,
    test_count: Optional[int] = None,
) -> Path:
    camera_params = Path(camera_params).resolve()
    phase_dir = Path(phase_dir).resolve()
    output_path = Path(output_path).resolve()
    calibration = load_camera_parameters(camera_params)
    phase_paths = sorted(phase_dir.glob(phase_glob), key=_natural_key)
    pose_count = calibration["R"].shape[0]
    if len(phase_paths) != pose_count:
        raise ValueError(
            f"相位图数量 {len(phase_paths)} 与标定位姿数量 {pose_count} 不一致。"
            "请确保按 cameraParams 的图像顺序对齐。"
        )
    first_phase = np.asarray(load_array(phase_paths[0], "phase")).squeeze()
    if first_phase.ndim != 2:
        raise ValueError("相位图必须为 HxW")
    height, width = first_phase.shape
    split_names = _split_names(
        pose_count,
        val_fraction,
        test_fraction,
        seed,
        val_count=val_count,
        test_count=test_count,
    )
    samples: list[dict[str, Any]] = []
    mask_root = None if mask_dir is None else Path(mask_dir).resolve()
    for index, (phase_path, rotation, translation, split) in enumerate(
        zip(phase_paths, calibration["R"], calibration["T"], split_names), start=1
    ):
        # 复现 get_lut_cali.m: Corner_coordinate = World_coordinate * R + T
        camera_points = calibration["world_points"] @ rotation + translation
        plane = fit_plane(camera_points)
        sample: dict[str, Any] = {
            "id": f"plane_{index:03d}",
            "plane_id": f"plane_{index:03d}",
            "split": split,
            "phase": str(phase_path),
            "plane": plane.tolist(),
        }
        if mask_root is not None:
            mask_path = mask_root / mask_pattern.format(index=index)
            if mask_path.exists():
                sample["mask"] = str(mask_path)
        samples.append(sample)
    manifest = {
        "format_version": 1,
        "description": "Direct MLP calibration dataset generated from MATLAB camera calibration poses",
        "phase_type": "absolute_unwrapped",
        "phase_unit": "rad",
        "depth_unit": "mm",
        "depth_definition": "camera_axis_Z",
        "label_source": "camera-ray/calibration-board-plane intersection; independent of phase-to-point LUT",
        "image_size": [height, width],
        "camera": {
            "intrinsics": calibration["K"].tolist(),
            "radial_distortion": calibration["radial"].tolist(),
            "tangential_distortion": calibration["tangential"].tolist(),
            "distortion_model": distortion_model,
        },
        "source_camera_params": str(camera_params),
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 cameraParams 和 MATLAB phase 生成 Direct MLP 数据 manifest"
    )
    parser.add_argument("--camera-params", required=True)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--phase-glob", default="*.mat")
    parser.add_argument("--mask-dir")
    parser.add_argument("--mask-pattern", default="mask_{index}.bmp")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--distortion-model",
        choices=["none", "matlab_legacy", "brown_conrady"],
        default="matlab_legacy",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_manifest(
        args.camera_params,
        args.phase_dir,
        args.output,
        args.phase_glob,
        args.mask_dir,
        args.mask_pattern,
        args.distortion_model,
        args.val_fraction,
        args.test_fraction,
        args.seed,
        args.val_count,
        args.test_count,
    )
    print(f"manifest 已保存：{result}")


if __name__ == "__main__":
    main()
