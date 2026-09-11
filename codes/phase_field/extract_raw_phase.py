from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.io import savemat


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=-1)
    return array.astype(np.float32)


def wrapped_phase_from_steps(step_images: Sequence[np.ndarray]) -> np.ndarray:
    """复现 ``psp_get_phase.m`` 的 N 步相移公式，输出 ``[-pi,pi]`` 相位。"""
    if len(step_images) < 3:
        raise ValueError("相移步数至少为 3")
    shape = np.asarray(step_images[0]).shape
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    step_count = len(step_images)
    for step, image in enumerate(step_images):
        array = np.asarray(image, dtype=np.float64)
        if array.shape != shape:
            raise ValueError("同一频率的相移图尺寸不一致")
        angle = 2.0 * step * np.pi / step_count
        numerator += array * np.sin(angle)
        denominator += array * np.cos(angle)
    return (-np.arctan2(numerator, denominator)).astype(np.float32)


def hierarchical_unwrap(
    wrapped_phases: Sequence[np.ndarray], frequency_ratios: Sequence[float]
) -> np.ndarray:
    """复现 ``psp_unwrap.m`` 的多频时间解包裹，全程不使用 mask。"""
    if len(wrapped_phases) < 1:
        raise ValueError("至少需要一个频率的相位")
    if len(frequency_ratios) != len(wrapped_phases) - 1:
        raise ValueError("frequency_ratios 数量必须是频率数量减 1")
    unwrapped = np.asarray(wrapped_phases[0], dtype=np.float64) + np.pi
    for wrapped, ratio in zip(wrapped_phases[1:], frequency_ratios):
        high = np.asarray(wrapped, dtype=np.float64) + np.pi
        fringe_order = np.rint((float(ratio) * unwrapped - high) / (2.0 * np.pi))
        unwrapped = high + fringe_order * (2.0 * np.pi)
    return unwrapped.astype(np.float32)


def extract_raw_phases(
    image_dir: str | Path,
    output_dir: str | Path,
    num_frequencies: int = 4,
    num_steps: int = 12,
    frequency_ratios: Sequence[float] = (4.0, 4.0, 4.0),
    blank_images_per_pose: int = 0,
    image_glob: str = "*.bmp",
    overwrite: bool = False,
) -> list[Path]:
    """直接从原始相移图生成绝对相位，不用 mask，不做 CPC 拟合。"""
    image_root = Path(image_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(image_root.glob(image_glob), key=_natural_key)
    if blank_images_per_pose < 0:
        raise ValueError("blank_images_per_pose 不能为负数")
    per_pose = blank_images_per_pose + num_frequencies * num_steps
    if len(files) == 0 or len(files) % per_pose != 0:
        raise ValueError(
            f"原始图数量 {len(files)} 不能被每组图数 {per_pose} 整除："
            f"{num_frequencies} 频×{num_steps} 步 + "
            f"{blank_images_per_pose} 张额外参考图"
        )
    if len(frequency_ratios) != num_frequencies - 1:
        raise ValueError("frequency_ratios 数量必须是 num_frequencies-1")
    pose_count = len(files) // per_pose
    outputs: list[Path] = []
    generated_outputs: list[str] = []
    reused_outputs: list[str] = []
    for pose_index in range(pose_count):
        output_path = output_root / f"PSP_{pose_index + 1}.mat"
        outputs.append(output_path)
        if output_path.exists() and not overwrite:
            print(f"skip existing: {output_path.name}")
            reused_outputs.append(output_path.name)
            continue
        group = files[pose_index * per_pose : (pose_index + 1) * per_pose]
        pattern_files = group[blank_images_per_pose:]
        wrapped: list[np.ndarray] = []
        for frequency_index in range(num_frequencies):
            start = frequency_index * num_steps
            step_paths = pattern_files[start : start + num_steps]
            wrapped.append(
                wrapped_phase_from_steps([_read_gray(p) for p in step_paths])
            )
        phase = hierarchical_unwrap(wrapped, frequency_ratios)
        savemat(output_path, {"phase": phase}, do_compression=True)
        generated_outputs.append(output_path.name)
        preview = (phase - np.nanmin(phase)) / max(
            float(np.nanmax(phase) - np.nanmin(phase)), 1e-8
        )
        Image.fromarray(np.round(preview * 255.0).astype(np.uint8)).save(
            output_root / f"i_PSP_{pose_index + 1}.bmp"
        )
        print(
            f"pose {pose_index + 1:03d}/{pose_count}: "
            f"phase=[{np.nanmin(phase):.4f}, {np.nanmax(phase):.4f}]"
        )
    metadata = {
        "method": "multi_frequency_phase_shift_hierarchical_unwrap",
        "phase_type": "absolute_unwrapped",
        "uses_mask": False,
        "uses_cpc": False,
        "image_dir": str(image_root),
        "output_dir": str(output_root),
        "image_glob": image_glob,
        "source_image_count": len(files),
        "source_images": [str(path) for path in files],
        "pose_count": pose_count,
        "num_frequencies": int(num_frequencies),
        "num_steps": int(num_steps),
        "frequency_ratios": [float(value) for value in frequency_ratios],
        "blank_images_per_pose": int(blank_images_per_pose),
        "images_per_pose": per_pose,
        "generated_outputs": generated_outputs,
        "reused_existing_outputs": reused_outputs,
        "provenance_status": (
            "all_outputs_generated_from_listed_sources"
            if not reused_outputs
            else "contains_existing_outputs_not_recomputed_in_this_run"
        ),
    }
    with (output_root / "phase_extraction_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="从原始相移图提取无 mask、无 CPC 拟合的绝对相位")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frequencies", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--frequency-ratios", type=float, nargs="+", default=[4, 4, 4])
    parser.add_argument(
        "--blank-images-per-pose",
        type=int,
        default=0,
        help="每组条纹前需跳过的额外参考图数；白图单独存放时保持0",
    )
    parser.add_argument("--image-glob", default="*.bmp")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = extract_raw_phases(
        args.image_dir,
        args.output,
        args.num_frequencies,
        args.num_steps,
        args.frequency_ratios,
        args.blank_images_per_pose,
        args.image_glob,
        args.overwrite,
    )
    print(f"完成：{len(outputs)} 张原始绝对相位图，保存于 {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
