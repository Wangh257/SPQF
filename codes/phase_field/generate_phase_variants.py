from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.io import savemat

from .cpc_phase import cpc_fit_phase
from .data import load_array


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _rectangle_mask(
    shape: tuple[int, int], region: Sequence[int] | None
) -> np.ndarray:
    if region is None:
        return np.ones(shape, dtype=bool)
    if len(region) != 4:
        raise ValueError("region 必须是 x y width height")
    x, y, width, height = (int(value) for value in region)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("region 要求 x/y>=0 且 width/height>0")
    image_height, image_width = shape
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"region={(x, y, width, height)} 超出相位尺寸 {shape}"
        )
    selected = np.zeros(shape, dtype=bool)
    selected[y : y + height, x : x + width] = True
    return selected


def _mask_path(
    mask_dir: Path,
    mask_pattern: str,
    source: Path,
    index: int,
) -> Path:
    try:
        relative = mask_pattern.format(
            index=index,
            name=source.name,
            stem=source.stem,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("mask-pattern 可使用 {index}、{name}、{stem}") from exc
    return (mask_dir / relative).resolve()


def _random_fringe_jump_blocks(
    phase: np.ndarray,
    rng: np.random.Generator,
    region: Sequence[int] | None,
    orders: int,
    block_count: int,
    block_sizes: Sequence[int],
) -> np.ndarray:
    """Add independent random +/- integer-period jumps to square blocks."""
    if block_count < 1:
        raise ValueError("block_count 必须大于 0")
    sizes = tuple(int(size) for size in block_sizes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("block_sizes 必须包含正整数")
    jump_orders = abs(int(orders))
    if jump_orders < 1:
        raise ValueError("随机小块跳变的 orders 不能为 0")

    height, width = phase.shape
    if region is None:
        region_x, region_y, region_width, region_height = 0, 0, width, height
    else:
        _rectangle_mask(phase.shape, region)  # 统一执行范围检查。
        region_x, region_y, region_width, region_height = (
            int(value) for value in region
        )
    too_large = [
        size
        for size in sizes
        if size > region_width or size > region_height
    ]
    if too_large:
        raise ValueError(
            f"block_sizes={too_large} 超出允许区域 "
            f"{(region_x, region_y, region_width, region_height)}"
        )

    output = phase.copy()
    jump = np.float32(2.0 * np.pi * jump_orders)
    for _ in range(block_count):
        size = int(rng.choice(sizes))
        x = int(rng.integers(region_x, region_x + region_width - size + 1))
        y = int(rng.integers(region_y, region_y + region_height - size + 1))
        sign = int(rng.choice((-1, 1)))
        block = output[y : y + size, x : x + size]
        valid = np.isfinite(block)
        block[valid] += np.float32(sign) * jump
    return output


def _random_fringe_jump_pixels(
    phase: np.ndarray,
    rng: np.random.Generator,
    region: Sequence[int] | None,
    orders: int,
    pixel_count: int,
) -> np.ndarray:
    """Add independent random +/- integer-period jumps to sparse pixels."""
    if pixel_count < 1:
        raise ValueError("pixel_count 必须大于 0")
    jump_orders = abs(int(orders))
    if jump_orders < 1:
        raise ValueError("随机像素跳变的 orders 不能为 0")
    candidates = np.flatnonzero(
        _rectangle_mask(phase.shape, region).ravel() & np.isfinite(phase.ravel())
    )
    if pixel_count > candidates.size:
        raise ValueError(
            f"pixel_count={pixel_count} 超过允许区域内有效像素数 {candidates.size}"
        )
    selected = rng.choice(candidates, size=pixel_count, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=pixel_count)
    output = phase.copy()
    output.ravel()[selected] += signs * np.float32(2.0 * np.pi * jump_orders)
    return output


def transform_phase(
    phase: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    region: Sequence[int] | None = None,
    sigma: float = 0.0,
    mean: float = 0.0,
    value: float = 0.0,
    orders: int = 1,
    cpc_mask: np.ndarray | None = None,
    cpc_order: int = 2,
    cpc_ridge: float = 1e-12,
    chunk_pixels: int = 200_000,
    block_count: int = 10,
    block_sizes: Sequence[int] = (5, 10, 20),
    pixel_count: int = 500,
) -> np.ndarray:
    """Create one deterministic phase variant while preserving invalid pixels."""
    source = np.asarray(phase, dtype=np.float32).squeeze()
    if source.ndim != 2:
        raise ValueError(f"相位必须是 HxW，当前为 {source.shape}")
    if mode == "cpc":
        if region is not None:
            raise ValueError("CPC 拟合使用 mask 选点，不同时使用 --region")
        if cpc_mask is None:
            raise ValueError("CPC 模式必须提供 mask")
        return cpc_fit_phase(
            source,
            cpc_mask,
            order=cpc_order,
            ridge=cpc_ridge,
            chunk_pixels=chunk_pixels,
        )

    output = source.copy()
    selected = _rectangle_mask(source.shape, region) & np.isfinite(source)
    if mode == "copy":
        return output
    if mode == "gaussian":
        if sigma < 0:
            raise ValueError("sigma 不能为负数")
        noise = rng.normal(mean, sigma, size=int(selected.sum())).astype(np.float32)
        output[selected] += noise
        return output
    if mode == "offset":
        output[selected] += np.float32(value)
        return output
    if mode in {"fringe_jump", "fringe_jump_region"}:
        output[selected] += np.float32(2.0 * np.pi * int(orders))
        return output
    if mode == "fringe_jump_blocks":
        return _random_fringe_jump_blocks(
            source,
            rng,
            region,
            orders,
            block_count,
            block_sizes,
        )
    if mode == "fringe_jump_pixels":
        return _random_fringe_jump_pixels(
            source,
            rng,
            region,
            orders,
            pixel_count,
        )
    raise ValueError(f"不支持的相位变体模式：{mode}")


def generate_phase_variants(
    input_dir: str | Path,
    output_dir: str | Path,
    mode: str,
    input_pattern: str = "PSP_*.mat",
    region: Sequence[int] | None = None,
    sigma: float = 0.0,
    mean: float = 0.0,
    value: float = 0.0,
    orders: int = 1,
    seed: int = 42,
    mask_dir: str | Path | None = None,
    mask_pattern: str = "mask_{index}.bmp",
    cpc_order: int = 2,
    cpc_ridge: float = 1e-12,
    chunk_pixels: int = 200_000,
    overwrite: bool = False,
    block_count: int = 10,
    block_sizes: Sequence[int] = (5, 10, 20),
    pixel_count: int = 500,
) -> dict[str, Any]:
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"输入相位目录不存在：{input_root}")
    if input_root == output_root:
        raise ValueError("输出目录不能与原始相位目录相同，避免覆盖原始相位")
    sources = sorted(input_root.glob(input_pattern), key=_natural_key)
    if not sources:
        raise FileNotFoundError(
            f"{input_root} 中没有匹配 {input_pattern!r} 的相位文件"
        )
    resolved_mask_dir = None if mask_dir is None else Path(mask_dir).resolve()
    if mode == "cpc":
        if resolved_mask_dir is None or not resolved_mask_dir.is_dir():
            raise FileNotFoundError("CPC 模式需要有效的 --mask-dir")
    output_root.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "mode": mode,
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "input_pattern": input_pattern,
        "region_xywh": None if region is None else [int(x) for x in region],
        "sigma_rad": float(sigma),
        "mean_rad": float(mean),
        "offset_rad": float(value),
        "fringe_jump_orders": int(orders),
        "fringe_jump_block_count": int(block_count),
        "fringe_jump_block_sizes": [int(size) for size in block_sizes],
        "fringe_jump_pixel_count": int(pixel_count),
        "seed": int(seed),
        "mask_dir": None if resolved_mask_dir is None else str(resolved_mask_dir),
        "mask_pattern": mask_pattern,
        "cpc_order": int(cpc_order),
        "cpc_ridge": float(cpc_ridge),
    }
    generated: list[dict[str, Any]] = []
    for index, source_path in enumerate(sources, start=1):
        output_path = output_root / f"{source_path.stem}.mat"
        if output_path.exists() and not overwrite:
            print(f"skip existing: {output_path.name}")
            generated.append(
                {
                    "index": index,
                    "source": str(source_path),
                    "output": str(output_path),
                    "status": "skipped_existing",
                }
            )
            continue
        phase = np.asarray(load_array(source_path, "phase"), dtype=np.float32).squeeze()
        cpc_mask = None
        current_mask_path: Path | None = None
        if mode == "cpc":
            assert resolved_mask_dir is not None
            current_mask_path = _mask_path(
                resolved_mask_dir, mask_pattern, source_path, index
            )
            if not current_mask_path.is_file():
                raise FileNotFoundError(f"第 {index} 张相位的 mask 不存在：{current_mask_path}")
            cpc_mask = load_array(current_mask_path)
        rng = np.random.default_rng(int(seed) + index - 1)
        transformed = transform_phase(
            phase=phase,
            mode=mode,
            rng=rng,
            region=region,
            sigma=sigma,
            mean=mean,
            value=value,
            orders=orders,
            cpc_mask=cpc_mask,
            cpc_order=cpc_order,
            cpc_ridge=cpc_ridge,
            chunk_pixels=chunk_pixels,
            block_count=block_count,
            block_sizes=block_sizes,
            pixel_count=pixel_count,
        )
        valid = np.isfinite(phase) & np.isfinite(transformed)
        difference = transformed[valid] - phase[valid]
        item_metadata = {
            **config,
            "index": index,
            "source": str(source_path),
            "output": str(output_path),
            "mask": None if current_mask_path is None else str(current_mask_path),
            "changed_pixels": int(np.count_nonzero(difference)),
            "difference_mean_rad": float(difference.mean()) if difference.size else 0.0,
            "difference_std_rad": float(difference.std()) if difference.size else 0.0,
            "difference_min_rad": float(difference.min()) if difference.size else 0.0,
            "difference_max_rad": float(difference.max()) if difference.size else 0.0,
        }
        savemat(
            output_path,
            {
                "phase": transformed.astype(np.float32),
                "phase_variant_metadata_json": np.array(
                    json.dumps(item_metadata, ensure_ascii=False)
                ),
            },
            do_compression=True,
        )
        generated.append({**item_metadata, "status": "generated"})
        print(
            f"{index:03d}/{len(sources)} {source_path.name} -> {output_path.name} | "
            f"delta mean={item_metadata['difference_mean_rad']:.6g}, "
            f"std={item_metadata['difference_std_rad']:.6g} rad"
        )

    summary = {**config, "file_count": len(sources), "files": generated}
    with (output_root / "phase_variant_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从已缓存的展开相位批量生成噪声、偏置、跳周或 CPC 变体"
    )
    parser.add_argument("--input-dir", required=True, help="原始展开相位目录")
    parser.add_argument("--output-dir", required=True, help="新相位目录")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "copy",
            "gaussian",
            "offset",
            "fringe_jump",
            "fringe_jump_region",
            "fringe_jump_blocks",
            "fringe_jump_pixels",
            "cpc",
        ],
    )
    parser.add_argument("--input-pattern", default="PSP_*.mat")
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="只在指定矩形内加干扰；不给则处理整张图",
    )
    parser.add_argument("--sigma", type=float, default=0.0, help="高斯噪声标准差/rad")
    parser.add_argument("--mean", type=float, default=0.0, help="高斯噪声均值/rad")
    parser.add_argument("--value", type=float, default=0.0, help="offset 的相位偏置/rad")
    parser.add_argument("--orders", type=int, default=1, help="跳周数，1 表示 +2pi")
    parser.add_argument(
        "--block-count",
        type=int,
        default=10,
        help="fringe_jump_blocks 的随机小块数量",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="fringe_jump_blocks 随机候选边长，例如 5 10 20",
    )
    parser.add_argument(
        "--pixel-count",
        type=int,
        default=500,
        help="fringe_jump_pixels 的随机零散像素数量",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-dir", help="CPC 所用 mask 目录")
    parser.add_argument("--mask-pattern", default="mask_{index}.bmp")
    parser.add_argument("--cpc-order", type=int, default=2)
    parser.add_argument("--cpc-ridge", type=float, default=1e-12)
    parser.add_argument("--chunk-pixels", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = generate_phase_variants(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        input_pattern=args.input_pattern,
        region=args.region,
        sigma=args.sigma,
        mean=args.mean,
        value=args.value,
        orders=args.orders,
        seed=args.seed,
        mask_dir=args.mask_dir,
        mask_pattern=args.mask_pattern,
        cpc_order=args.cpc_order,
        cpc_ridge=args.cpc_ridge,
        chunk_pixels=args.chunk_pixels,
        overwrite=args.overwrite,
        block_count=args.block_count,
        block_sizes=args.block_sizes,
        pixel_count=args.pixel_count,
    )
    print(
        f"完成：{summary['file_count']} 张 {summary['mode']} 相位，"
        f"保存于 {summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
