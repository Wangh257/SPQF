from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.io import savemat

from .data import load_array
from .polynomial_model import load_polynomial_model
from .predictor import DirectMLPPredictor


def _load_mask(mask_path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    if mask_path is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(load_array(mask_path)).squeeze().astype(bool)
    if mask.shape != shape:
        raise ValueError(f"mask 尺寸 {mask.shape} 与相位尺寸 {shape} 不一致")
    return mask


def _save_sensitivity_previews(
    output_dir: Path,
    signed: np.ndarray,
    absolute: np.ndarray,
    valid: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("未安装 matplotlib，跳过相位敏感度 PNG 图片", stacklevel=2)
        return

    signed_values = signed[valid]
    absolute_values = absolute[valid]
    signed_limit = float(np.percentile(np.abs(signed_values), 99))
    absolute_limit = float(np.percentile(absolute_values, 99))
    signed_limit = max(signed_limit, np.finfo(np.float32).eps)
    absolute_limit = max(absolute_limit, np.finfo(np.float32).eps)

    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(
        np.ma.masked_where(~valid, signed),
        cmap="coolwarm",
        vmin=-signed_limit,
        vmax=signed_limit,
    )
    axis.set_title("Signed phase sensitivity dZ/dPhi")
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="mm/rad (clipped at |P99|)")
    figure.tight_layout()
    figure.savefig(output_dir / "sensitivity_signed.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(
        np.ma.masked_where(~valid, absolute),
        cmap="magma",
        vmin=0.0,
        vmax=absolute_limit,
    )
    axis.set_title("Absolute phase sensitivity |dZ/dPhi|")
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="mm/rad (clipped at P99)")
    figure.tight_layout()
    figure.savefig(output_dir / "sensitivity_abs.png", dpi=160)
    plt.close(figure)


def _model_predictor(
    model_path: Path,
    phase_shape: tuple[int, int],
    amplitude: np.ndarray | None,
    device: str,
    chunk_size: int,
) -> tuple[str, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        predictor = DirectMLPPredictor(model_path, device)
        if tuple(predictor.image_size) != phase_shape:
            raise ValueError(
                f"相位尺寸 {phase_shape} 与 MLP 训练尺寸 {predictor.image_size} 不一致"
            )

        def predict(current_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            _, depth, valid = predictor.predict(
                current_phase,
                amplitude=amplitude,
                chunk_size=chunk_size,
                warn_out_of_range=False,
            )
            return depth, valid

        return "direct_mlp", predict

    if suffix == ".npz":
        lut = load_polynomial_model(model_path)
        if lut.image_size != phase_shape:
            raise ValueError(
                f"相位尺寸 {phase_shape} 与多项式 LUT 尺寸 {lut.image_size} 不一致"
            )

        def predict(current_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            depth = lut.predict_z(current_phase)
            return depth, np.isfinite(current_phase) & np.isfinite(depth)

        method = str(lut.metadata.get("method", "polynomial"))
        return f"{method}_degree_{lut.degree}", predict

    raise ValueError("模型必须是 Direct MLP 的 .pt 或多项式 LUT 的 .npz")


def compute_phase_sensitivity(
    model_path: str | Path,
    phase_path: str | Path,
    output_dir: str | Path,
    epsilon: float = 0.01,
    mask_path: str | Path | None = None,
    amplitude_path: str | Path | None = None,
    device: str = "auto",
    chunk_size: int = 262_144,
) -> dict[str, Any]:
    """Compute and save the local depth sensitivity to phase in mm/rad."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须是大于 0 的有限数，单位为 rad")
    model = Path(model_path).resolve()
    phase_file = Path(str(phase_path).split("::", 1)[0]).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if not phase_file.is_file():
        raise FileNotFoundError(phase_file)

    phase = np.asarray(load_array(phase_path, "phase"), dtype=np.float32).squeeze()
    if phase.ndim != 2:
        raise ValueError(f"相位必须是 HxW，当前为 {phase.shape}")
    amplitude = (
        None
        if amplitude_path is None
        else np.asarray(load_array(amplitude_path), dtype=np.float32).squeeze()
    )
    if amplitude is not None and amplitude.shape != phase.shape:
        raise ValueError(
            f"amplitude 尺寸 {amplitude.shape} 与相位尺寸 {phase.shape} 不一致"
        )
    requested_mask = _load_mask(mask_path, phase.shape)
    model_type, predict = _model_predictor(
        model, phase.shape, amplitude, device, chunk_size
    )

    clean_depth, clean_valid = predict(phase)
    plus_depth, plus_valid = predict(phase + np.float32(epsilon))
    minus_depth, minus_valid = predict(phase - np.float32(epsilon))
    valid = (
        requested_mask
        & clean_valid
        & plus_valid
        & minus_valid
        & np.isfinite(clean_depth)
        & np.isfinite(plus_depth)
        & np.isfinite(minus_depth)
    )
    if not np.any(valid):
        raise ValueError("没有可计算相位敏感度的有效像素")

    sensitivity = np.full(phase.shape, np.nan, dtype=np.float32)
    sensitivity[valid] = (
        (plus_depth[valid] - minus_depth[valid]) / (2.0 * float(epsilon))
    ).astype(np.float32)
    sensitivity_abs = np.abs(sensitivity)
    clean_depth = np.asarray(clean_depth, dtype=np.float32).copy()
    clean_depth[~valid] = np.nan
    values = sensitivity[valid].astype(np.float64)
    absolute_values = np.abs(values)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "sensitivity_mm_per_rad.npy", sensitivity)
    np.save(output / "sensitivity_abs_mm_per_rad.npy", sensitivity_abs)
    np.save(output / "clean_depth_mm.npy", clean_depth)
    np.save(output / "valid_mask.npy", valid)
    savemat(
        output / "phase_sensitivity.mat",
        {
            "sensitivity_mm_per_rad": sensitivity,
            "sensitivity_abs_mm_per_rad": sensitivity_abs,
            "clean_depth_mm": clean_depth,
            "valid_mask": valid.astype(np.uint8),
            "epsilon_rad": np.float64(epsilon),
        },
        do_compression=True,
    )

    summary: dict[str, Any] = {
        "definition": "(Z(phase+epsilon)-Z(phase-epsilon))/(2*epsilon)",
        "model_type": model_type,
        "model": str(model),
        "phase": str(phase_file),
        "mask": None if mask_path is None else str(Path(mask_path).resolve()),
        "amplitude": (
            None if amplitude_path is None else str(Path(amplitude_path).resolve())
        ),
        "image_size": [int(phase.shape[0]), int(phase.shape[1])],
        "epsilon_rad": float(epsilon),
        "unit": "mm/rad",
        "valid_pixels": int(valid.sum()),
        "signed_mean": float(values.mean()),
        "signed_min": float(values.min()),
        "signed_max": float(values.max()),
        "absolute_mean": float(absolute_values.mean()),
        "absolute_median": float(np.median(absolute_values)),
        "absolute_p95": float(np.percentile(absolute_values, 95)),
        "absolute_p99": float(np.percentile(absolute_values, 99)),
        "absolute_max": float(absolute_values.max()),
    }
    with (output / "sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _save_sensitivity_previews(output, sensitivity, sensitivity_abs, valid)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 MLP 或逐像素多项式的相位—深度敏感度图"
    )
    parser.add_argument("--model", required=True, help="MLP .pt 或多项式 .npz")
    parser.add_argument("--phase", required=True, help="一张展开相位 .mat/.npy/.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="中心差分相位步长/rad，默认 0.01",
    )
    parser.add_argument("--mask", help="可选，仅统计并显示 mask 内像素")
    parser.add_argument("--amplitude", help="仅在 MLP 训练使用振幅特征时提供")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=262_144)
    args = parser.parse_args()
    summary = compute_phase_sensitivity(
        args.model,
        args.phase,
        args.output,
        args.epsilon,
        args.mask,
        args.amplitude,
        args.device,
        args.chunk_size,
    )
    print(
        f"完成：{summary['model_type']}，有效像素 {summary['valid_pixels']}，"
        f"|dZ/dPhi| P95={summary['absolute_p95']:.6g} mm/rad，"
        f"输出位于 {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
