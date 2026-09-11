from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f

from .data import (
    CalibrationDataset,
    FeatureSpec,
    build_features,
    load_array,
    validate_absolute_phase,
)
from .models import PhaseToDepthMLP
from .normalization import NormalizationStats
from .utils import (
    camera_to_dict,
    choose_device,
    load_yaml,
    regression_metrics,
    seed_everything,
)


def _loss_per_pixel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    name: str,
    beta: float,
) -> torch.Tensor:
    if name == "smooth_l1":
        return torch_f.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    if name == "l1":
        return torch.abs(prediction - target)
    if name == "mse":
        return torch.square(prediction - target)
    raise ValueError("loss 必须是 smooth_l1、l1 或 mse")


@torch.no_grad()
def evaluate_random_points(
    model: PhaseToDepthMLP,
    dataset: CalibrationDataset,
    split: str,
    stats: NormalizationStats,
    feature_spec: FeatureSpec,
    max_points_per_plane: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
    mask_dir: str | Path | None = None,
    mask_pattern: str = "mask_{index}.bmp",
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    resolved_mask_dir = None if mask_dir is None else Path(mask_dir).resolve()
    for index in dataset.indices(split):
        record = dataset.records[index]
        arrays = dataset.load(index)
        evaluation_valid = arrays.valid.copy()
        if resolved_mask_dir is not None:
            mask_path = resolved_mask_dir / mask_pattern.format(
                index=index + 1,
                id=record.sample_id,
                sample_id=record.sample_id,
                plane_id=record.plane_id,
            )
            if not mask_path.is_file():
                raise FileNotFoundError(f"评价 mask 不存在：{mask_path}")
            evaluation_mask = np.asarray(load_array(mask_path)).squeeze().astype(bool)
            if evaluation_mask.shape != arrays.valid.shape:
                raise ValueError(
                    f"{mask_path} 的 mask 尺寸为 {evaluation_mask.shape}，"
                    f"期望 {arrays.valid.shape}"
                )
            evaluation_valid &= evaluation_mask
        flat = np.flatnonzero(evaluation_valid)
        if max_points_per_plane > 0 and flat.size > max_points_per_plane:
            flat = rng.choice(flat, size=max_points_per_plane, replace=False)
        v, u = np.unravel_index(flat, (dataset.height, dataset.width))
        phase = arrays.phase.ravel()[flat]
        amplitude = None if arrays.amplitude is None else arrays.amplitude.ravel()[flat]
        features = build_features(
            u,
            v,
            phase,
            dataset.height,
            dataset.width,
            stats,
            feature_spec,
            amplitude,
        )
        plane_predictions: list[np.ndarray] = []
        for start in range(0, features.shape[0], chunk_size):
            tensor = torch.from_numpy(features[start : start + chunk_size]).to(device)
            zn = model(tensor).float().cpu().numpy()
            plane_predictions.append(stats.denormalize_depth(zn))
        predictions.append(np.concatenate(plane_predictions))
        targets.append(arrays.depth_mm.ravel()[flat])
    if not predictions:
        raise ValueError(f"数据集没有 split={split!r}，无法评估")
    return regression_metrics(np.concatenate(predictions), np.concatenate(targets))


def _checkpoint(
    model: PhaseToDepthMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_mae: float,
    config: dict[str, Any],
    stats: NormalizationStats,
    feature_spec: FeatureSpec,
    dataset: CalibrationDataset,
    selection_split: str | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "direct_mlp",
        "epoch": epoch,
        "best_val_mae_mm": best_val_mae,
        "best_selection_mae_mm": best_val_mae,
        "selection_split": selection_split,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config_dict(),
        "feature_spec": feature_spec.to_dict(),
        "normalization": stats.to_dict(),
        "image_size": [dataset.height, dataset.width],
        "camera": camera_to_dict(dataset.camera),
        "phase_type": dataset.phase_type,
        "depth_unit": dataset.depth_unit,
        "train_plane_ids": [
            dataset.records[index].plane_id for index in dataset.indices("train")
        ],
        "val_plane_ids": [
            dataset.records[index].plane_id for index in dataset.indices("val")
        ],
        "test_plane_ids": [
            dataset.records[index].plane_id for index in dataset.indices("test")
        ],
        "config": config,
    }


def train(
    config_path: str | Path,
    phase_dir: str | Path | None = None,
    phase_pattern: str | None = None,
    output_dir_override: str | Path | None = None,
    evaluation_mask_dir: str | Path | None = None,
    evaluation_mask_pattern: str = "mask_{index}.bmp",
    manifest_override: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = copy.deepcopy(load_yaml(config_path))
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    dataset_config = config["dataset"]
    if manifest_override is not None:
        dataset_config["manifest"] = str(Path(manifest_override).resolve())
    manifest_path = Path(dataset_config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()
    configured_phase_dir = dataset_config.get("phase_dir")
    if phase_dir is not None:
        selected_phase_dir = Path(phase_dir).resolve()
    elif configured_phase_dir:
        selected_phase_dir = Path(configured_phase_dir)
        if not selected_phase_dir.is_absolute():
            selected_phase_dir = (config_path.parent / selected_phase_dir).resolve()
    else:
        selected_phase_dir = None
    selected_phase_pattern = str(
        phase_pattern
        if phase_pattern is not None
        else dataset_config.get("phase_pattern", "PSP_{index}.mat")
    )
    if selected_phase_dir is not None:
        dataset_config["phase_dir"] = str(selected_phase_dir)
        dataset_config["phase_pattern"] = selected_phase_pattern
    dataset = CalibrationDataset(
        manifest_path,
        cache_size=int(dataset_config.get("cache_planes", 3)),
        phase_dir=selected_phase_dir,
        phase_pattern=selected_phase_pattern,
    )
    selected_evaluation_mask_dir = (
        None if evaluation_mask_dir is None else Path(evaluation_mask_dir).resolve()
    )
    if (
        selected_evaluation_mask_dir is not None
        and not selected_evaluation_mask_dir.is_dir()
    ):
        raise FileNotFoundError(
            f"评价 mask 目录不存在：{selected_evaluation_mask_dir}"
        )
    config["evaluation_mask_dir"] = (
        None
        if selected_evaluation_mask_dir is None
        else str(selected_evaluation_mask_dir)
    )
    config["evaluation_mask_pattern"] = evaluation_mask_pattern
    validate_absolute_phase(dataset)
    if not dataset.indices("train"):
        raise ValueError("manifest 必须包含 train 平面")
    has_validation = bool(dataset.indices("val"))

    feature_config = config.get("features", {})
    feature_spec = FeatureSpec(
        use_periodic_phase=bool(feature_config.get("use_periodic_phase", True)),
        use_amplitude=bool(feature_config.get("use_amplitude", False)),
    )
    training = config.get("training", {})
    default_selection_split = "val" if has_validation else ""
    selection_split = str(
        training.get("selection_split", default_selection_split) or ""
    ).strip()
    if selection_split and not dataset.indices(selection_split):
        raise ValueError(f"manifest 中没有 selection_split={selection_split!r}")
    has_selection_split = bool(selection_split)
    resume = training.get("resume")
    resume_state: dict[str, Any] | None = None
    if resume:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = (config_path.parent / resume_path).resolve()
        resume_state = torch.load(resume_path, map_location="cpu")
        saved_features = FeatureSpec(**resume_state["feature_spec"])
        if saved_features != feature_spec:
            raise ValueError("恢复训练时的输入特征配置与 checkpoint 不一致")
    if feature_spec.use_amplitude:
        missing = [
            dataset.records[index].sample_id
            for index in dataset.indices("train") + dataset.indices("val")
            if dataset.records[index].amplitude_path is None
        ]
        if missing:
            raise ValueError(f"训练要求 amplitude，但以下样本未提供：{missing}")
    stats = (
        NormalizationStats.from_dict(resume_state["normalization"])
        if resume_state is not None
        else dataset.compute_normalization(
            "train",
            dataset_config.get("normalization_pixels_per_plane", 500_000),
            seed,
        )
    )

    model_config = dict(config.get("model", {}))
    model_config["input_dim"] = feature_spec.input_dim
    model = PhaseToDepthMLP(**model_config)
    device = torch.device(choose_device(str(training.get("device", "auto"))))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(training.get("min_lr", 1e-6))
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rng = np.random.default_rng(seed)

    if output_dir_override is not None:
        output_dir = Path(output_dir_override).resolve()
        config["output_dir"] = str(output_dir)
    else:
        output_dir = Path(config.get("output_dir", "runs/direct_mlp"))
        if not output_dir.is_absolute():
            output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    start_epoch, best_val_mae = 1, np.inf
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_val_mae = float(
            resume_state.get(
                "best_selection_mae_mm",
                resume_state.get("best_val_mae_mm", np.inf),
            )
        )

    batch_size = int(training.get("batch_size", 32768))
    steps_per_epoch = int(training.get("steps_per_epoch", 200))
    loss_name = str(training.get("loss", "smooth_l1")).lower()
    loss_beta = float(training.get("smooth_l1_beta", 0.01))
    validate_every = int(training.get("validate_every", 1))
    val_points = int(training.get("val_pixels_per_plane", 200_000))
    chunk_size = int(training.get("inference_chunk_size", 262_144))
    patience = int(training.get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    log_path = output_dir / "metrics.csv"
    write_header = not log_path.exists() or start_epoch == 1

    print(
        f"Direct MLP | device={device} | input_dim={feature_spec.input_dim} | "
        f"train_planes={len(dataset.indices('train'))} | "
        f"val_planes={len(dataset.indices('val'))} | "
        f"test_planes={len(dataset.indices('test'))} | "
        f"selection_split={selection_split or 'none'}"
    )
    print(
        f"phase train range=[{stats.phase_min:.6g}, {stats.phase_max:.6g}], "
        f"Z train range=[{stats.depth_min_mm:.3f}, {stats.depth_max_mm:.3f}] mm"
    )
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        losses: list[float] = []
        for _ in range(steps_per_epoch):
            batch = dataset.sample_batch("train", batch_size, rng)
            features_np = build_features(
                batch["u"],
                batch["v"],
                batch["phase"],
                dataset.height,
                dataset.width,
                stats,
                feature_spec,
                batch["amplitude"] if feature_spec.use_amplitude else None,
            )
            target_np = stats.normalize_depth(batch["depth_mm"])
            features = torch.from_numpy(features_np).to(device)
            target = torch.from_numpy(target_np).to(device)
            confidence = torch.from_numpy(batch["confidence"].astype(np.float32)).to(
                device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(features)
                pixel_loss = _loss_per_pixel(prediction, target, loss_name, loss_beta)
                loss = (pixel_loss * confidence).sum() / confidence.sum().clamp_min(
                    1e-8
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        metrics: dict[str, float] = {}
        if has_selection_split and (epoch % validate_every == 0 or epoch == epochs):
            metrics = evaluate_random_points(
                model,
                dataset,
                selection_split,
                stats,
                feature_spec,
                val_points,
                chunk_size,
                device,
                seed + epoch,
                selected_evaluation_mask_dir,
                evaluation_mask_pattern,
            )
            improved = metrics["mae_mm"] < best_val_mae
            if improved:
                best_val_mae = metrics["mae_mm"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            state = _checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_mae,
                config,
                stats,
                feature_spec,
                dataset,
                selection_split,
            )
            torch.save(state, output_dir / "last.pt")
            if improved:
                torch.save(state, output_dir / "best.pt")
            elapsed = time.time() - epoch_start
            print(
                f"epoch {epoch:04d} | train_loss={np.mean(losses):.6g} | "
                f"{selection_split}_MAE={metrics['mae_mm']:.4f} mm | "
                f"{selection_split}_RMSE={metrics['rmse_mm']:.4f} mm | {elapsed:.1f}s"
            )
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "learning_rate",
                        *metrics.keys(),
                    ],
                )
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **metrics,
                    }
                )
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"early stopping: {patience} 次验证未改善")
                break
        elif not has_selection_split and (
            epoch % validate_every == 0 or epoch == epochs
        ):
            # 用户选择“其余平面全部训练”时，不使用 test 误差选模型。
            # best.pt 与 last.pt 都指向当前固定 epoch 的模型，便于统一推理入口。
            state = _checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                float("nan"),
                config,
                stats,
                feature_spec,
                dataset,
                None,
            )
            torch.save(state, output_dir / "last.pt")
            torch.save(state, output_dir / "best.pt")
            elapsed = time.time() - epoch_start
            print(
                f"epoch {epoch:04d} | train_loss={np.mean(losses):.6g} | "
                f"no validation split | {elapsed:.1f}s"
            )
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["epoch", "train_loss", "learning_rate"]
                )
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
        else:
            print(f"epoch {epoch:04d} | train_loss={np.mean(losses):.6g}")
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 (u,v,Phi) -> zn 的 Direct MLP")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--manifest", help="覆盖 YAML 中的 manifest，相对路径从运行目录解析")
    parser.add_argument(
        "--phase-dir",
        help="覆盖 manifest 中的相位路径，例如 data/phase_variants/gaussian_sigma0.05",
    )
    parser.add_argument(
        "--phase-pattern",
        default=None,
        help="相位文件名模板，默认 PSP_{index}.mat",
    )
    parser.add_argument("--output-dir", help="覆盖 YAML 中的 output_dir")
    parser.add_argument(
        "--eval-mask-dir",
        help="仅在每个 epoch 评价时使用的 mask 目录，不影响训练采样",
    )
    parser.add_argument("--eval-mask-pattern", default="mask_{index}.bmp")
    args = parser.parse_args()
    best = train(
        args.config,
        args.phase_dir,
        args.phase_pattern,
        args.output_dir,
        args.eval_mask_dir,
        args.eval_mask_pattern,
        manifest_override=args.manifest,
    )
    print(f"最佳 checkpoint: {best}")


if __name__ == "__main__":
    main()
