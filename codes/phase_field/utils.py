from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("读取训练配置需要 PyYAML") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("配置文件顶层必须为 mapping")
    return value


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def regression_metrics(
    prediction_mm: np.ndarray, target_mm: np.ndarray
) -> dict[str, float]:
    prediction = np.asarray(prediction_mm, dtype=np.float64)
    target = np.asarray(target_mm, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        raise ValueError("没有可评估的有效深度")
    error = prediction[valid] - target[valid]
    absolute = np.abs(error)
    return {
        "count": int(error.size),
        "mae_mm": float(absolute.mean()),
        "rmse_mm": float(np.sqrt(np.square(error).mean())),
        "median_ae_mm": float(np.median(absolute)),
        "p95_ae_mm": float(np.percentile(absolute, 95)),
        "max_ae_mm": float(absolute.max()),
        "bias_mm": float(error.mean()),
    }


def camera_to_dict(camera: Any) -> dict[str, Any]:
    return {
        "intrinsics": np.asarray(camera.intrinsics).tolist(),
        "radial_distortion": list(camera.radial_distortion),
        "tangential_distortion": list(camera.tangential_distortion),
        "distortion_model": camera.distortion_model,
    }
