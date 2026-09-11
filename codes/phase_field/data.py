from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .geometry import intersect_camera_rays_with_plane, normalized_camera_rays
from .normalization import NormalizationStats, RunningStats, validate_stats


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _first_numeric_array(mapping: dict[str, Any]) -> np.ndarray:
    for key, value in mapping.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            return array
    raise ValueError("文件中没有找到数值数组")


def load_array(path_spec: str | Path, default_key: Optional[str] = None) -> np.ndarray:
    """读取 npy/npz/mat/常见图像；``file::key`` 可指定容器内字段。"""
    raw = str(path_spec)
    if "::" in raw:
        raw, key = raw.rsplit("::", 1)
    else:
        key = default_key
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key:
                if key not in data:
                    raise KeyError(f"{path} 中没有字段 {key!r}")
                return np.asarray(data[key])
            return _first_numeric_array(dict(data.items()))
    if suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("读取 .mat 需要 scipy") from exc
        data = loadmat(path, simplify_cells=True)
        selected_key = key or ("phase" if "phase" in data else None)
        if selected_key:
            if selected_key not in data:
                raise KeyError(f"{path} 中没有字段 {selected_key!r}")
            return np.asarray(data[selected_key])
        return _first_numeric_array(data)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("读取图像需要 Pillow") from exc
    with Image.open(path) as image:
        return np.asarray(image)


@dataclass(frozen=True)
class FeatureSpec:
    use_periodic_phase: bool = True
    use_amplitude: bool = False

    @property
    def input_dim(self) -> int:
        return (
            3 + (2 if self.use_periodic_phase else 0) + (1 if self.use_amplitude else 0)
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "use_periodic_phase": self.use_periodic_phase,
            "use_amplitude": self.use_amplitude,
        }


@dataclass(frozen=True)
class CameraConfig:
    intrinsics: np.ndarray
    radial_distortion: tuple[float, ...]
    tangential_distortion: tuple[float, ...]
    distortion_model: str


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    plane_id: str
    split: str
    phase_path: Path
    mask_path: Optional[Path]
    amplitude_path: Optional[Path]
    confidence_path: Optional[Path]
    depth_path: Optional[Path]
    z_mm: Optional[float]
    plane: Optional[tuple[float, float, float, float]]


@dataclass
class SampleArrays:
    phase: np.ndarray
    depth_mm: np.ndarray
    valid: np.ndarray
    amplitude: Optional[np.ndarray]
    confidence: np.ndarray


class CalibrationDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        cache_size: int = 3,
        phase_dir: str | Path | None = None,
        phase_pattern: str = "PSP_{index}.mat",
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.manifest = manifest
        self.base_dir = self.manifest_path.parent
        self.height, self.width = (int(value) for value in manifest["image_size"])
        camera = manifest["camera"]
        self.camera = CameraConfig(
            intrinsics=np.asarray(camera["intrinsics"], dtype=np.float64),
            radial_distortion=tuple(
                float(x) for x in camera.get("radial_distortion", [])
            ),
            tangential_distortion=tuple(
                float(x) for x in camera.get("tangential_distortion", [])
            ),
            distortion_model=str(camera.get("distortion_model", "none")),
        )
        self.phase_type = str(manifest.get("phase_type", "unknown"))
        self.depth_unit = str(manifest.get("depth_unit", "mm"))
        if self.depth_unit != "mm":
            raise ValueError(f"当前仅支持 mm 深度，manifest 为 {self.depth_unit!r}")
        self.records = [self._parse_record(value) for value in manifest["samples"]]
        if not self.records:
            raise ValueError("manifest 中没有样本")
        self.phase_dir = None if phase_dir is None else Path(phase_dir).resolve()
        self.phase_pattern = str(phase_pattern)
        if self.phase_dir is not None:
            if not self.phase_dir.is_dir():
                raise FileNotFoundError(f"相位目录不存在：{self.phase_dir}")
            self.records = self._override_phase_paths(self.records)
        self._validate_splits()
        self.xn, self.yn = normalized_camera_rays(
            self.height,
            self.width,
            self.camera.intrinsics,
            self.camera.radial_distortion,
            self.camera.tangential_distortion,
            self.camera.distortion_model,
        )
        self._cache: dict[int, SampleArrays] = {}
        self._cache_order: list[int] = []
        self._valid_flat_cache: dict[int, np.ndarray] = {}
        self.cache_size = max(int(cache_size), 0)

    def _override_phase_paths(
        self, records: list[SampleRecord]
    ) -> list[SampleRecord]:
        """Use one replaceable phase directory without rewriting the manifest."""
        assert self.phase_dir is not None
        output: list[SampleRecord] = []
        missing: list[Path] = []
        for index, record in enumerate(records, start=1):
            try:
                relative = self.phase_pattern.format(
                    index=index,
                    id=record.sample_id,
                    sample_id=record.sample_id,
                    plane_id=record.plane_id,
                    original_name=record.phase_path.name,
                    original_stem=record.phase_path.stem,
                )
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(
                    "phase_pattern 可使用 {index}、{id}、{sample_id}、"
                    "{plane_id}、{original_name}、{original_stem}"
                ) from exc
            phase_path = (self.phase_dir / relative).resolve()
            if not phase_path.is_file():
                missing.append(phase_path)
            output.append(replace(record, phase_path=phase_path))
        if missing:
            preview = "\n".join(f"  - {path}" for path in missing[:10])
            suffix = "" if len(missing) <= 10 else f"\n  ... 共 {len(missing)} 个"
            raise FileNotFoundError(
                f"指定相位目录缺少 manifest 所需文件：\n{preview}{suffix}"
            )
        return output

    def _parse_record(self, value: dict[str, Any]) -> SampleRecord:
        phase_value = value.get("phase")
        if not phase_value:
            raise ValueError("每个样本都必须提供 phase")
        plane = value.get("plane")
        if (
            sum(
                item is not None
                for item in (value.get("depth"), value.get("z_mm"), plane)
            )
            != 1
        ):
            raise ValueError(f"样本 {value.get('id')} 必须且只能提供 depth、z_mm、plane 之一")
        parsed_plane = None if plane is None else tuple(float(x) for x in plane)
        if parsed_plane is not None and len(parsed_plane) != 4:
            raise ValueError("plane 必须为 [A,B,C,D]")
        return SampleRecord(
            sample_id=str(value.get("id", value.get("plane_id", "sample"))),
            plane_id=str(value.get("plane_id", value.get("id", "sample"))),
            split=str(value.get("split", "train")),
            phase_path=_resolve_path(self.base_dir, str(phase_value)),  # type: ignore[arg-type]
            mask_path=_resolve_path(self.base_dir, value.get("mask")),
            amplitude_path=_resolve_path(self.base_dir, value.get("amplitude")),
            confidence_path=_resolve_path(self.base_dir, value.get("confidence")),
            depth_path=_resolve_path(self.base_dir, value.get("depth")),
            z_mm=None if value.get("z_mm") is None else float(value["z_mm"]),
            plane=parsed_plane,  # type: ignore[arg-type]
        )

    def _validate_splits(self) -> None:
        plane_splits: dict[str, set[str]] = {}
        for record in self.records:
            plane_splits.setdefault(record.plane_id, set()).add(record.split)
        leaking = {key: value for key, value in plane_splits.items() if len(value) > 1}
        if leaking:
            raise ValueError(f"发现平面级数据泄漏：{leaking}")

    def indices(self, split: str) -> list[int]:
        return [
            index for index, record in enumerate(self.records) if record.split == split
        ]

    def _check_shape(
        self, array: np.ndarray, name: str, record: SampleRecord
    ) -> np.ndarray:
        array = np.asarray(array).squeeze()
        expected = (self.height, self.width)
        if array.shape != expected:
            raise ValueError(
                f"样本 {record.sample_id} 的 {name} 尺寸为 {array.shape}，期望 {expected}。"
                "Python 按 (H,W) 读取 MATLAB phase，不需要像旧 MATLAB LUT 那样转置。"
            )
        return array

    def load(self, index: int) -> SampleArrays:
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        phase = self._check_shape(
            load_array(record.phase_path, "phase"), "phase", record
        ).astype(np.float32)
        if record.depth_path is not None:
            depth = self._check_shape(
                load_array(record.depth_path, "depth"), "depth", record
            ).astype(np.float32)
        elif record.z_mm is not None:
            depth = np.full((self.height, self.width), record.z_mm, dtype=np.float32)
        else:
            depth = intersect_camera_rays_with_plane(
                self.xn, self.yn, record.plane or ()
            )
        if record.mask_path is None:
            valid = np.ones((self.height, self.width), dtype=bool)
        else:
            valid = self._check_shape(
                load_array(record.mask_path), "mask", record
            ).astype(bool)
        amplitude = None
        if record.amplitude_path is not None:
            amplitude = self._check_shape(
                load_array(record.amplitude_path), "amplitude", record
            ).astype(np.float32)
        if record.confidence_path is None:
            confidence = np.ones((self.height, self.width), dtype=np.float32)
        else:
            confidence = self._check_shape(
                load_array(record.confidence_path), "confidence", record
            ).astype(np.float32)
            confidence = np.clip(confidence, 0.0, None)
        valid &= np.isfinite(phase) & np.isfinite(depth) & (depth > 0)
        valid &= np.isfinite(confidence) & (confidence > 0)
        if amplitude is not None:
            valid &= np.isfinite(amplitude)
        arrays = SampleArrays(phase, depth, valid, amplitude, confidence)
        if not np.any(valid):
            raise ValueError(f"样本 {record.sample_id} 没有有效像素")
        if self.cache_size > 0:
            self._cache[index] = arrays
            self._cache_order.append(index)
            while len(self._cache_order) > self.cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return arrays

    def valid_flat_indices(self, index: int) -> np.ndarray:
        """缓存每个平面的有效索引，避免每个 batch 重复扫描 H×W mask。"""
        cached = self._valid_flat_cache.get(index)
        if cached is None:
            cached = np.flatnonzero(self.load(index).valid).astype(np.int32)
            self._valid_flat_cache[index] = cached
        return cached

    def compute_normalization(
        self,
        split: str = "train",
        max_pixels_per_plane: Optional[int] = 500_000,
        seed: int = 0,
    ) -> NormalizationStats:
        phase_stats, depth_stats, amplitude_stats = (
            RunningStats(),
            RunningStats(),
            RunningStats(),
        )
        rng = np.random.default_rng(seed)
        selected = self.indices(split)
        if not selected:
            raise ValueError(f"数据集没有 split={split!r}")
        has_amplitude = False
        for index in selected:
            arrays = self.load(index)
            flat = np.flatnonzero(arrays.valid)
            if max_pixels_per_plane and flat.size > max_pixels_per_plane:
                flat = rng.choice(flat, size=max_pixels_per_plane, replace=False)
            phase_stats.update(arrays.phase.ravel()[flat])
            depth_stats.update(arrays.depth_mm.ravel()[flat])
            if arrays.amplitude is not None:
                amplitude_stats.update(arrays.amplitude.ravel()[flat])
                has_amplitude = True
        phase_mean, phase_std, phase_min, phase_max = phase_stats.finish()
        _, _, depth_min, depth_max = depth_stats.finish()
        if has_amplitude:
            amplitude_mean, amplitude_std, _, _ = amplitude_stats.finish()
        else:
            amplitude_mean, amplitude_std = 0.0, 1.0
        stats = NormalizationStats(
            phase_mean,
            phase_std,
            phase_min,
            phase_max,
            depth_min,
            depth_max,
            amplitude_mean,
            amplitude_std,
        )
        validate_stats(stats)
        return stats

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """先均匀选平面，再在平面内均匀选有效像素。"""
        selected = self.indices(split)
        if not selected:
            raise ValueError(f"数据集没有 split={split!r}")
        plane_choices = rng.choice(selected, size=batch_size, replace=True)
        output: dict[str, list[np.ndarray]] = {
            "u": [],
            "v": [],
            "phase": [],
            "depth_mm": [],
            "confidence": [],
            "amplitude": [],
            "record_index": [],
        }
        for record_index in np.unique(plane_choices):
            count = int(np.sum(plane_choices == record_index))
            arrays = self.load(int(record_index))
            valid_flat = self.valid_flat_indices(int(record_index))
            # 对百万像素 mask 使用 replace=False 会每次构造大型排列。
            # 有放回均匀采样仍是无偏的，且本 batch 内的重复概率很低。
            chosen = rng.choice(valid_flat, size=count, replace=True)
            v, u = np.unravel_index(chosen, (self.height, self.width))
            output["u"].append(u.astype(np.int64))
            output["v"].append(v.astype(np.int64))
            output["phase"].append(arrays.phase.ravel()[chosen])
            output["depth_mm"].append(arrays.depth_mm.ravel()[chosen])
            output["confidence"].append(arrays.confidence.ravel()[chosen])
            amplitude = (
                np.zeros(count, dtype=np.float32)
                if arrays.amplitude is None
                else arrays.amplitude.ravel()[chosen]
            )
            output["amplitude"].append(amplitude)
            output["record_index"].append(
                np.full(count, int(record_index), dtype=np.int64)
            )
        merged = {key: np.concatenate(value) for key, value in output.items()}
        permutation = rng.permutation(batch_size)
        return {key: value[permutation] for key, value in merged.items()}


def build_features(
    u: np.ndarray,
    v: np.ndarray,
    phase: np.ndarray,
    height: int,
    width: int,
    stats: NormalizationStats,
    spec: FeatureSpec,
    amplitude: Optional[np.ndarray] = None,
) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    u_n = np.zeros_like(u) if width == 1 else 2.0 * u / (width - 1) - 1.0
    v_n = np.zeros_like(v) if height == 1 else 2.0 * v / (height - 1) - 1.0
    columns: list[np.ndarray] = [u_n, v_n, stats.normalize_phase(phase)]
    if spec.use_periodic_phase:
        columns.extend((np.sin(phase), np.cos(phase)))
    if spec.use_amplitude:
        if amplitude is None:
            raise ValueError("配置要求 amplitude，但数据未提供")
        columns.append(stats.normalize_amplitude(amplitude))
    return np.stack(columns, axis=-1).astype(np.float32)


def full_image_features(
    phase: np.ndarray,
    stats: NormalizationStats,
    spec: FeatureSpec,
    amplitude: Optional[np.ndarray] = None,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim != 2:
        raise ValueError("phase 必须为 (H,W)")
    height, width = phase.shape
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return build_features(
        u.ravel(),
        v.ravel(),
        phase.ravel(),
        height,
        width,
        stats,
        spec,
        None if amplitude is None else np.asarray(amplitude).ravel(),
    )


def validate_absolute_phase(dataset: CalibrationDataset) -> None:
    accepted = {"absolute", "absolute_unwrapped", "unwrapped"}
    if dataset.phase_type.lower() not in accepted:
        raise ValueError(
            f"phase_type={dataset.phase_type!r}。(u,v,phase)->Z 要求绝对展开相位；"
            "若是包裹相位，必须额外提供条纹级次或多频消歧信息。"
        )
