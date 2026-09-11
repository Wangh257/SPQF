from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from phase_field.infer import run_inference
from phase_field.batch_test import batch_test
from phase_field.evaluate import evaluate
from phase_field.convert_zn_to_xyz import convert
from phase_field.predictor import DirectMLPPredictor
from phase_field.prepare_manifest import create_manifest
from phase_field.train import train


def _make_dataset(root: Path) -> tuple[Path, Path]:
    height, width = 8, 10
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    depths = [420.0, 460.0, 500.0, 440.0, 480.0]
    splits = ["train", "train", "train", "val", "test"]
    samples = []
    for index, (depth, split) in enumerate(zip(depths, splits), start=1):
        # 同一 Z 平面上的相位仍随像素位置变化。
        phase = 0.02 * depth + 0.01 * u - 0.015 * v
        phase_path = root / f"phase_{index}.npy"
        np.save(phase_path, phase.astype(np.float32))
        samples.append(
            {
                "id": f"plane_{index}",
                "plane_id": f"plane_{index}",
                "split": split,
                "phase": str(phase_path),
                "z_mm": depth,
            }
        )
    manifest = {
        "image_size": [height, width],
        "phase_type": "absolute_unwrapped",
        "depth_unit": "mm",
        "camera": {
            "intrinsics": [[100.0, 0.0, 4.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]],
            "distortion_model": "none",
        },
        "samples": samples,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, root / "phase_5.npy"


def test_training_checkpoint_and_inference_smoke(tmp_path: Path) -> None:
    import yaml

    manifest_path, test_phase = _make_dataset(tmp_path)
    config = {
        "seed": 1,
        "dataset": {"manifest": str(manifest_path), "cache_planes": 2},
        "features": {"use_periodic_phase": True, "use_amplitude": False},
        "model": {
            "hidden_dim": 16,
            "hidden_layers": 3,
            "activation": "silu",
            "skip_layer": 2,
        },
        "training": {
            "device": "cpu",
            "epochs": 2,
            "steps_per_epoch": 4,
            "batch_size": 64,
            "learning_rate": 0.005,
            "amp": False,
            "val_pixels_per_plane": 80,
            "inference_chunk_size": 128,
        },
        "output_dir": str(tmp_path / "run"),
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = train(config_path)
    assert checkpoint.exists()
    predictor = DirectMLPPredictor(checkpoint, "cpu")
    phase = np.load(test_phase)
    zn, depth, valid = predictor.predict(phase)
    assert zn.shape == phase.shape
    assert depth.shape == phase.shape
    assert valid.all()
    output = tmp_path / "inference"
    metadata = run_inference(
        checkpoint, test_phase, output, device="cpu", chunk_size=128, save_ply=False
    )
    assert metadata["valid_pixels"] == phase.size
    for name in ["predicted_zn.npy", "XC.npy", "YC.npy", "ZC.npy", "point_map.npy"]:
        assert (output / name).exists()
    zc, point_map = convert(
        output / "predicted_zn.npy", checkpoint, tmp_path / "converted"
    )
    assert zc.shape == phase.shape
    assert point_map.shape == (*phase.shape, 3)
    metrics = evaluate(
        checkpoint,
        manifest_path,
        "test",
        tmp_path / "evaluation",
        chunk_size=128,
        device="cpu",
    )
    assert metrics["summary"]["count"] == phase.size
    batch_metrics = batch_test(
        checkpoint,
        manifest_path,
        tmp_path,
        tmp_path / "batch_test",
        split="test",
        phase_pattern="phase_{index}.npy",
        device="cpu",
        chunk_size=128,
    )
    assert batch_metrics["sample_indices"] == [5]
    assert batch_metrics["summary"]["count"] == phase.size

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    evaluation_mask = np.ones(phase.shape, dtype=bool)
    evaluation_mask[0, 0] = False
    np.save(mask_dir / "mask_5.npy", evaluation_mask)
    masked_metrics = batch_test(
        checkpoint,
        manifest_path,
        tmp_path,
        tmp_path / "batch_test_masked",
        split="test",
        phase_pattern="phase_{index}.npy",
        mask_dir=mask_dir,
        mask_pattern="mask_{index}.npy",
        device="cpu",
        chunk_size=128,
    )
    assert masked_metrics["summary"]["count"] == phase.size - 1
    # 评价 mask 只影响指标，不删除完整重建结果。
    masked_z = np.load(tmp_path / "batch_test_masked" / "PSP_5" / "ZC.npy")
    assert np.isfinite(masked_z[0, 0])


def test_prepare_manifest_from_numeric_camera_parameters(tmp_path: Path) -> None:
    phase_dir = tmp_path / "phases"
    phase_dir.mkdir()
    for index in range(3):
        np.save(phase_dir / f"phase_{index + 1}.npy", np.ones((4, 5), np.float32))
    rotations = np.repeat(np.eye(3)[:, :, None], 3, axis=2)
    translations = np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 450.0], [0.0, 0.0, 500.0]])
    camera_path = tmp_path / "camera.npz"
    np.savez(
        camera_path,
        K=np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]),
        R=rotations,
        T=translations,
        WorldPoints=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]),
    )
    manifest_path = create_manifest(
        camera_path,
        phase_dir,
        tmp_path / "prepared.json",
        phase_glob="*.npy",
        distortion_model="none",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["image_size"] == [4, 5]
    assert {sample["split"] for sample in manifest["samples"]} == {
        "train",
        "val",
        "test",
    }
    depths = sorted(
        round(-sample["plane"][3] / sample["plane"][2])
        for sample in manifest["samples"]
    )
    assert depths == [400, 450, 500]
