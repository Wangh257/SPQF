from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

from phase_field.data import CalibrationDataset
from phase_field.extract_raw_phase import extract_raw_phases
from phase_field.generate_phase_variants import (
    generate_phase_variants,
    transform_phase,
)
from phase_field.phase_sensitivity import compute_phase_sensitivity
from phase_field.polynomial_lut import PolynomialLUT


def test_raw_phase_extraction_writes_cache_metadata(tmp_path: Path) -> None:
    images = tmp_path / "images"
    output = tmp_path / "raw_phase"
    images.mkdir()
    for index in range(6):
        array = np.full((3, 4), 20 + index * 10, dtype=np.uint8)
        Image.fromarray(array).save(images / f"image_{index + 1}.bmp")
    paths = extract_raw_phases(
        images,
        output,
        num_frequencies=2,
        num_steps=3,
        frequency_ratios=(2.0,),
    )
    metadata = json.loads(
        (output / "phase_extraction_metadata.json").read_text(encoding="utf-8")
    )
    assert len(paths) == 1
    assert metadata["images_per_pose"] == 6
    assert metadata["provenance_status"] == "all_outputs_generated_from_listed_sources"


def test_phase_variant_region_and_batch_output(tmp_path: Path) -> None:
    phase = np.arange(20, dtype=np.float32).reshape(4, 5)
    phase[0, 0] = np.nan
    changed = transform_phase(
        phase,
        mode="offset",
        rng=np.random.default_rng(1),
        region=(1, 1, 3, 2),
        value=0.25,
    )
    expected = phase.copy()
    expected[1:3, 1:4] += 0.25
    np.testing.assert_allclose(changed, expected, equal_nan=True)

    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "variant"
    input_dir.mkdir()
    savemat(input_dir / "PSP_1.mat", {"phase": phase})
    summary = generate_phase_variants(
        input_dir,
        output_dir,
        mode="offset",
        value=-0.5,
    )
    generated = np.asarray(loadmat(output_dir / "PSP_1.mat")["phase"])
    np.testing.assert_allclose(generated, phase - 0.5, equal_nan=True)
    assert summary["file_count"] == 1
    assert (output_dir / "phase_variant_metadata.json").is_file()


def test_random_fringe_jump_blocks_are_seeded_and_stay_in_region() -> None:
    phase = np.zeros((60, 80), dtype=np.float32)
    kwargs = {
        "phase": phase,
        "mode": "fringe_jump_blocks",
        "region": (20, 10, 30, 35),
        "orders": 1,
        "block_count": 8,
        "block_sizes": (5, 10, 20),
    }
    first = transform_phase(rng=np.random.default_rng(42), **kwargs)
    second = transform_phase(rng=np.random.default_rng(42), **kwargs)
    np.testing.assert_array_equal(first, second)

    outside = np.ones(phase.shape, dtype=bool)
    outside[10:45, 20:50] = False
    assert np.all(first[outside] == 0)
    assert np.count_nonzero(first) > 0
    jump_orders = first[first != 0] / np.float32(2.0 * np.pi)
    np.testing.assert_allclose(jump_orders, np.round(jump_orders), atol=1e-6)


def test_random_fringe_jump_pixels_are_unique_seeded_and_in_region() -> None:
    phase = np.zeros((40, 60), dtype=np.float32)
    kwargs = {
        "phase": phase,
        "mode": "fringe_jump_pixels",
        "region": (10, 8, 30, 20),
        "orders": 1,
        "pixel_count": 75,
    }
    first = transform_phase(rng=np.random.default_rng(7), **kwargs)
    second = transform_phase(rng=np.random.default_rng(7), **kwargs)
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) == 75

    outside = np.ones(phase.shape, dtype=bool)
    outside[8:28, 10:40] = False
    assert np.all(first[outside] == 0)
    jump_orders = first[first != 0] / np.float32(2.0 * np.pi)
    np.testing.assert_allclose(np.abs(jump_orders), 1.0, atol=1e-6)


def test_dataset_can_replace_manifest_phase_directory(tmp_path: Path) -> None:
    variants = tmp_path / "variants"
    variants.mkdir()
    savemat(variants / "PSP_1.mat", {"phase": np.full((2, 3), 11, np.float32)})
    savemat(variants / "PSP_2.mat", {"phase": np.full((2, 3), 22, np.float32)})
    manifest = {
        "image_size": [2, 3],
        "phase_type": "absolute_unwrapped",
        "depth_unit": "mm",
        "camera": {
            "intrinsics": [[100, 0, 1], [0, 100, 0.5], [0, 0, 1]],
            "distortion_model": "none",
        },
        "samples": [
            {
                "id": "one",
                "plane_id": "one",
                "split": "train",
                "phase": "missing_original_1.mat",
                "z_mm": 400,
            },
            {
                "id": "two",
                "plane_id": "two",
                "split": "test",
                "phase": "missing_original_2.mat",
                "z_mm": 450,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = CalibrationDataset(manifest_path, phase_dir=variants)
    assert np.all(dataset.load(0).phase == 11)
    assert np.all(dataset.load(1).phase == 22)


def test_polynomial_phase_sensitivity_is_mm_per_rad(tmp_path: Path) -> None:
    # Z = 5 + 3*q, q=(phase-0)/2, therefore dZ/dphase=1.5 mm/rad.
    coefficients = np.zeros((3, 4, 2), dtype=np.float32)
    coefficients[..., 0] = 5.0
    coefficients[..., 1] = 3.0
    lut = PolynomialLUT(coefficients, 0.0, 2.0, {"camera": {}})
    model_path = lut.save(tmp_path / "linear.npz")
    phase_path = tmp_path / "phase.npy"
    np.save(phase_path, np.full((3, 4), 0.7, dtype=np.float32))

    summary = compute_phase_sensitivity(
        model_path,
        phase_path,
        tmp_path / "sensitivity",
        epsilon=0.01,
    )
    sensitivity = np.load(
        tmp_path / "sensitivity" / "sensitivity_mm_per_rad.npy"
    )
    np.testing.assert_allclose(sensitivity, 1.5, rtol=1e-4, atol=1e-4)
    assert abs(summary["absolute_mean"] - 1.5) < 1e-4
    assert (tmp_path / "sensitivity" / "phase_sensitivity.mat").is_file()
