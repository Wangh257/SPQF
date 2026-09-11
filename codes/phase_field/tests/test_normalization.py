import numpy as np

from phase_field.normalization import NormalizationStats


def test_depth_normalization_round_trip() -> None:
    stats = NormalizationStats(10.0, 2.0, 5.0, 15.0, 400.0, 600.0)
    depth = np.array([400.0, 450.0, 600.0], dtype=np.float32)
    restored = stats.denormalize_depth(stats.normalize_depth(depth))
    np.testing.assert_allclose(restored, depth)
