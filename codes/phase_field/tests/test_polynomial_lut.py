import numpy as np

from phase_field.cpc_phase import cpc_fit_phase
from phase_field.inverse_polynomial_lut import fit_inverse_polynomial_lut
from phase_field.polynomial_lut import fit_polynomial_lut
from phase_field.polynomial_model import load_polynomial_model


def test_masked_polynomial_fit_ignores_rejected_observation() -> None:
    phase = np.arange(5, dtype=np.float32)[:, None, None]
    depth = 2.0 + 3.0 * phase + 0.5 * phase**2
    depth[2, 0, 0] = 10_000.0
    valid = np.ones_like(phase, dtype=bool)
    valid[2, 0, 0] = False

    lut = fit_polynomial_lut(
        phase,
        depth,
        degree=2,
        ridge=0.0,
        valid_stack=valid,
    )
    prediction = lut.predict_z(np.array([[1.5]], dtype=np.float32))

    np.testing.assert_allclose(prediction, [[7.625]], atol=1e-5)
    assert lut.metadata["uses_mask_for_fit"] is True


def test_cpc_phase_uses_only_mask_valid_pixels() -> None:
    v, u = np.indices((9, 11), dtype=np.float32)
    truth = 4.0 + 0.2 * u - 0.3 * v + 0.01 * u * v + 0.02 * u**2
    phase = truth.copy()
    mask = np.ones_like(phase, dtype=bool)
    mask[3:6, 4:8] = False
    phase[~mask] = 999.0

    fitted = cpc_fit_phase(phase, mask, order=2, ridge=0.0, chunk_pixels=17)

    np.testing.assert_allclose(fitted, truth, atol=1e-4)


def test_inverse_cubic_polynomial_uses_cubic_as_denominator(tmp_path) -> None:
    values = np.linspace(-3.0, 3.0, 8, dtype=np.float32)
    phase = np.broadcast_to(values[:, None, None], (8, 2, 3)).copy()
    phase_center = float(phase.mean(dtype=np.float64))
    phase_scale = float(phase.std(dtype=np.float64))
    q = (phase - phase_center) / phase_scale
    denominator = 0.002 + 0.0002 * q + 0.00003 * q**2 - 0.00001 * q**3
    depth = 1.0 / denominator

    lut = fit_inverse_polynomial_lut(phase, depth, degree=3, ridge=0.0)
    query = np.full((2, 3), 0.37, dtype=np.float32)
    query_q = (query - lut.phase_center) / lut.phase_scale
    expected = 1.0 / (
        0.002
        + 0.0002 * query_q
        + 0.00003 * query_q**2
        - 0.00001 * query_q**3
    )
    np.testing.assert_allclose(lut.predict_z(query), expected, rtol=1e-5, atol=1e-3)

    model_path = lut.save(tmp_path / "inverse_degree3.npz")
    loaded = load_polynomial_model(model_path)
    assert loaded.metadata["method"] == "per_pixel_inverse_polynomial_lut"
    np.testing.assert_allclose(loaded.predict_z(query), expected, rtol=1e-5, atol=1e-3)
