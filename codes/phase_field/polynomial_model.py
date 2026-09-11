from __future__ import annotations

from pathlib import Path

import numpy as np

from .inverse_polynomial_lut import InversePolynomialLUT
from .polynomial_lut import PolynomialLUT


PolynomialDepthModel = PolynomialLUT | InversePolynomialLUT


def load_polynomial_model(path: str | Path) -> PolynomialDepthModel:
    """自动识别直接多项式或逆多项式 ``.npz``。"""
    resolved = Path(path).resolve()
    with np.load(resolved, allow_pickle=False) as source:
        keys = set(source.files)
    if "denominator_coefficients" in keys:
        return InversePolynomialLUT.load(resolved)
    if "coefficients_z" in keys:
        return PolynomialLUT.load(resolved)
    raise ValueError(f"{resolved} 不是支持的多项式模型")
