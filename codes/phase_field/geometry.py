from __future__ import annotations

from typing import Iterable

import numpy as np


def _validate_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"相机内参应为 (3,3)，实际为 {k.shape}")
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("相机内参无效")
    return k


def normalized_camera_rays(
    height: int,
    width: int,
    intrinsics: np.ndarray,
    radial_distortion: Iterable[float] = (),
    tangential_distortion: Iterable[float] = (),
    distortion_model: str = "none",
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 ``[xn, yn, 1]`` 形式的相机射线，射线不做单位化。

    不单位化是因为本项目的深度定义为相机 Z 轴深度，
    ``[XC, YC, ZC] = [xn*ZC, yn*ZC, ZC]``。
    """
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正数")
    k = _validate_intrinsics(intrinsics)
    fx, fy, skew, cx, cy = k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2]
    v, u = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    yd = (v - cy) / fy
    xd = (u - cx - skew * yd) / fx

    radial = list(radial_distortion)
    radial = (radial + [0.0, 0.0, 0.0])[:3]
    tangential = list(tangential_distortion)
    tangential = (tangential + [0.0, 0.0])[:2]
    k1, k2, k3 = (float(value) for value in radial)
    p1, p2 = (float(value) for value in tangential)

    if distortion_model == "none":
        return xd.astype(np.float32), yd.astype(np.float32)

    if distortion_model == "matlab_legacy":
        # 严格复现 get_lut_cali.m 的一次近似去径向畸变，便于对照旧 LUT。
        r2 = xd * xd + yd * yd
        correction = k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return (
            (xd - xd * correction).astype(np.float32),
            (yd - yd * correction).astype(np.float32),
        )

    if distortion_model != "brown_conrady":
        raise ValueError("distortion_model 必须是 none、matlab_legacy 或 brown_conrady")

    # 反解 Brown-Conrady 畸变。这与 OpenCV 常见的 k1,k2,p1,p2,k3 模型一致。
    x = xd.copy()
    y = yd.copy()
    for _ in range(max(iterations, 1)):
        r2 = x * x + y * y
        radial_factor = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        safe_radial = np.where(np.abs(radial_factor) > 1e-12, radial_factor, 1.0)
        x = (xd - delta_x) / safe_radial
        y = (yd - delta_y) / safe_radial
    return x.astype(np.float32), y.astype(np.float32)


def intersect_camera_rays_with_plane(
    xn: np.ndarray,
    yn: np.ndarray,
    plane: Iterable[float],
) -> np.ndarray:
    """计算射线 ``Z*[xn,yn,1]`` 与 ``AX+BY+CZ+D=0`` 的交点 Z。"""
    coefficients = np.asarray(list(plane), dtype=np.float64)
    if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("平面系数应为有限的 [A,B,C,D]")
    a, b, c, d = coefficients
    denominator = a * xn + b * yn + c
    z = np.full(np.broadcast_shapes(xn.shape, yn.shape), np.nan, dtype=np.float32)
    valid = np.abs(denominator) > 1e-12
    z[valid] = (-d / denominator[valid]).astype(np.float32)
    z[z <= 0] = np.nan
    return z


def depth_to_point_map(
    depth_z_mm: np.ndarray,
    intrinsics: np.ndarray,
    radial_distortion: Iterable[float] = (),
    tangential_distortion: Iterable[float] = (),
    distortion_model: str = "none",
) -> np.ndarray:
    """将相机轴向深度 ``ZC(H,W)`` 转换为 ``[XC,YC,ZC](H,W,3)``。"""
    depth = np.asarray(depth_z_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_z_mm 应为 (H,W)，实际为 {depth.shape}")
    height, width = depth.shape
    xn, yn = normalized_camera_rays(
        height,
        width,
        intrinsics,
        radial_distortion,
        tangential_distortion,
        distortion_model,
    )
    points = np.stack((xn * depth, yn * depth, depth), axis=-1)
    invalid = ~np.isfinite(depth) | (depth <= 0)
    points[invalid] = np.nan
    return points.astype(np.float32)


def normalized_depth_to_camera_points(
    zn: np.ndarray,
    depth_min_mm: float,
    depth_max_mm: float,
    intrinsics: np.ndarray,
    radial_distortion: Iterable[float] = (),
    tangential_distortion: Iterable[float] = (),
    distortion_model: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """反归一化 ``zn`` 并返回 ``ZC`` 和 ``[XC,YC,ZC]``。"""
    if depth_max_mm <= depth_min_mm:
        raise ValueError("depth_max_mm 必须大于 depth_min_mm")
    depth = (
        np.asarray(zn, dtype=np.float32) * (depth_max_mm - depth_min_mm) + depth_min_mm
    )
    points = depth_to_point_map(
        depth,
        intrinsics,
        radial_distortion,
        tangential_distortion,
        distortion_model,
    )
    return depth, points


def fit_plane(points_xyz: np.ndarray) -> np.ndarray:
    """通过 SVD 拟合平面，返回归一化后的 ``[A,B,C,D]``。"""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 3:
        raise ValueError("拟合平面至少需要 3 个有效点")
    design = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    plane = vh[-1]
    norm = np.linalg.norm(plane[:3])
    if norm <= 1e-12:
        raise ValueError("平面拟合退化")
    return (plane / norm).astype(np.float64)
