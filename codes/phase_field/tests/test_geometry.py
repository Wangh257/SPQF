import numpy as np

from phase_field.geometry import (
    depth_to_point_map,
    intersect_camera_rays_with_plane,
    normalized_camera_rays,
    normalized_depth_to_camera_points,
)


def test_depth_to_point_map_uses_camera_z_not_ray_range() -> None:
    intrinsics = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    depth = np.full((3, 3), 10.0, dtype=np.float32)
    points = depth_to_point_map(depth, intrinsics)
    np.testing.assert_allclose(points[1, 1], [0.0, 0.0, 10.0])
    np.testing.assert_allclose(points[0, 0], [-5.0, -5.0, 10.0])


def test_zn_denormalization_and_camera_coordinates() -> None:
    intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    zn = np.full((3, 3), 0.25, dtype=np.float32)
    depth, points = normalized_depth_to_camera_points(zn, 400.0, 600.0, intrinsics)
    np.testing.assert_allclose(depth, 450.0)
    np.testing.assert_allclose(points[..., 2], 450.0)
    np.testing.assert_allclose(points[1, 1], [0.0, 0.0, 450.0])


def test_ray_plane_intersection() -> None:
    intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    xn, yn = normalized_camera_rays(3, 3, intrinsics)
    depth = intersect_camera_rays_with_plane(xn, yn, [0.0, 0.0, 1.0, -500.0])
    np.testing.assert_allclose(depth, 500.0)
