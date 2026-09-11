"""相位到深度的神经标定场。

本包默认的 ``zn`` 定义为相机轴向深度 ``ZC`` 在训练深度
范围内的归一化值，而不是射线的欧氏距离。
"""

from .geometry import depth_to_point_map, normalized_depth_to_camera_points
from .normalization import NormalizationStats

__all__ = [
    "NormalizationStats",
    "depth_to_point_map",
    "normalized_depth_to_camera_points",
]

__version__ = "0.1.0"
