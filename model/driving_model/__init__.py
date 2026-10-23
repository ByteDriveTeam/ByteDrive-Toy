"""四帧块因果驾驶模型与局部二维加速度词表规划的公开 API。

模块: model/driving_model/__init__.py
依赖: model.driving_model.unified_driving_model
读取配置: —（转由 DrivingModel 读取 config.model.unified_driving 等键）
对外接口:
    - DrivingModel(cfg) -> nn.Module
    - nearest_planar_action_words(gt_xy, initial_velocity, actions, dt_s, weight_floor, decay_s) -> Tensor
说明: 跨模块统一 `from model.driving_model import DrivingModel`；实现见 unified_driving_model.py。
"""

from model.driving_model.unified_driving_model import UnifiedDrivingModel, nearest_planar_action_words

DrivingModel = UnifiedDrivingModel

__all__ = ["DrivingModel", "UnifiedDrivingModel", "nearest_planar_action_words"]
