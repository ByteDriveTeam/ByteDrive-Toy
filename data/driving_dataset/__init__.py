"""驾驶模型双帧数据集：生成帧间刚性变换、独立道路线图与驾驶多任务监督。公开 API 重导出入口。

模块: data/driving_dataset/__init__.py
依赖: data.driving_dataset.driving_dataset
读取配置: —（转由 DrivingDataset 读取 config.data.driving / data.dataset / model.driving）
对外接口:
    - DrivingDataset(cfg) -> torch.utils.data.Dataset
说明: 跨模块统一 `from data.driving_dataset import DrivingDataset`；实现见 driving_dataset.py，校验见 checks/。
"""

from data.driving_dataset.driving_dataset import DrivingDataset

__all__ = ["DrivingDataset"]
