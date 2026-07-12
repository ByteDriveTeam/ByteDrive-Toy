"""感知模型时序开窗数据集：把落盘场景切成 5 帧窗口，产出归一化 RGB 与四任务监督目标。公开 API 重导出入口。

模块: data/perception_dataset/__init__.py
依赖: data.perception_dataset.perception_dataset
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - PerceptionDataset(cfg) -> Dataset   # 时序开窗数据集
说明: 跨模块统一 `from data.perception_dataset import ...`；实现见 perception_dataset.py，入参校验见 checks/。
"""

from data.perception_dataset.perception_dataset import PerceptionDataset

__all__ = ["PerceptionDataset"]
