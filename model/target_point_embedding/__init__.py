"""目标点嵌入层：ego 目标点经栅格向量场⊕栅格坐标（Symlog 拼接）与卷积编码为目标导航特征图。公开 API 重导出入口。

模块: model/target_point_embedding/__init__.py
依赖: model.target_point_embedding.target_point_embedding
读取配置: —
对外接口:
    - TargetPointEmbedding(...) -> nn.Module   # 目标点嵌入层
说明: 跨模块统一 `from model.target_point_embedding import ...`；实现见 target_point_embedding.py，入参校验见 checks/。
"""

from model.target_point_embedding.target_point_embedding import TargetPointEmbedding

__all__ = ["TargetPointEmbedding"]
