"""目标点嵌入层：BEV 栅格 xyz + 目标点相对向量 → 初始 BEV 查询网格。公开 API 重导出入口。

模块: model/target_point_embedding/__init__.py
依赖: model.target_point_embedding.target_point_embedding
读取配置: —
对外接口:
    - TargetPointEmbedding(...) -> nn.Module   # 初始 BEV 查询网格生成
说明: 跨模块统一 `from model.target_point_embedding import ...`；实现见 target_point_embedding.py，入参校验见 checks/。
"""

from model.target_point_embedding.target_point_embedding import TargetPointEmbedding

__all__ = ["TargetPointEmbedding"]
