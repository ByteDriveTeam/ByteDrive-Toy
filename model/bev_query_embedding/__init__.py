"""BEV 查询几何嵌入：默认或刚性变换后的 BEV xyz 几何生成查询网格。公开 API 重导出入口。

模块: model/bev_query_embedding/__init__.py
依赖: model.bev_query_embedding.bev_query_embedding
读取配置: —
对外接口:
    - BevQueryEmbedding(...) -> nn.Module   # 纯几何初始 BEV 查询网格生成
说明: 跨模块统一 `from model.bev_query_embedding import BevQueryEmbedding`；实现与校验分别见同模块文件和 checks/。
"""

from model.bev_query_embedding.bev_query_embedding import BevQueryEmbedding

__all__ = ["BevQueryEmbedding"]
