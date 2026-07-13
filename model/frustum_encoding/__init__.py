"""深度 frustum 位置编码：每 patch 视锥候选 3D 坐标 → 逐 patch 几何特征。公开 API 重导出入口。

模块: model/frustum_encoding/__init__.py
依赖: model.frustum_encoding.frustum_encoding
读取配置: —
对外接口:
    - FrustumEncoding(out_dim, patch_size, depth_min_m, depth_max_m, step_near_m, step_far_m,
                      coord_symlog_scale, mlp_hidden) -> nn.Module
说明: 跨模块统一 `from model.frustum_encoding import FrustumEncoding`；实现见 frustum_encoding.py，校验见 checks/。
"""

from model.frustum_encoding.frustum_encoding import FrustumEncoding

__all__ = ["FrustumEncoding"]
