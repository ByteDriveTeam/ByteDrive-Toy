"""DINOv3 ViT-S+ 视觉骨干：全程冻结 + eval，输出选定层完整 Token 序列。公开 API 重导出入口。

模块: model/dinov3_backbone/__init__.py
依赖: model.dinov3_backbone.dinov3_backbone
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - DinoV3Backbone(cfg) -> nn.Module   # 冻结 DINOv3 骨干，输出 [N,L,1+R+P,hidden]
说明: 跨模块统一 `from model.dinov3_backbone import ...`；实现见 dinov3_backbone.py，入参校验见 checks/。
"""

from model.dinov3_backbone.dinov3_backbone import DinoV3Backbone

__all__ = ["DinoV3Backbone"]
