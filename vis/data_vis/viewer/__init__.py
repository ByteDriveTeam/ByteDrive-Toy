"""OpenCV 交互窗口：帧滑条 + 键盘播放/单步/图层切换/截图。公开 API 重导出入口。

模块: vis/data_vis/viewer/__init__.py
依赖: vis.data_vis.viewer.viewer
读取配置: —（实现文件经传入的 cfg 读取）
对外接口:
    - Viewer(reader, vcfg) -> 交互窗口   # OpenCV 帧浏览窗口
说明: 跨模块统一 `from vis.data_vis.viewer import ...`；实现见 viewer.py，入参校验见 checks/。
"""

from vis.data_vis.viewer.viewer import Viewer

__all__ = ["Viewer"]
