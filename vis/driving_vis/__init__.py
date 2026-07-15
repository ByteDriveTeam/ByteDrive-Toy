"""驾驶模型可视化子模块包标识：渲染 RGB/Seg/Depth 与 GT/预测三场、道路线图及多模态轨迹对照。

模块: vis/driving_vis/__init__.py
依赖: —（CLI 入口见 run.py，渲染见 render/）
读取配置: —
对外接口: —（本包经 run.py 命令行使用；渲染 API 见 vis.driving_vis.render）
说明: 与 vis/pred_vis（感知预测可视化）平行，专用于驾驶模型；复用 pred_vis 的 RGB/Seg/Depth 着色与
      data.driving_targets 的 BEV 几何，保证像素/口径一致。
"""
