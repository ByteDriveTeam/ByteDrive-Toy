"""基于完整10Hz GT世界轨迹计算模型候选与实际执行原始代价。公开 API 重导出入口。

模块: collector/costs/__init__.py
依赖: collector.costs.costs
读取配置: —（参数由调用方传入）
对外接口:
    - COST_TERMS
    - evaluate_drive_costs(world_states, model_steps, route_geometry, cfg_cost, cfg_model) -> None
说明: 函数就地给 model_steps 添加逐项原始代价数组，不生成裁剪、归一化或加权总分。
"""

from collector.costs.costs import COST_TERMS, evaluate_drive_costs

__all__ = ["COST_TERMS", "evaluate_drive_costs"]
