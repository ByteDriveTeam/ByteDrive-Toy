"""验证时间对齐观测与滚动轨迹诊断可稳定写入 JSONL。

模块: clone_loop/logger/tests/test_logger.py
依赖: json, pathlib, tempfile, unittest, numpy, clone_loop.logger
读取配置: —
对外接口: —
说明: 临时目录显式创建在项目 clone_loop 目录内，并由上下文管理器回收。
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clone_loop.logger import RunLogger


class RunLoggerTests(unittest.TestCase):
    """覆盖新版逐步日志的字段对齐与 NumPy 数组转换。"""

    def test_time_aligned_step_is_json_serializable(self):
        """输入/下一观测和 active reference 应落入同一条可解析 step。"""
        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(
                prefix=".logger_test_", dir=str(project_root / "clone_loop")) as temp:
            logger = RunLogger(Path(temp))
            route = {"start": [0.0] * 6, "end": [1.0] * 6}
            logger.start_episode(0, route, 1)
            input_observation = {"step": 0, "route_deviation_m": 0.1}
            next_observation = {"step": 1, "route_deviation_m": 0.2}
            command = {
                "throttle": 0.1, "steer": 0.0, "brake": 0.0,
                "target_speed_mps": 2.0,
            }
            decision = {
                "mode": 0,
                "mode_scores": np.zeros(2, dtype=np.float32),
                "confidence": np.ones(2, dtype=np.float32),
                "behavior_probabilities": np.zeros(2, dtype=np.float32),
                "history_valid": True,
                "trajectory": np.zeros((3, 2), dtype=np.float32),
            }
            execution = {
                "reference_trajectory": np.ones((3, 2), dtype=np.float32),
                "tracking_error_m": 0.2,
                "prediction_shift_m": 0.3,
                "plan_reseeded": False,
                "reseed_reason": None,
                "committed_waypoints": 3,
                "fresh_target_speed_mps": 2.0,
                "target_speed_mps": 2.0,
                "lookahead_m": 3.0,
                "path_curvature_inv_m": 0.1,
                "stop_requested": False,
                "stop_probability": 0.2,
            }
            logger.write_step(
                input_observation, next_observation, command, decision, execution)
            logger.close()

            path = next(logger.run_dir.glob("episode_*.jsonl"))
            payloads = [json.loads(line) for line in path.read_text(
                encoding="utf-8").splitlines()]
            step = payloads[1]
            self.assertEqual(step["input_observation"]["step"], 0)
            self.assertEqual(step["next_observation"]["step"], 1)
            self.assertEqual(step["execution"]["reference_trajectory"], [[1.0, 1.0]] * 3)


if __name__ == "__main__":
    unittest.main()
