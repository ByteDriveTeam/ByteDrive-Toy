"""把时间对齐的闭环观测、滚动计划、控制与选择结果写为 JSONL，并生成运行汇总。

模块: clone_loop/logger/logger.py
依赖: datetime, json, pathlib
读取配置: clone_loop.output.root（由构造参数传入并由编排器解析）
对外接口:
    - RunLogger(output_root)
        .start_episode(index, route, seed) -> None
        .write_step(input_observation, next_observation, command, decision, execution) -> None
        .finish_episode(observation, artifacts=None) -> dict
        .finish_run() -> dict
        .close() -> None
"""

from datetime import datetime
import json
from pathlib import Path


__all__ = ["RunLogger"]


class RunLogger:
    """一次闭环运行的落盘日志管理器。"""

    def __init__(self, output_root):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(output_root) / ("run_" + stamp)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._summaries = []
        self._stream = None
        self._episode = None

    def start_episode(self, index, route, seed):
        """打开一个 episode JSONL，并写入首行元数据。"""
        path = self.run_dir / "episode_{:04d}.jsonl".format(index)
        self._stream = open(path, "w", encoding="utf-8")
        self._episode = {"episode": index, "route": route, "seed": seed}
        self._max_deviation = 0.0
        self._write({"type": "episode", **self._episode})

    def write_step(self, input_observation, next_observation, command,
                   decision, execution):
        """记录时间对齐的输入/下一观测、模型决策、滚动计划与实际控制。"""
        self._max_deviation = max(
            self._max_deviation, float(next_observation["route_deviation_m"]))
        self._write({
            "type": "step",
            "input_observation": input_observation,
            "next_observation": next_observation,
            "control": command,
            "decision": {
                "mode": decision["mode"],
                "mode_scores": decision["mode_scores"].tolist(),
                "confidence": decision["confidence"].tolist(),
                "behavior_probabilities": decision["behavior_probabilities"].tolist(),
                "history_valid": decision["history_valid"],
                "selected_trajectory": decision["trajectory"].tolist(),
            },
            "execution": {
                **execution,
                "reference_trajectory": execution["reference_trajectory"].tolist(),
            },
        })

    def finish_episode(self, observation, artifacts=None):
        """关闭当前 JSONL 并返回该 episode 的终态摘要。"""
        summary = {
            **self._episode,
            "status": observation["status"],
            "steps": observation["step"],
            "route_completion": observation["route_completion"],
            "distance_travelled_m": observation["distance_travelled_m"],
            "lane_invasions": observation["lane_invasions"],
            "end_distance_m": observation["end_distance_m"],
            "max_route_deviation_m": self._max_deviation,
            "artifacts": artifacts or {},
        }
        self._write({"type": "summary", **summary})
        self._stream.close()
        self._stream = None
        self._summaries.append(summary)
        return summary

    def finish_run(self):
        """写运行级汇总并返回同一字典。"""
        success = sum(item["status"] == "success" for item in self._summaries)
        aggregate = {
            "num_episodes": len(self._summaries),
            "successes": success,
            "success_rate": success / max(len(self._summaries), 1),
            "episodes": self._summaries,
        }
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as stream:
            json.dump(aggregate, stream, ensure_ascii=False, indent=2)
        return aggregate

    def close(self):
        """异常退出时关闭尚未关闭的 episode 文件。"""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _write(self, payload):
        self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._stream.flush()
