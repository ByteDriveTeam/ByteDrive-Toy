"""词表可视化输入校验。"""

import numpy as np


def check_bundle(bundle):
    """校验对象: 分解词表字典，需包含词序列与聚类距离。"""
    if not isinstance(bundle, dict) or "vocab" not in bundle or "metrics" not in bundle:
        raise ValueError("分解词表缺少 vocab/metrics")
    if bundle.get("kind") not in ("geometry", "speed") or "mean_nearest_distance" not in bundle["metrics"]:
        raise ValueError("分解词表 kind 或 metrics 无效")
    vocab = np.asarray(bundle["vocab"])
    if vocab.ndim == 0 or vocab.shape[0] == 0 or (bundle["kind"] == "geometry" and
                             (vocab.ndim != 3 or vocab.shape[-1] != 2)) or (
                             bundle["kind"] == "speed" and vocab.ndim != 2):
        raise ValueError("分解词表 vocab 形状无效")


def check_action_bundle(bundle):
    """校验对象: 耦合动作词表，需包含二维加速度序列与采样率。"""
    if "actions" not in bundle or "hz" not in bundle:
        raise ValueError("耦合词表缺少 actions/hz")
    actions = bundle["actions"]
    if actions.ndim != 3 or actions.shape[-1] != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError("耦合词表 actions 必须为非空 [K,T,2]")
    if not np.isfinite(actions).all() or not np.isfinite(bundle["hz"]).all() or float(bundle["hz"]) <= 0:
        raise ValueError("耦合词表 actions/hz 必须有限且采样率为正")
    if "normalized_actions" in bundle:
        if bundle["normalized_actions"].shape != actions.shape or not np.allclose(
                bundle["normalized_actions"],
                (actions - bundle["action_mean"]) / bundle["action_std"], atol=1e-5):
            raise ValueError("耦合词表标准化加速度与物理词及统计量不一致")
    source_keys = {"source_trajectory", "source_velocity"}
    if source_keys & bundle.keys() and not source_keys <= bundle.keys():
        raise ValueError("耦合词表源轨迹与源初速必须同时提供")
    if source_keys <= bundle.keys():
        trajectory, velocity = bundle["source_trajectory"], bundle["source_velocity"]
        if trajectory.shape != actions.shape or velocity.shape != (len(actions), 2):
            raise ValueError("耦合词表源轨迹或源初速形状无效")
        if not np.isfinite(trajectory).all() or not np.isfinite(velocity).all():
            raise ValueError("耦合词表源轨迹与源初速必须有限")
        dt = 1.0 / float(bundle["hz"])
        reconstructed = np.cumsum((velocity[:, None] + np.cumsum(actions * dt, axis=1)) * dt, axis=1)
        if not np.allclose(reconstructed, trajectory, atol=1e-3, rtol=0):
            raise ValueError("耦合词表源轨迹与动作及源初速不一致")
