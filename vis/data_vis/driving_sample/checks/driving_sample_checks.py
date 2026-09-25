# 本文件为 vis/data_vis/driving_sample/driving_sample.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_driving_sample(sample):
    """校验对象: render_driving_sample.sample —— 需包含完整时序、分离占用与监督场。"""
    required = ("rgb", "history_rgb", "history_valid", "lidar_occupied", "drivable",
                "lane_class", "lane_direction", "lane_direction_valid",
                "stop_line", "scene_occ", "scene_occ_mask", "agent_occ", "agent_occ_mask",
                "detect_class", "detect_box", "detect_future", "detect_future_valid",
                "trajectory", "traj_valid", "target_point")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError("驾驶样本缺少可视化字段：{}".format(missing))
    if sample["rgb"].ndim != 4 or sample["history_rgb"].ndim != 5:
        raise ValueError("驾驶图像须分别为 [V,3,H,W] 与 [4,V,3,H,W]")
    if sample["scene_occ"].ndim != 3 or sample["agent_occ"].shape != sample["scene_occ"].shape:
        raise ValueError("场景与 Agent 占用须分别为同形状的 [Z,X,Y]")
    lane_shape = tuple(sample["lane_class"].shape)
    if sample["lane_class"].ndim != 2 \
            or tuple(sample["lane_direction"].shape) != (2, *lane_shape) \
            or tuple(sample["lane_direction_valid"].shape) != lane_shape:
        raise ValueError("车道语义须为 [H,W]，方向须为 [2,H,W]，有效掩码须为 [H,W]")
