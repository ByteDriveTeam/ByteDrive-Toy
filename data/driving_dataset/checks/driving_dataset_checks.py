# 本文件为 data/driving_dataset/driving_dataset.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_selected_scene(index, scene_name):
    """校验对象: DrivingDataset.scene_name —— 指定场景必须存在且至少有一帧。"""
    if not index:
        raise ValueError("驾驶数据集未找到指定场景 {} 的样本帧。".format(scene_name))


def check_camera_calib(meta, cameras):
    """校验对象: 场景 meta —— 三目相机须按配置全部提供内参与外参。"""
    intr = meta.get("intrinsics", {})
    extr = meta.get("extrinsics", {})
    missing = [camera for camera in cameras if camera not in intr or camera not in extr]
    if missing:
        raise KeyError("场景 meta 缺三目相机 {} 的内参/外参。".format(missing))


def check_frame_cadence(meta, frame_offset, interval_s):
    """校验对象: DrivingDataset 历史和 Agent 未来帧 —— 采样时间须为 2Hz。"""
    import math
    if not math.isclose(float(meta["sensor_dt_s"]) * frame_offset,
                        interval_s, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("场景 sensor_dt_s 与未来 2Hz/历史帧间隔不一致")


def check_behavior_annotations(meta, frame, cameras):
    """校验驾驶输入：RGB 必需；监督源须有完整 Depth 或 LiDAR；Seg 可选。"""
    missing_rgb = [camera for camera in cameras if camera not in frame.get("rgb", {})]
    if missing_rgb:
        raise KeyError("驾驶输入缺 RGB 相机：{}。".format(missing_rgb))
    has_depth = all(camera in frame.get("depth", {}) for camera in cameras)
    has_lidar = frame.get("lidar") is not None and meta.get("lidar_extrinsic") is not None
    if not has_depth and not has_lidar:
        raise KeyError("驾驶监督至少需要完整三目 Depth 或 LiDAR。")
    missing_meta = [key for key in ("traffic_lights", "static_bboxes") if key not in meta]
    missing_frame = [key for key in ("bboxes", "traffic_light_states") if key not in frame]
    if missing_meta or missing_frame:
        raise KeyError("行为监督缺标注：场景级 {}，帧级 {}。".format(missing_meta, missing_frame))


def check_ego_box_annotations(ego_boxes):
    """校验对象: DrivingDataset LiDAR 自车点剔除源 —— 每帧须有唯一且完整的 ego Box。"""
    if len(ego_boxes) != 1:
        raise ValueError("逐帧标注期望唯一 ego Box，实际 {} 个。".format(len(ego_boxes)))
    box = ego_boxes[0]
    if any(len(box.get(key, ())) != 3 for key in ("location", "extent", "rotation")):
        raise ValueError("ego Box 期望 location/extent/rotation 均为三维。")
