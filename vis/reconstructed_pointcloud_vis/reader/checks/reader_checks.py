from pathlib import Path

import torch


_REQUIRED_KEYS = {"static", "dynamic_objects", "dynamic_poses", "ego_pose", "metadata"}


def check_pointcloud_path(path):
    # 校验对象: FusionPointcloud 入参 path —— 必须指向一个融合产物 PT 文件
    path = Path(path)
    assert path.is_file() and path.suffix.lower() == ".pt", \
        "重建点云文件不存在或不是 .pt: {}".format(path)


def check_pointcloud_payload(payload):
    # 校验对象: torch.load 返回的 payload —— 须符合静态/对象模型/逐帧位姿分离格式
    present = set(payload) if isinstance(payload, dict) else set()
    assert isinstance(payload, dict) and _REQUIRED_KEYS <= present, \
        "重建点云缺少字段: {}".format(sorted(_REQUIRED_KEYS - present))
    static, objects, poses = payload["static"], payload["dynamic_objects"], payload["dynamic_poses"]
    assert {"xyz", "obj_tag"} <= set(static), "static 缺少 xyz/obj_tag"
    assert {"actor_id", "class_id", "extent", "point_offsets", "xyz_local", "obj_tag"} \
        <= set(objects), "dynamic_objects 字段不完整"
    assert {"object_index", "frame_index", "transform"} <= set(poses), \
        "dynamic_poses 字段不完整"
    tensors = tuple(static.values()) + tuple(objects.values()) + tuple(poses.values()) \
        + (payload["ego_pose"],)
    assert all(torch.is_tensor(value) for value in tensors), "点云数值字段必须是 torch.Tensor"
    assert static["xyz"].dtype == torch.float32 and static["xyz"].ndim == 2 \
        and static["xyz"].shape[1] == 3 \
        and static["obj_tag"].dtype == torch.uint8 \
        and static["obj_tag"].shape == (len(static["xyz"]),), "static 字段形状或 dtype 非法"
    count, points = len(objects["actor_id"]), len(objects["xyz_local"])
    assert objects["actor_id"].dtype == torch.int64 \
        and objects["class_id"].shape == (count,) and objects["class_id"].dtype == torch.uint8 \
        and objects["extent"].shape == (count, 3) and objects["extent"].dtype == torch.float32 \
        and objects["point_offsets"].shape == (count + 1,) \
        and objects["point_offsets"].dtype == torch.int64 \
        and objects["xyz_local"].shape == (points, 3) \
        and objects["xyz_local"].dtype == torch.float32 \
        and objects["obj_tag"].shape == (points,) and objects["obj_tag"].dtype == torch.uint8, \
        "dynamic_objects 字段形状或 dtype 非法"
    pose_count = len(poses["object_index"])
    assert poses["object_index"].shape == (pose_count,) \
        and poses["object_index"].dtype == torch.int64 \
        and poses["frame_index"].shape == (pose_count,) \
        and poses["frame_index"].dtype == torch.int32 \
        and poses["transform"].shape == (pose_count, 6) \
        and poses["transform"].dtype == torch.float32, "dynamic_poses 字段形状或 dtype 非法"
    ego_pose = payload["ego_pose"]
    assert ego_pose.ndim == 2 and ego_pose.shape[1] == 6 \
        and ego_pose.dtype == torch.float32 and len(ego_pose) > 0, \
        "ego_pose 必须为 float32[F,6]，且至少包含一帧"
    assert isinstance(payload["metadata"], dict) \
        and payload["metadata"].get("coordinate_frames", {}).get("static.xyz") \
        == "carla_world", "metadata 坐标系声明非法"
    assert bool(torch.isfinite(static["xyz"]).all()) \
        and bool(torch.isfinite(objects["xyz_local"]).all()) \
        and bool(torch.isfinite(poses["transform"]).all()) \
        and bool(torch.isfinite(ego_pose).all()), "点云或位姿含 NaN/Inf"
