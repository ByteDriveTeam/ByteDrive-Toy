from pathlib import Path

import torch


_LIDAR_FIELDS = {"x", "y", "z", "obj_idx", "obj_tag"}


def check_input_path(path):
    """校验对象: discover_scenes 的 input_path —— 输入必须是已存在目录。"""
    if not Path(path).is_dir():
        raise ValueError("融合输入目录不存在：{}".format(path))


def check_scene_dir(scene_dir):
    """校验对象: fuse_scene 的 scene_dir —— 必须包含可读 LMDB 数据文件。"""
    data_path = Path(scene_dir) / "lmdb" / "data.mdb"
    if not data_path.is_file():
        raise ValueError("场景缺少 lmdb/data.mdb：{}".format(scene_dir))


def check_output_dir(output_dir, repo_root):
    """校验对象: 融合输出目录 —— 解析后必须位于项目目录内部。"""
    output = Path(output_dir).resolve()
    root = Path(repo_root).resolve()
    if output != root and root not in output.parents:
        raise ValueError("输出目录必须位于项目目录内部：{}".format(output))


def check_scene_header(meta, num_frames):
    """校验对象: 场景 LMDB 的 meta/num_frames —— 融合所需标定与帧数必须完整。"""
    if not isinstance(meta, dict) or "lidar_extrinsic" not in meta:
        raise ValueError("场景 meta 缺少 lidar_extrinsic")
    if len(meta["lidar_extrinsic"]) != 3:
        raise ValueError("lidar_extrinsic 必须是三维平移")
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("num_frames 必须是正整数")


def check_frame(frame_index, frame_meta, lidar):
    """校验对象: 单帧 meta/lidar —— 位姿、动态 Box 与语义点字段必须完整。"""
    if not isinstance(frame_meta, dict) or "ego" not in frame_meta or "bboxes" not in frame_meta:
        raise ValueError("第 {} 帧缺少 ego/bboxes".format(frame_index))
    pose = frame_meta["ego"].get("transform") if isinstance(frame_meta["ego"], dict) else None
    if pose is None or len(pose) != 6:
        raise ValueError("第 {} 帧 ego.transform 必须为六维".format(frame_index))
    boxes = frame_meta["bboxes"]
    ego_boxes = [box for box in boxes if box.get("semantic") == "ego"]
    if len(ego_boxes) != 1 or ego_boxes[0].get("id") is None:
        raise ValueError("第 {} 帧必须有唯一且带 id 的 ego Box".format(frame_index))
    dynamic = [box for box in boxes if box.get("semantic") in ("vehicle", "pedestrian")]
    required = ("id", "location", "extent", "rotation")
    if any(box.get("id") is None or any(len(box.get(key, ())) != 3 for key in required[1:])
           or any(value <= 0 for value in box["extent"]) for box in dynamic):
        raise ValueError("第 {} 帧动态 Box 字段不完整或 extent 非正".format(frame_index))
    actor_ids = [box["id"] for box in dynamic]
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError("第 {} 帧动态 Box actor id 重复".format(frame_index))
    if lidar is None or lidar.dtype.names is None \
            or not _LIDAR_FIELDS.issubset(lidar.dtype.names):
        raise ValueError("第 {} 帧缺少语义 LiDAR 必要字段".format(frame_index))


def check_run_checkpoint(checkpoint, fingerprint, scene_names):
    """校验对象: 数据集场景级断点 —— 输入、配置、场景顺序与进度结构须一致。"""
    required = {"fingerprint", "scene_names", "completed", "current_scene", "errors"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("场景级断点缺少必要字段")
    if checkpoint["fingerprint"] != fingerprint or checkpoint["scene_names"] != scene_names:
        raise ValueError("场景级断点与当前输入、场景顺序或融合配置不一致")
    completed, errors = checkpoint["completed"], checkpoint["errors"]
    if not isinstance(completed, dict) or not isinstance(errors, dict):
        raise ValueError("场景级断点 completed/errors 必须为字典")
    if not set(completed).issubset(scene_names) or not set(errors).issubset(scene_names):
        raise ValueError("场景级断点包含当前数据集之外的场景")
    completed_fields = {"output", "fingerprint", "output_size", "output_mtime_ns"}
    if any(not isinstance(value, dict) or not completed_fields.issubset(value)
           for value in completed.values()):
        raise ValueError("场景级断点的已完成记录不完整")
    if any(not isinstance(value["output"], str) or not isinstance(value["fingerprint"], str)
           for value in completed.values()):
        raise ValueError("场景级断点的输出路径或指纹类型非法")
    if any(not isinstance(value["output_size"], int)
           or not isinstance(value["output_mtime_ns"], int)
           or value["output_size"] < 0 or value["output_mtime_ns"] < 0
           for value in completed.values()):
        raise ValueError("场景级断点的输出文件签名非法")
    if any(not isinstance(value, str) for value in errors.values()):
        raise ValueError("场景级断点的错误记录必须为字符串")
    current = checkpoint["current_scene"]
    if current is not None and current not in scene_names:
        raise ValueError("场景级断点 current_scene 非法")


def check_output_payload(payload, fingerprint):
    """校验对象: 最终 PT —— 静态、对象模型、逐帧位姿与自车位姿须规范化且一致。"""
    required = {"static", "dynamic_objects", "dynamic_poses", "ego_pose", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("最终 PT 缺少必要字段")
    if payload["metadata"].get("fingerprint") != fingerprint:
        raise ValueError("最终 PT 指纹与当前输入或配置不一致")
    num_frames = payload["metadata"].get("input_signature", {}).get("num_frames")
    ego_pose = payload["ego_pose"]
    if not torch.is_tensor(ego_pose) or ego_pose.dtype != torch.float32 \
            or ego_pose.ndim != 2 or ego_pose.shape != (num_frames, 6) \
            or not bool(torch.isfinite(ego_pose).all()):
        raise ValueError("最终 PT ego_pose 必须为有限 float32[F,6]，且 F 等于场景帧数")
    static = payload["static"]
    if not isinstance(static, dict) or not {"xyz", "obj_tag"}.issubset(static) \
            or static["xyz"].dtype != torch.float32 or static["xyz"].ndim != 2 \
            or static["xyz"].shape[1] != 3 or static["obj_tag"].dtype != torch.uint8 \
            or static["obj_tag"].shape != (len(static["xyz"]),) \
            or not bool(torch.isfinite(static["xyz"]).all()):
        raise ValueError("最终 PT static 必须包含有限 float32[Ns,3] xyz 与 uint8[Ns] obj_tag")
    objects = payload["dynamic_objects"]
    object_fields = {"actor_id", "class_id", "extent", "point_offsets", "xyz_local", "obj_tag"}
    if not isinstance(objects, dict) or not object_fields.issubset(objects):
        raise ValueError("最终 PT dynamic_objects 缺少必要字段")
    actor_id, offsets, local = objects["actor_id"], objects["point_offsets"], objects["xyz_local"]
    count = len(actor_id)
    if actor_id.dtype != torch.int64 or actor_id.shape != (count,) \
            or objects["class_id"].dtype != torch.uint8 \
            or objects["class_id"].shape != (count,) \
            or objects["extent"].dtype != torch.float32 \
            or objects["extent"].shape != (count, 3) \
            or offsets.dtype != torch.int64 or offsets.shape != (count + 1,) \
            or local.dtype != torch.float32 or local.ndim != 2 or local.shape[1] != 3 \
            or objects["obj_tag"].dtype != torch.uint8 \
            or objects["obj_tag"].shape != (len(local),):
        raise ValueError("最终 PT dynamic_objects 字段形状或 dtype 非法")
    if offsets[0].item() != 0 or offsets[-1].item() != len(local) \
            or not bool((offsets[1:] >= offsets[:-1]).all()) \
            or not bool(torch.isfinite(local).all()) \
            or not bool(torch.isfinite(objects["extent"]).all()) \
            or not bool((objects["extent"] > 0).all()) \
            or (count and not bool((actor_id[1:] > actor_id[:-1]).all())) \
            or not bool((objects["class_id"] <= 1).all()):
        raise ValueError("最终 PT dynamic_objects 偏移、actor 顺序、类别或数值非法")
    poses = payload["dynamic_poses"]
    pose_fields = {"object_index", "frame_index", "transform"}
    if not isinstance(poses, dict) or not pose_fields.issubset(poses):
        raise ValueError("最终 PT dynamic_poses 缺少必要字段")
    pose_count = len(poses["object_index"])
    if poses["object_index"].dtype != torch.int64 \
            or poses["object_index"].shape != (pose_count,) \
            or poses["frame_index"].dtype != torch.int32 \
            or poses["frame_index"].shape != (pose_count,) \
            or poses["transform"].dtype != torch.float32 \
            or poses["transform"].shape != (pose_count, 6) \
            or not bool(torch.isfinite(poses["transform"]).all()):
        raise ValueError("最终 PT dynamic_poses 字段形状、dtype 或数值非法")
    if pose_count and (count == 0 \
            or int(poses["object_index"].min()) < 0 \
            or int(poses["object_index"].max()) >= count \
            or int(poses["frame_index"].min()) < 0 \
            or int(poses["frame_index"].max()) >= num_frames):
        raise ValueError("最终 PT dynamic_poses 对象或帧索引越界")
