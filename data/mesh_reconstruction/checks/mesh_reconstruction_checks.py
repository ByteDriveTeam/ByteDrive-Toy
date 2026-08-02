from pathlib import Path

import torch


_SOURCE_KEYS = {"static", "dynamic_objects", "dynamic_poses", "ego_pose", "metadata"}


def check_input_path(path):
    # 校验对象: discover_pointclouds 入参 path —— 必须是存在的 PT 或目录
    path = Path(path)
    assert path.exists() and (path.is_dir() or path.suffix.lower() == ".pt"), \
        "Mesh 重建输入必须是存在的 .pt 或目录: {}".format(path)


def check_output_dir(output_dir, input_path, project_root):
    # 校验对象: run_reconstruction 输出目录 —— 必须在项目内且不能嵌套于输入目录
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    source = Path(input_path).resolve()
    assert output != root and root in output.parents, \
        "Mesh 输出目录必须严格位于项目目录内: {}".format(output)
    if source.is_dir():
        assert output != source and source not in output.parents, \
            "Mesh 输出目录不得位于输入目录内部: {}".format(output)


def check_output_path(path, project_root):
    # 校验对象: reconstruct_scene 输出文件 —— 必须是项目内 .mesh.pt
    root = Path(project_root).resolve()
    target = Path(path).resolve()
    assert target.name.endswith(".mesh.pt") and root in target.parents, \
        "Mesh 输出文件必须是项目内 .mesh.pt: {}".format(target)


def check_source_payload(payload):
    # 校验对象: reconstruct_scene 加载的 payload —— 须符合融合 schema v3 数据契约
    present = set(payload) if isinstance(payload, dict) else set()
    assert isinstance(payload, dict) and _SOURCE_KEYS <= present, \
        "融合 PT 缺少字段: {}".format(sorted(_SOURCE_KEYS - present))
    static, objects, poses = payload["static"], payload["dynamic_objects"], payload["dynamic_poses"]
    assert {"xyz", "obj_tag"} <= set(static) \
        and {"actor_id", "class_id", "extent", "point_offsets", "xyz_local", "obj_tag"} \
        <= set(objects) \
        and {"object_index", "frame_index", "transform"} <= set(poses), \
        "融合 PT 子字段不完整"
    points = static["xyz"]
    assert points.dtype == torch.float32 and points.ndim == 2 \
        and points.shape[1] == 3 and len(points) >= 4 \
        and static["obj_tag"].dtype == torch.uint8 \
        and static["obj_tag"].shape == (len(points),), "static 点或标签非法"
    count, dynamic_points = len(objects["actor_id"]), len(objects["xyz_local"])
    assert objects["actor_id"].dtype == torch.int64 \
        and objects["class_id"].dtype == torch.uint8 \
        and objects["class_id"].shape == (count,) \
        and objects["extent"].dtype == torch.float32 \
        and objects["extent"].shape == (count, 3) \
        and bool((objects["extent"] > 0).all()), "动态对象属性非法"
    assert objects["point_offsets"].dtype == torch.int64 \
        and objects["point_offsets"].shape == (count + 1,) \
        and int(objects["point_offsets"][0]) == 0 \
        and int(objects["point_offsets"][-1]) == dynamic_points \
        and objects["xyz_local"].shape == (dynamic_points, 3) \
        and objects["xyz_local"].dtype == torch.float32 \
        and objects["obj_tag"].shape == (dynamic_points,) \
        and objects["obj_tag"].dtype == torch.uint8, "动态对象点打包非法"
    pose_count = len(poses["object_index"])
    assert poses["object_index"].dtype == torch.int64 \
        and poses["object_index"].shape == (pose_count,) \
        and poses["frame_index"].dtype == torch.int32 \
        and poses["frame_index"].shape == (pose_count,) \
        and poses["transform"].dtype == torch.float32 \
        and poses["transform"].shape == (pose_count, 6), "动态位姿非法"
    ego = payload["ego_pose"]
    assert ego.dtype == torch.float32 and ego.ndim == 2 \
        and ego.shape[1] == 6 and len(ego) > 0, "自车位姿非法"
    if pose_count:
        assert int(poses["object_index"].min()) >= 0 \
            and int(poses["object_index"].max()) < count \
            and int(poses["frame_index"].min()) >= 0 \
            and int(poses["frame_index"].max()) < len(ego), "动态位姿索引越界"
    tensors = (points, objects["xyz_local"], poses["transform"], ego)
    assert all(bool(torch.isfinite(value).all()) for value in tensors), "融合 PT 含 NaN/Inf"
    metadata = payload["metadata"]
    assert isinstance(metadata, dict) \
        and metadata.get("coordinate_frames", {}).get("static.xyz") == "carla_world" \
        and metadata.get("coordinate_frames", {}).get("dynamic_objects.xyz_local") \
        == "actor_box_local", "融合 PT 坐标系声明非法"


def check_output_payload(payload):
    # 校验对象: reconstruct_scene 返回 payload —— 静态、动态、位姿和元数据必须完整
    required = {"static_mesh", "dynamic_meshes", "dynamic_poses", "ego_pose", "metadata"}
    assert required <= set(payload), "Mesh 输出字段不完整"
    static = payload["static_mesh"]
    assert len(static["vertices"]) > 0 and len(static["triangles"]) > 0, \
        "静态 Mesh 必须非空"
    repair_enabled = bool(payload["metadata"].get("reconstruction_config", {})
                          .get("repair", {}).get("enabled", True))
    assert not repair_enabled or bool(static["is_watertight"]), \
        "开启水密修复时静态 Mesh 必须通过水密验收"
    assert payload["metadata"].get("schema_version") == 1, "Mesh schema_version 非法"
