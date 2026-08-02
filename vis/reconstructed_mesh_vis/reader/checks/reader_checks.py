from pathlib import Path

import torch


def check_mesh_path(path):
    # 校验对象: ReconstructedMesh 入参 path —— 必须是存在的 .mesh.pt
    path = Path(path)
    assert path.is_file() and path.name.endswith(".mesh.pt"), \
        "重建 Mesh 文件不存在或后缀非法: {}".format(path)


def check_mesh_payload(payload):
    # 校验对象: torch.load 返回 payload —— 须符合 Mesh schema v1
    required = {"static_mesh", "dynamic_meshes", "dynamic_poses", "ego_pose", "metadata"}
    assert isinstance(payload, dict) and required <= set(payload), "Mesh PT 字段不完整"
    assert payload["metadata"].get("schema_version") == 1, "仅支持 Mesh schema v1"
    static, dynamic, poses = (
        payload["static_mesh"], payload["dynamic_meshes"], payload["dynamic_poses"])
    static_required = {"vertices", "triangles", "vertex_normals", "vertex_obj_tag",
                       "is_watertight"}
    dynamic_required = {"actor_id", "class_id", "extent", "vertex_offsets",
                        "triangle_offsets", "vertices_local", "triangles",
                        "vertex_normals_local", "vertex_obj_tag", "method_code",
                        "donor_object_index", "is_watertight"}
    assert static_required <= set(static) and dynamic_required <= set(dynamic), \
        "Mesh 几何子字段不完整"
    vertices, triangles = static["vertices"], static["triangles"]
    assert vertices.dtype == torch.float32 and vertices.ndim == 2 \
        and vertices.shape[1] == 3 and triangles.dtype == torch.int64 \
        and triangles.ndim == 2 and triangles.shape[1] == 3, "静态 Mesh 形状非法"
    count = len(dynamic["actor_id"])
    assert dynamic["vertex_offsets"].shape == (count + 1,) \
        and dynamic["triangle_offsets"].shape == (count + 1,) \
        and dynamic["method_code"].shape == (count,) \
        and dynamic["donor_object_index"].shape == (count,), "动态 Mesh 索引非法"
    assert int(dynamic["vertex_offsets"][-1]) == len(dynamic["vertices_local"]) \
        and int(dynamic["triangle_offsets"][-1]) == len(dynamic["triangles"]), \
        "动态 Mesh offsets 非法"
    assert bool((dynamic["vertex_offsets"][1:] >= dynamic["vertex_offsets"][:-1]).all()) \
        and bool((dynamic["triangle_offsets"][1:] >= dynamic["triangle_offsets"][:-1]).all()) \
        and bool((dynamic["method_code"] <= 4).all()), "动态 Mesh offsets 或方法码非法"
    if len(dynamic["triangles"]):
        assert int(dynamic["triangles"].min()) >= 0 \
            and int(dynamic["triangles"].max()) < len(dynamic["vertices_local"]), \
            "动态 Mesh 三角形索引越界"
    pose_count = len(poses["object_index"])
    assert poses["object_index"].shape == (pose_count,) \
        and poses["frame_index"].shape == (pose_count,) \
        and poses["transform"].shape == (pose_count, 6), "动态位姿非法"
    ego = payload["ego_pose"]
    assert ego.dtype == torch.float32 and ego.ndim == 2 \
        and ego.shape[1] == 6 and len(ego) > 0, "自车位姿非法"
    if pose_count:
        assert int(poses["object_index"].min()) >= 0 \
            and int(poses["object_index"].max()) < count \
            and int(poses["frame_index"].min()) >= 0 \
            and int(poses["frame_index"].max()) < len(ego), "动态位姿索引越界"
        keys = poses["frame_index"].to(torch.int64) * max(count, 1) + poses["object_index"]
        assert len(torch.unique(keys)) == pose_count, "同一帧同一对象存在重复位姿"
    tensor_values = (vertices, dynamic["vertices_local"], poses["transform"], ego)
    assert all(bool(torch.isfinite(value).all()) for value in tensor_values), \
        "Mesh 或位姿含 NaN/Inf"
