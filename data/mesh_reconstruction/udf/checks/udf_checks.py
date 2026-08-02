"""稀疏 TUDF 张量契约校验。"""

from pathlib import Path

import torch


def check_udf_inputs(points, tags):
    # 校验对象: build_sparse_udf 输入 —— 点为有限 float32[N,3]，标签逐点对齐
    assert points.dtype == torch.float32 and points.ndim == 2 \
        and points.shape[1] == 3 and tags.dtype == torch.uint8 \
        and tags.shape == (len(points),), "TUDF 输入点或标签非法"
    assert bool(torch.isfinite(points).all()), "TUDF 输入点含 NaN/Inf"


def check_udf_output_path(path, project_root):
    # 校验对象: reconstruct_udf_scene 输出 —— 必须是项目内 .udf.pt
    path, root = Path(path).resolve(), Path(project_root).resolve()
    assert path.name.endswith(".udf.pt") and root in path.parents, \
        "TUDF 输出文件必须是项目内 .udf.pt: {}".format(path)


def check_udf_field(field, require_unique=True):
    # 校验对象: 单个稀疏 TUDF —— 坐标唯一且距离、权重、标签、法线逐体素对齐
    required = {"voxel_coords", "udf", "weight", "observation_count",
                "obj_tag", "normal", "voxel_size_m", "truncation_m"}
    assert required <= set(field), "稀疏 TUDF 字段不完整"
    count = len(field["voxel_coords"])
    assert field["voxel_coords"].dtype == torch.int32 \
        and field["voxel_coords"].shape == (count, 3), "TUDF 坐标非法"
    assert field["udf"].dtype == torch.float32 and field["udf"].shape == (count,) \
        and field["weight"].dtype == torch.float32 \
        and field["weight"].shape == (count,), "TUDF 数值或权重非法"
    assert field["observation_count"].dtype == torch.int32 \
        and field["observation_count"].shape == (count,) \
        and field["obj_tag"].dtype == torch.uint8 \
        and field["obj_tag"].shape == (count,), "TUDF 观测或标签非法"
    assert field["normal"].dtype == torch.float32 \
        and field["normal"].shape == (count, 3), "TUDF 法线非法"
    assert field["voxel_size_m"].numel() == 1 \
        and field["truncation_m"].numel() == 1, "TUDF 尺度必须为标量"
    assert bool(torch.isfinite(field["udf"]).all()) \
        and bool(torch.isfinite(field["weight"]).all()) \
        and bool(torch.isfinite(field["normal"]).all()), "TUDF 含 NaN/Inf"
    assert bool((field["udf"] >= 0).all()) \
        and bool((field["udf"] <= field["truncation_m"] + 1e-6).all()) \
        and bool((field["weight"] >= 0).all()) \
        and bool((field["weight"] <= 1).all()), "TUDF 数值范围非法"
    if count and require_unique:
        assert len(torch.unique(field["voxel_coords"], dim=0)) == count, \
            "TUDF 含重复体素坐标"


def check_packed_dynamic(field, verify_unique=True):
    # 校验对象: 动态对象 packed TUDF —— 对象属性、offsets 与逐体素字段须对齐
    required = {"actor_id", "class_id", "extent", "voxel_offsets",
                "voxel_coords_local", "udf", "weight", "observation_count",
                "obj_tag", "normal_local", "voxel_size_m", "truncation_m"}
    assert required <= set(field), "动态 TUDF 字段不完整"
    objects, voxels = len(field["actor_id"]), len(field["voxel_coords_local"])
    assert field["actor_id"].dtype == torch.int64 \
        and field["class_id"].shape == (objects,) \
        and field["extent"].shape == (objects, 3), "动态 TUDF 对象属性非法"
    assert field["voxel_offsets"].dtype == torch.int64 \
        and field["voxel_offsets"].shape == (objects + 1,) \
        and int(field["voxel_offsets"][0]) == 0 \
        and int(field["voxel_offsets"][-1]) == voxels \
        and bool((field["voxel_offsets"][1:] >= field["voxel_offsets"][:-1]).all()), \
        "动态 TUDF offsets 非法"
    proxy = {
        "voxel_coords": field["voxel_coords_local"], "udf": field["udf"],
        "weight": field["weight"], "observation_count": field["observation_count"],
        "obj_tag": field["obj_tag"], "normal": field["normal_local"],
        "voxel_size_m": field["voxel_size_m"], "truncation_m": field["truncation_m"],
    }
    check_udf_field(proxy, require_unique=False)
    if verify_unique:
        offsets = field["voxel_offsets"]
        for index in range(objects):
            first, last = int(offsets[index]), int(offsets[index + 1])
            assert len(torch.unique(field["voxel_coords_local"][first:last], dim=0)) \
                == last - first, "单个动态对象 TUDF 含重复体素坐标"


def check_udf_payload(payload, verify_unique=True):
    # 校验对象: 统一 TUDF PT —— 静动态场、逐帧位姿及元数据须完整
    required = {"static_udf", "dynamic_udfs", "dynamic_poses", "ego_pose", "metadata"}
    assert required <= set(payload), "TUDF 产物字段不完整"
    check_udf_field(payload["static_udf"], require_unique=verify_unique)
    check_packed_dynamic(payload["dynamic_udfs"], verify_unique=verify_unique)
    assert payload["metadata"].get("schema_version") == 1 \
        and payload["metadata"].get("representation") == "sparse_tudf", \
        "TUDF schema 声明非法"
    assert payload["ego_pose"].dtype == torch.float32 \
        and payload["ego_pose"].ndim == 2 and payload["ego_pose"].shape[1] == 6, \
        "TUDF 自车位姿非法"
    poses = payload["dynamic_poses"]
    pose_count = len(poses["object_index"])
    assert poses["object_index"].shape == (pose_count,) \
        and poses["frame_index"].shape == (pose_count,) \
        and poses["transform"].shape == (pose_count, 6), "TUDF 动态位姿非法"
