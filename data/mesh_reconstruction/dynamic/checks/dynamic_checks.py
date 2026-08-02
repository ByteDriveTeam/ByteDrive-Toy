import torch


def check_dynamic_inputs(objects, source_voxel_size):
    # 校验对象: reconstruct_dynamic_objects 入参 objects —— 须符合融合对象打包格式
    required = {"actor_id", "class_id", "extent", "point_offsets", "xyz_local", "obj_tag"}
    assert isinstance(objects, dict) and required <= set(objects), "动态对象输入字段不完整"
    count = len(objects["actor_id"])
    assert objects["actor_id"].dtype == torch.int64 \
        and objects["class_id"].shape == (count,) \
        and objects["extent"].shape == (count, 3) \
        and objects["point_offsets"].shape == (count + 1,), "动态对象索引字段非法"
    assert objects["point_offsets"].dtype == torch.int64 \
        and int(objects["point_offsets"][0]) == 0 \
        and int(objects["point_offsets"][-1]) == len(objects["xyz_local"]), \
        "动态点 offsets 非法"
    assert bool((objects["point_offsets"][1:] >= objects["point_offsets"][:-1]).all()), \
        "动态点 offsets 必须单调"
    assert source_voxel_size > 0, "源融合体素尺寸必须 > 0"


def check_dynamic_output(meshes):
    # 校验对象: reconstruct_dynamic_objects 返回 meshes —— packed offsets/索引须自洽
    count = len(meshes["actor_id"])
    assert meshes["vertex_offsets"].shape == (count + 1,) \
        and meshes["triangle_offsets"].shape == (count + 1,), "动态 Mesh offsets 非法"
    assert int(meshes["vertex_offsets"][-1]) == len(meshes["vertices_local"]) \
        and int(meshes["triangle_offsets"][-1]) == len(meshes["triangles"]), \
        "动态 Mesh offsets 与打包数据不一致"
    triangles = meshes["triangles"]
    if len(triangles):
        assert int(triangles.min()) >= 0 \
            and int(triangles.max()) < len(meshes["vertices_local"]), \
            "动态 Mesh 三角形索引越界"
    has_mesh = meshes["vertex_offsets"][1:] > meshes["vertex_offsets"][:-1]
    assert meshes["is_watertight"].dtype == torch.bool \
        and meshes["is_watertight"].shape == (count,), "动态 Mesh 水密标志非法"
    assert meshes["unsupported_triangles_removed"].dtype == torch.int64 \
        and meshes["unsupported_triangles_removed"].shape == (count,) \
        and bool((meshes["unsupported_triangles_removed"] >= 0).all()), \
        "动态 Mesh 无支撑裁剪统计非法"
    assert bool((~meshes["is_watertight"][~has_mesh]).all()), \
        "空动态 Mesh 不得标记为水密"
    assert bool((meshes["method_code"][~has_mesh] == 0).all()), \
        "空动态 Mesh 只能标记为 unobserved"
