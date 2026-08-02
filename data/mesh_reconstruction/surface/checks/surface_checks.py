import torch


def check_surface_inputs(points, tags, orientation_targets):
    # 校验对象: reconstruct_surface 入参 points/tags —— 点与标签须逐点对齐且有限
    assert torch.is_tensor(points) and points.dtype == torch.float32 \
        and points.ndim == 2 and points.shape[1] == 3 and len(points) >= 4, \
        "表面重建 points 必须为至少四点的 float32[N,3]"
    assert torch.is_tensor(tags) and tags.dtype == torch.uint8 \
        and tags.shape == (len(points),), "表面重建 tags 必须为 uint8[N]"
    assert bool(torch.isfinite(points).all()), "表面重建 points 含 NaN/Inf"
    # 校验对象: reconstruct_surface 入参 orientation_targets —— 静态朝向目标须为有限 [M,3]
    if orientation_targets is not None:
        assert torch.is_tensor(orientation_targets) \
            and orientation_targets.dtype == torch.float32 \
            and orientation_targets.ndim == 2 \
            and orientation_targets.shape[1] == 3 and len(orientation_targets) > 0 \
            and bool(torch.isfinite(orientation_targets).all()), \
            "orientation_targets 必须为非空有限 float32[M,3]"


def check_sampled_points(points):
    # 校验对象: Poisson 前的体素点 —— 原生求解器要求点集在三轴上具有有效跨度
    centered = points - points.mean(dim=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    tolerance = torch.finfo(points.dtype).eps * eigenvalues[-1].clamp_min(1)
    assert bool(eigenvalues[0] > tolerance), \
        "点集近似退化到平面或直线，拒绝进入原生 Poisson 求解器"


def check_surface_output(mesh):
    # 校验对象: reconstruct_surface 返回值 —— 数值、索引与真实水密标志必须自洽
    required = {"vertices", "triangles", "vertex_normals", "vertex_obj_tag",
                "is_watertight", "unsupported_triangles_removed"}
    assert required <= set(mesh), "表面 Mesh 字段不完整"
    vertices, triangles = mesh["vertices"], mesh["triangles"]
    assert vertices.dtype == torch.float32 and vertices.ndim == 2 \
        and vertices.shape[1] == 3 and len(vertices) > 0, "Mesh vertices 非法"
    assert triangles.dtype == torch.int64 and triangles.ndim == 2 \
        and triangles.shape[1] == 3 and len(triangles) > 0, "Mesh triangles 非法"
    assert mesh["vertex_normals"].shape == vertices.shape \
        and mesh["vertex_normals"].dtype == torch.float32, "Mesh normals 非法"
    assert mesh["vertex_obj_tag"].shape == (len(vertices),) \
        and mesh["vertex_obj_tag"].dtype == torch.uint8, "Mesh tags 非法"
    assert bool(torch.isfinite(vertices).all()) \
        and bool(torch.isfinite(mesh["vertex_normals"]).all()), "Mesh 含 NaN/Inf"
    assert int(triangles.min()) >= 0 and int(triangles.max()) < len(vertices), \
        "Mesh triangles 顶点索引越界"
    assert torch.is_tensor(mesh["is_watertight"]) \
        and mesh["is_watertight"].dtype == torch.bool \
        and mesh["is_watertight"].numel() == 1, "Mesh 水密标志非法"
    assert mesh["unsupported_triangles_removed"].dtype == torch.int64 \
        and mesh["unsupported_triangles_removed"].numel() == 1 \
        and int(mesh["unsupported_triangles_removed"]) >= 0, \
        "Mesh 无支撑裁剪统计非法"
