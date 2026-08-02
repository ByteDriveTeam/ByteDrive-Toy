from pathlib import Path


def check_viewer_inputs(data, screenshot_dir, project_root):
    # 校验对象: MeshViewer 入参 data —— 须含静态几何、自车位姿和至少一帧
    assert hasattr(data, "static") and hasattr(data, "dynamic") \
        and hasattr(data, "ego_pose") and data.num_frames > 0, \
        "data 不是 ReconstructedMesh 兼容对象"
    # 校验对象: screenshot_dir —— 所有截图必须严格写在项目目录内
    root, target = Path(project_root).resolve(), Path(screenshot_dir).resolve()
    assert target != root and root in target.parents, \
        "Mesh 截图目录必须位于项目目录内部: {}".format(target)
