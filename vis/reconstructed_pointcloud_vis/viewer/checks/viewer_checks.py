from pathlib import Path


def check_viewer_inputs(data, screenshot_dir, project_root):
    # 校验对象: PointcloudViewer 入参 data —— 须为已解析的融合点云
    assert hasattr(data, "xyz") and hasattr(data, "dynamic_frames") \
        and hasattr(data, "scene_name") and hasattr(data, "ego_pose") \
        and data.num_frames > 0, "data 不是含自车位姿的 FusionPointcloud 兼容对象"
    # 校验对象: screenshot_dir —— 所有写入必须严格位于项目目录内部
    root = Path(project_root).resolve()
    target = Path(screenshot_dir).resolve()
    assert target != root and root in target.parents, \
        "截图目录必须位于项目目录内部: {}".format(target)
