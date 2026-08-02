"""稀疏 TUDF 查看器输入路径校验。"""

from pathlib import Path


def check_viewer_inputs(data, screenshot_dir, project_root):
    root, screenshot = Path(project_root).resolve(), Path(screenshot_dir).resolve()
    assert data.num_frames > 0, "TUDF 查看器要求至少一帧"
    assert screenshot != root and root in screenshot.parents, \
        "TUDF 截图目录必须严格位于项目内"

