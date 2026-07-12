# 本文件为 vis/pred_vis/render.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_grid_rows(rows):
    """校验对象: render_grid 入参 rows —— 非空、每行 (label, panels) 且各行帧数一致。"""
    if not rows:
        raise ValueError("render_grid 的 rows 不能为空。")
    frame_counts = {len(panels) for _, panels in rows}
    if len(frame_counts) != 1:
        raise ValueError("各行的帧数必须一致，实际为 {}。".format(sorted(frame_counts)))
    if 0 in frame_counts:
        raise ValueError("每行至少需要一帧面板。")
