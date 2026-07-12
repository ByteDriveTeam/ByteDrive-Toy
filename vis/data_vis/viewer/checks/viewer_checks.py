# 校验对象: Viewer 的入参 reader —— 必须含可读帧（空场景无可视内容）
def check_viewer(reader):
    assert reader.num_frames > 0, "场景无帧可视化（num_frames=0）"
    assert len(reader.camera_names) > 0, "场景未记录任何相机，无法渲染"
