_COLOR_MODES = {"semantic", "source", "actor", "height"}


def check_render_state(state, data):
    # 校验对象: render_pointcloud 入参 state —— 着色枚举与动态帧必须可解释
    assert state.color_mode in _COLOR_MODES, "未知点云着色模式: {}".format(state.color_mode)
    assert state.spatial_scope in ("global", "bev"), "spatial_scope 仅支持 global/bev"
    assert hasattr(data, "xyz") and hasattr(data, "source"), "data 不是 FusionPointcloud 兼容对象"
    assert 0 <= int(state.frame_index) < data.num_frames, "frame_index 超出 ego_pose 帧范围"
