def check_render_state(data, state):
    # 校验对象: Mesh RenderState —— 帧索引、着色方式与数据类型必须可解释
    assert state.color_mode in ("semantic", "source", "actor", "method"), \
        "未知 Mesh 着色模式: {}".format(state.color_mode)
    assert 0 <= int(state.frame_index) < data.num_frames, "Mesh frame_index 越界"
    assert hasattr(data, "static") and hasattr(data, "dynamic") \
        and hasattr(data, "poses"), "data 不是 ReconstructedMesh 兼容对象"
