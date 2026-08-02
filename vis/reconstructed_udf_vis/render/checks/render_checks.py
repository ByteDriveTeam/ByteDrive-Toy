"""稀疏 TUDF 渲染状态校验。"""


def check_render_state(data, state):
    assert state.color_mode in ("udf", "weight", "semantic", "source", "actor"), \
        "未知 TUDF 着色模式"
    assert 0 <= state.frame_index < data.num_frames, "TUDF 帧索引越界"

