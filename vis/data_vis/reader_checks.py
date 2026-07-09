from pathlib import Path


# 校验对象: SceneReader 的入参 scene_dir —— 必须是含 lmdb 子库的场景目录
def check_scene_dir(scene_dir):
    p = Path(scene_dir)
    assert p.is_dir(), "场景目录不存在: {}".format(p)
    assert (p / "lmdb").is_dir(), "场景目录缺少 lmdb 子库: {}".format(p)


# 校验对象: SceneReader.frame 的入参 i —— 必须落在 [0, num_frames)
def check_frame_index(i, num_frames):
    assert 0 <= i < num_frames, "帧序号 {} 越界（共 {} 帧）".format(i, num_frames)
