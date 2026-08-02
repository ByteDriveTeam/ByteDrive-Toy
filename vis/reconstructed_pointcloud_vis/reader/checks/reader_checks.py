from pathlib import Path

import torch


_REQUIRED_KEYS = {
    "xyz", "obj_tag", "source", "actor_id", "frame_index", "ego_pose", "metadata"}


def check_pointcloud_path(path):
    # 校验对象: FusionPointcloud 入参 path —— 必须指向一个融合产物 PT 文件
    path = Path(path)
    assert path.is_file() and path.suffix.lower() == ".pt", \
        "重建点云文件不存在或不是 .pt: {}".format(path)


def check_pointcloud_payload(payload):
    # 校验对象: torch.load 返回的 payload —— 须符合多帧融合统一点级格式
    present = set(payload) if isinstance(payload, dict) else set()
    assert isinstance(payload, dict) and _REQUIRED_KEYS <= set(payload), \
        "重建点云缺少字段: {}".format(sorted(_REQUIRED_KEYS - present))
    xyz, tag, source = payload["xyz"], payload["obj_tag"], payload["source"]
    actor, frame, ego_pose = payload["actor_id"], payload["frame_index"], payload["ego_pose"]
    assert all(torch.is_tensor(value) for value in (xyz, tag, source, actor, frame, ego_pose)), \
        "重建点云点级字段必须是 torch.Tensor"
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float32, \
        "xyz 必须为 float32[N,3]"
    assert tag.shape == (count,) and tag.dtype == torch.uint8, "obj_tag 必须为 uint8[N]"
    assert source.shape == (count,) and source.dtype == torch.uint8, "source 必须为 uint8[N]"
    assert actor.shape == (count,) and actor.dtype == torch.int64, "actor_id 必须为 int64[N]"
    assert frame.shape == (count,) and frame.dtype == torch.int32, "frame_index 必须为 int32[N]"
    assert ego_pose.ndim == 2 and ego_pose.shape[1] == 6 \
        and ego_pose.dtype == torch.float32 and len(ego_pose) > 0, \
        "ego_pose 必须为 float32[F,6]，且至少包含一帧"
    assert isinstance(payload["metadata"], dict), "metadata 必须为 dict"
    assert payload["metadata"].get("coordinate_frame") == "carla_world", \
        "metadata.coordinate_frame 必须为 carla_world"
    assert bool(torch.isfinite(xyz).all()), "xyz 含 NaN/Inf，Open3D 无法可靠显示"
    assert bool(torch.isfinite(ego_pose).all()), "ego_pose 含 NaN/Inf"
    assert bool(((source == 0) | (source == 1)).all()), "source 仅允许 0/1"
    static = source == 0
    dynamic = source == 1
    assert bool(((actor[static] == -1) & (frame[static] == -1)).all()), \
        "静态点 actor_id/frame_index 必须为 -1"
    assert bool(((actor[dynamic] >= 0) & (frame[dynamic] >= 0)).all()), \
        "动态点 actor_id/frame_index 必须非负"
    assert not bool(dynamic.any()) or int(frame[dynamic].max()) < len(ego_pose), \
        "动态点 frame_index 超出 ego_pose 帧数"
