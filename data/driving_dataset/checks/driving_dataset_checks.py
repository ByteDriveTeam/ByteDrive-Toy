# 本文件为 data/driving_dataset/driving_dataset.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_camera_calib(meta, cameras):
    """校验对象: 场景 meta —— 三目相机须按配置全部提供内参与外参。"""
    intr = meta.get("intrinsics", {})
    extr = meta.get("extrinsics", {})
    missing = [camera for camera in cameras if camera not in intr or camera not in extr]
    if missing:
        raise KeyError("场景 meta 缺三目相机 {} 的内参/外参。".format(missing))


def check_behavior_annotations(meta, frame, cameras):
    """校验对象: DrivingDataset 三目输入/监督源 —— RGB、Depth、Seg、框与灯标注须存在。"""
    missing_modalities = {
        modality: [camera for camera in cameras if camera not in frame.get(modality, {})]
        for modality in ("rgb", "depth", "semantic")
    }
    missing_modalities = {key: value for key, value in missing_modalities.items() if value}
    if missing_modalities:
        raise KeyError("驾驶三目输入/监督缺相机模态：{}。".format(missing_modalities))
    missing_meta = [key for key in ("traffic_lights", "static_bboxes") if key not in meta]
    missing_frame = [key for key in ("bboxes", "traffic_light_states") if key not in frame]
    if missing_meta or missing_frame:
        raise KeyError("行为监督缺标注：场景级 {}，帧级 {}。".format(missing_meta, missing_frame))
