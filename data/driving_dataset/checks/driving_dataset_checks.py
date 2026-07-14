# 本文件为 data/driving_dataset/driving_dataset.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_camera_calib(meta, camera):
    """校验对象: 场景 meta —— 指定相机须有内参与外参（驾驶模型依赖标定做几何反投影）。"""
    intr = meta.get("intrinsics", {})
    extr = meta.get("extrinsics", {})
    if camera not in intr or camera not in extr:
        raise KeyError("场景 meta 缺相机 {} 的内参/外参（intrinsics/extrinsics）。".format(camera))


def check_behavior_annotations(meta, frame, camera):
    """校验对象: DrivingDataset 行为监督源 —— Seg、动态框、灯静态元数据与逐帧状态须存在。"""
    if camera not in frame.get("semantic", {}):
        raise KeyError("行为红灯可见性判定需要相机 {} 的 semantic Seg。".format(camera))
    missing_meta = [key for key in ("traffic_lights", "static_bboxes") if key not in meta]
    missing_frame = [key for key in ("bboxes", "traffic_light_states") if key not in frame]
    if missing_meta or missing_frame:
        raise KeyError("行为监督缺标注：场景级 {}，帧级 {}。".format(missing_meta, missing_frame))
