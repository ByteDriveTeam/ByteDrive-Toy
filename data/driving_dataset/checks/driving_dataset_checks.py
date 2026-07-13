# 本文件为 data/driving_dataset/driving_dataset.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_camera_calib(meta, camera):
    """校验对象: 场景 meta —— 指定相机须有内参与外参（驾驶模型依赖标定做几何反投影）。"""
    intr = meta.get("intrinsics", {})
    extr = meta.get("extrinsics", {})
    if camera not in intr or camera not in extr:
        raise KeyError("场景 meta 缺相机 {} 的内参/外参（intrinsics/extrinsics）。".format(camera))
