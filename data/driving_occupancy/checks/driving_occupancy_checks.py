"""独立占用缓存的数据来源、设备与预生成参数校验。"""

import torch


def check_occupancy_device(device):
    """校验对象: DrivingOccupancyCache.__init__ —— 显式 CUDA 设备必须可用。"""
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("占用缓存配置要求 CUDA，但当前 PyTorch/CUDA 不可用")


def check_cache_prepare(dataset, num_workers, prefetch_factor, progress_every):
    """校验对象: prepare_driving_occupancy_cache —— 数据集与并行参数合法。"""
    if not hasattr(dataset, "_prepare_occupancy_sample") or num_workers < 0 \
            or prefetch_factor < 1 or progress_every < 1:
        raise ValueError("驾驶占用预生成参数非法")


def check_fused_sources(missing):
    """校验对象: DrivingOccupancyCache.check_sources —— 每个训练场景须有融合 PT。"""
    if missing:
        raise FileNotFoundError("缺少融合场景 PT，请先运行 multiframe_pointcloud_fusion: "
                                + ", ".join(sorted(missing)))


def check_fused_signature(signature, scene, stat):
    """校验对象: DrivingOccupancyCache._payload —— PT 输入指纹须匹配当前 LMDB。"""
    if signature["scene_name"] != scene or signature["data_mdb_size"] != stat.st_size \
            or signature["data_mdb_mtime_ns"] != stat.st_mtime_ns:
        raise ValueError("融合 PT 与当前原始场景不匹配: {}".format(scene))
