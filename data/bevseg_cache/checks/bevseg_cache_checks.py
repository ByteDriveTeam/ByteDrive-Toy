import numpy as np


def check_cache_arrays(semantic, direction, history_frames, layers, resolution):
    """校验对象: BevSegDiskCache.store 输入 —— 栅格形状与可无损编码的数据类型。"""
    semantic_shape = (layers, resolution, resolution)
    direction_shape = (2, resolution, resolution)
    if semantic.shape != semantic_shape or direction.shape != direction_shape:
        raise ValueError(
            "BEVSeg 缓存输入形状错误: semantic={} direction={}".format(
                semantic.shape, direction.shape))
    if semantic.dtype != np.float32 or direction.dtype != np.float32:
        raise TypeError("BEVSeg 缓存仅接受 rasterizer 输出的 float32 数组")
    if not np.logical_or(semantic == 0.0, semantic == 1.0).all():
        raise ValueError("BEVSeg 语义缓存只能 bit-pack 二值栅格")


def check_cache_payload(semantic_bits, direction, semantic_count, direction_shape):
    """校验对象: BevSegDiskCache.load 负载 —— 文件字段形状必须匹配当前契约。"""
    expected_bytes = (semantic_count + 7) // 8
    if semantic_bits.dtype != np.uint8 or semantic_bits.shape != (expected_bytes,):
        raise ValueError("BEVSeg 缓存 semantic_bits 形状或类型错误")
    if direction.dtype != np.float32 or direction.shape != direction_shape:
        raise ValueError("BEVSeg 缓存 direction 形状或类型错误")


def check_cache_prepare(dataset, num_workers, prefetch_factor, progress_every):
    """校验对象: prepare_bevseg_cache 输入 —— 缓存须可写且 worker 参数合法。"""
    if not dataset.cache_enabled or not dataset.cache_writable:
        raise ValueError("离线预热要求 data.bevseg.cache.enabled/write_missing 均为 true")
    if len(dataset) == 0:
        raise ValueError("BEVSeg 数据集为空，无法预热缓存")
    if num_workers < 0 or prefetch_factor <= 0 or progress_every <= 0:
        raise ValueError(
            "缓存预热 num_workers 必须 >= 0，prefetch_factor/progress_every 必须 > 0")
