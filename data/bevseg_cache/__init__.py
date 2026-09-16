"""BEVSeg 压缩磁盘缓存公共 API。

模块: data/bevseg_cache/__init__.py
依赖: data.bevseg_cache.bevseg_cache
读取配置: —
对外接口:
    - BevSegDiskCache(cfg, fingerprint_payload) -> object
    - prepare_bevseg_cache(dataset, num_workers, prefetch_factor, in_order, progress_every) -> dict
"""

from data.bevseg_cache.bevseg_cache import BevSegDiskCache, prepare_bevseg_cache

__all__ = ["BevSegDiskCache", "prepare_bevseg_cache"]
