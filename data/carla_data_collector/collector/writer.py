"""把场景的非 RGB 数据写入 LMDB（深度/语义Lidar/包围框/主车状态/元数据/视频引用）。

模块: collector/writer.py
依赖: lmdb, msgpack, numpy, collector.writer_checks
读取配置: 由构造函数接收 output.lmdb_map_size_gb 与库路径，自身不读 config
对外接口:
    - LmdbWriter(path, map_size_gb)
        .write_scene(scene_meta, frames, est_bytes=0) -> int   # 返回写入帧数
        .close() -> None
    - pack_array(arr) -> bytes / unpack_array(blob) -> np.ndarray   # 数组打包/还原（含结构化 dtype）
说明: Design ⑧——RGB 之外的信息进 LMDB。每个场景一个独立 DB（co-located 于该场景目录），故键不带 scene_id
      前缀：直接 meta / num_frames / "{帧序号}/meta" / "{帧序号}/{模态}"。数组以 (dtype,shape,bytes) 打包，
      结构化 dtype（语义Lidar）用 descr 列表保存，故还原无损。scene_meta 含 scene_id/seed/天气/路线/内外参/
      静态包围框/视频引用等，使单场景自描述、可独立读取。
      map_size（lmdb_map_size_gb）是「单场景 DB 的增长上限」而非初始大小：Windows 下 LMDB 会把数据文件
      实占到 map_size，故初始只开一小块、写入前按 est_bytes 估算按需扩容（封顶 map_size），避免预占满几十 GB。
"""

import lmdb
import msgpack
import numpy as np


def pack_array(arr):
    """把 ndarray 打包为 msgpack 字节：保留 dtype（结构化用 descr）与 shape，data 为原始字节。"""
    dtype_field = [list(t) for t in arr.dtype.descr] if arr.dtype.fields else arr.dtype.str
    return msgpack.packb({"dtype": dtype_field, "shape": list(arr.shape),
                          "data": arr.tobytes()}, use_bin_type=True)


def unpack_array(blob):
    """pack_array 的逆操作，还原为 ndarray（结构化 dtype 一并还原）。"""
    obj = msgpack.unpackb(blob, raw=False)
    dtype = np.dtype([tuple(t) for t in obj["dtype"]]) if isinstance(obj["dtype"], list) \
        else np.dtype(obj["dtype"])
    return np.frombuffer(obj["data"], dtype=dtype).reshape(obj["shape"])


def _key(*parts):
    return "/".join(str(p) for p in parts).encode("utf-8")


def read_scene_route(lmdb_path):
    """读取已落盘场景 LMDB 的 meta，返回其路线键 (start_idx, end_idx)。

    供断点续采据此排除已采路线。库不存在/打不开/无 meta/缺路线字段时返回 None
    （视作该场景路线不可识别，不纳入排除）。
    """
    try:
        env = lmdb.open(str(lmdb_path), readonly=True, subdir=True, lock=False)
    except lmdb.Error:
        return None
    try:
        with env.begin() as txn:
            blob = txn.get(_key("meta"))
        if blob is None:
            return None
        route = msgpack.unpackb(blob, raw=False).get("route") or {}
        if "start_idx" not in route or "end_idx" not in route:
            return None
        return (int(route["start_idx"]), int(route["end_idx"]))
    finally:
        env.close()


# 初始映射大小：开小块、按需增长，规避 Windows 下一次性预占满 map_size
_INITIAL_MAP_BYTES = 64 * 1024 * 1024


class LmdbWriter:
    def __init__(self, path, map_size_gb):
        self._max_bytes = int(map_size_gb * 1024 ** 3)
        self._env = lmdb.open(str(path), map_size=min(_INITIAL_MAP_BYTES, self._max_bytes),
                              subdir=True)

    def _ensure_capacity(self, extra_bytes):
        """确保映射上限能再容下 extra_bytes；不足则扩容（封顶 max_bytes）。"""
        info, stat = self._env.info(), self._env.stat()
        used = (info["last_pgno"] + 1) * stat["psize"]  # 已用字节估算
        need = used + extra_bytes
        assert need <= self._max_bytes, \
            "场景预计需 {} 字节，超出 LMDB 上限；请调大 output.lmdb_map_size_gb".format(extra_bytes)
        if need > info["map_size"]:
            self._env.set_mapsize(min(self._max_bytes, max(need, info["map_size"] * 2)))

    def write_scene(self, scene_meta, frames, est_bytes=0):
        """原子写入本场景的全部数据（单事务，确保场景级一致性），返回写入帧数。

        参数:
            scene_meta: 场景级元数据 dict（scene_id/seed/天气/路线/内外参/静态包围框/视频引用等）
            frames:     可迭代，每项 {"meta": dict, "arrays": {key: np.ndarray}}
            est_bytes:  本场景预估写入字节（由调用方据形状算出），据此按需扩容；留 30% 余量
        """
        self._ensure_capacity(int(est_bytes * 1.3) + 16 * 1024 * 1024)
        # frames 通常是惰性生成器（逐帧消费、内存只驻留一帧），故帧数靠迭代计数、写在末尾
        count = 0
        with self._env.begin(write=True) as txn:
            txn.put(_key("meta"), msgpack.packb(scene_meta, use_bin_type=True))
            for idx, frame in enumerate(frames):
                txn.put(_key(idx, "meta"), msgpack.packb(frame["meta"], use_bin_type=True))
                for name, arr in frame["arrays"].items():
                    txn.put(_key(idx, name), pack_array(arr))
                count = idx + 1
            txn.put(_key("num_frames"), msgpack.packb(count))
        return count

    def close(self):
        self._env.close()
