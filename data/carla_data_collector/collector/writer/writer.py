"""把场景的非 RGB 数据与独立运动学时间序列写入 LMDB。

模块: collector/writer/writer.py
依赖: lmdb, msgpack, numpy, collector.writer_checks
读取配置: 由构造函数接收 output.lmdb_map_size_gb 与库路径，自身不读 config
对外接口:
    - LmdbWriter(path, map_size_gb)
        .write_scene(scene_meta, frames, kinematics=(), est_bytes=0) -> int
        .close() -> None
    - compact_lmdb(path, verify=True) -> (before_bytes, after_bytes)
    - append_model_data(path, map_size_gb, meta_updates, world_states, model_steps) -> None
    - read_scene_identity(path) -> tuple | None
    - pack_array(arr) -> bytes / unpack_array(blob) -> np.ndarray   # 数组打包/还原（含结构化 dtype）
说明: Design ⑧——RGB 之外的信息进 LMDB。每个场景一个独立 DB（co-located 于该场景目录），故键不带 scene_id
      前缀：直接 meta / num_frames / "{帧序号}/meta" / "{帧序号}/{模态}"；
      高频运动学另存 num_kinematics / "kinematics/{序号}"。数组以 (dtype,shape,bytes) 打包，
      结构化 dtype（语义Lidar）用 descr 列表保存，故还原无损。scene_meta 含 scene_id/seed/天气/路线/内外参/
      静态包围框/视频引用等，使单场景自描述、可独立读取。
      map_size（lmdb_map_size_gb）是「单场景 DB 的增长上限」而非初始大小：Windows 下 LMDB 会把数据文件
      实占到 map_size，故初始只开一小块、写入前按 est_bytes 估算按需扩容（封顶 map_size），避免预占满几十 GB。
"""

import os
import shutil
import tempfile
from pathlib import Path

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


def read_scene_map_route(lmdb_path):
    """读取场景地图与路线键，返回 ``(map, start_idx, end_idx)``。

    供多地图断点续采按地图隔离路线索引。旧场景缺少地图、库不可读或路线字段不完整时
    返回 None；保留 :func:`read_scene_route` 的原接口供其他调用方使用。
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
        meta = msgpack.unpackb(blob, raw=False)
        route = meta.get("route") or {}
        map_name = meta.get("map")
        if not map_name or "start_idx" not in route or "end_idx" not in route:
            return None
        return (str(map_name), int(route["start_idx"]), int(route["end_idx"]))
    finally:
        env.close()


def read_scene_identity(lmdb_path):
    """读取 ``(map, controller, start_idx, end_idx)``，旧数据默认专家控制。"""
    try:
        env = lmdb.open(str(lmdb_path), readonly=True, subdir=True, lock=False)
    except lmdb.Error:
        return None
    try:
        with env.begin() as txn:
            blob = txn.get(_key("meta"))
        if blob is None:
            return None
        meta = msgpack.unpackb(blob, raw=False)
        route = meta.get("route") or {}
        if not meta.get("map") or "start_idx" not in route or "end_idx" not in route:
            return None
        controller = str(meta.get("controller", "behavior_agent"))
        if not meta.get("complete", True):
            return None
        if controller == "model" and not meta.get("drive_terminal_segment", False):
            return None
        return (str(meta["map"]), controller,
                int(route["start_idx"]), int(route["end_idx"]))
    finally:
        env.close()


def append_model_data(path, map_size_gb, meta_updates, world_states, model_steps):
    """原子回填一个模型段的 10Hz 世界、预测和离线原始代价时间轴。"""
    writer = LmdbWriter(path, map_size_gb)
    try:
        writer.append_model_data(meta_updates, world_states, model_steps)
    finally:
        writer.close()


def _verify_lmdb_equal(source_path, compact_path):
    """逐键、逐值校验两个 LMDB 完全一致；不解包数组，避免额外峰值内存。"""
    source = lmdb.open(
        str(source_path), readonly=True, subdir=True, lock=False, readahead=False)
    compact = lmdb.open(
        str(compact_path), readonly=True, subdir=True, lock=False, readahead=False)
    try:
        if source.stat()["entries"] != compact.stat()["entries"]:
            raise RuntimeError("LMDB compact 校验失败：条目数不一致")
        with source.begin(buffers=True) as source_txn, compact.begin(buffers=True) as compact_txn:
            source_cursor = source_txn.cursor()
            compact_cursor = compact_txn.cursor()
            source_item = source_cursor.first()
            compact_item = compact_cursor.first()
            while source_item and compact_item:
                source_key = source_cursor.key()
                compact_key = compact_cursor.key()
                if source_key != compact_key:
                    raise RuntimeError(
                        "LMDB compact 校验失败：键不一致 {!r} != {!r}".format(
                            bytes(source_key), bytes(compact_key)))
                if source_cursor.value() != compact_cursor.value():
                    raise RuntimeError(
                        "LMDB compact 校验失败：键 {!r} 的值不一致".format(
                            bytes(source_key)))
                source_item = source_cursor.next()
                compact_item = compact_cursor.next()
            if source_item != compact_item:
                raise RuntimeError("LMDB compact 校验失败：游标长度不一致")
    finally:
        compact.close()
        source.close()


def _copy_windows_dacl(source, destination):
    """把 source 的 Windows DACL 复制到 destination，避免临时文件带入错误权限。"""
    if os.name != "nt":
        return

    import ctypes
    from ctypes import wintypes

    dacl_security_information = 0x00000004
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    get_file_security = advapi32.GetFileSecurityW
    get_file_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
    ]
    get_file_security.restype = wintypes.BOOL
    set_file_security = advapi32.SetFileSecurityW
    set_file_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    set_file_security.restype = wintypes.BOOL

    needed = wintypes.DWORD()
    get_file_security(
        str(source), dacl_security_information, None, 0, ctypes.byref(needed))
    if needed.value == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(needed.value)
    if not get_file_security(
            str(source), dacl_security_information, descriptor, needed.value,
            ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not set_file_security(
            str(destination), dacl_security_information, descriptor):
        raise ctypes.WinError(ctypes.get_last_error())


def compact_lmdb(path, verify=True):
    """安全压实一个已关闭的 LMDB，并返回压实前后的 data.mdb 字节数。

    compact 副本始终建立在原库同级目录；可选的逐键值校验通过后，才原子替换
    data.mdb。键、值及其序列化格式均保持不变，只重排数据库页并移除尾部空洞。
    """
    path = Path(path).resolve()
    data_path = path / "data.mdb"
    if not path.is_dir() or not data_path.is_file():
        raise ValueError("LMDB 目录无效：{}".format(path))

    before = data_path.stat().st_size
    temp_path = Path(tempfile.mkdtemp(
        prefix=path.name + ".compact-", dir=str(path.parent))).resolve()
    if temp_path.parent != path.parent or not temp_path.name.startswith(path.name + ".compact-"):
        raise RuntimeError("拒绝在 LMDB 同级目录之外创建 compact 临时目录")

    try:
        source = lmdb.open(
            str(path), readonly=True, subdir=True, lock=False, readahead=False)
        try:
            source.copy(str(temp_path), compact=True)
        finally:
            source.close()

        if verify:
            _verify_lmdb_equal(path, temp_path)
        compact_data = temp_path / "data.mdb"
        after = compact_data.stat().st_size
        _copy_windows_dacl(data_path, compact_data)
        os.replace(str(compact_data), str(data_path))
        return before, after
    finally:
        if temp_path.exists():
            shutil.rmtree(str(temp_path))


# 初始映射大小：开小块、按需增长，规避 Windows 下一次性预占满 map_size
_INITIAL_MAP_BYTES = 64 * 1024 * 1024


class LmdbWriter:
    def __init__(self, path, map_size_gb):
        self._max_bytes = int(map_size_gb * 1024 ** 3)
        data_path = Path(path) / "data.mdb"
        existing = data_path.stat().st_size if data_path.is_file() else 0
        initial = min(self._max_bytes, max(_INITIAL_MAP_BYTES, existing * 2))
        self._env = lmdb.open(str(path), map_size=initial, subdir=True)

    def _ensure_capacity(self, extra_bytes):
        """确保映射上限能再容下 extra_bytes；不足则扩容（封顶 max_bytes）。"""
        info, stat = self._env.info(), self._env.stat()
        used = (info["last_pgno"] + 1) * stat["psize"]  # 已用字节估算
        need = used + extra_bytes
        assert need <= self._max_bytes, \
            "场景预计需 {} 字节，超出 LMDB 上限；请调大 output.lmdb_map_size_gb".format(extra_bytes)
        if need > info["map_size"]:
            self._env.set_mapsize(min(self._max_bytes, max(need, info["map_size"] * 2)))

    def write_scene(self, scene_meta, frames, kinematics=(), est_bytes=0):
        """原子写入本场景的全部数据（单事务，确保场景级一致性），返回写入帧数。

        参数:
            scene_meta: 场景级元数据 dict（scene_id/seed/天气/路线/内外参/静态包围框/视频引用等）
            frames:     可迭代，每项 {"meta": dict, "arrays": {key: np.ndarray}}
            kinematics: 独立异频运动学记录，可迭代，每项含 frame_id/sim_time/ego
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
            kinematics_count = 0
            for idx, state in enumerate(kinematics):
                txn.put(_key("kinematics", idx), msgpack.packb(state, use_bin_type=True))
                kinematics_count = idx + 1
            txn.put(_key("num_kinematics"), msgpack.packb(kinematics_count))
        return count

    def append_model_data(self, meta_updates, world_states, model_steps):
        """给已存在场景追加模型 10Hz 时间轴及逐候选、逐点原始代价数组。"""
        array_names = (
            "trajectories", "reference_trajectory",
            "candidate_cost_terms", "candidate_cost_valid",
            "current_cost_terms", "current_cost_valid", "next_cost_terms",
            "next_cost_valid", "historical_cost_terms", "historical_cost_valid",
        )
        step_metadata = [{key: value for key, value in step.items()
                          if key not in array_names} for step in model_steps]
        estimated = (
            sum(np.asarray(step[name]).nbytes
                for step in model_steps for name in array_names if name in step)
            + sum(len(msgpack.packb(state, use_bin_type=True)) for state in world_states)
            + sum(len(msgpack.packb(state, use_bin_type=True)) for state in step_metadata))
        self._ensure_capacity(int(estimated * 1.3) + 32 * 1024 * 1024)
        with self._env.begin(write=True) as txn:
            meta_blob = txn.get(_key("meta"))
            if meta_blob is None:
                raise RuntimeError("模型回填目标 LMDB 缺少 meta")
            meta = msgpack.unpackb(meta_blob, raw=False)
            meta.update(meta_updates)
            txn.put(_key("meta"), msgpack.packb(meta, use_bin_type=True))
            for index, state in enumerate(world_states):
                txn.put(_key("world", index), msgpack.packb(state, use_bin_type=True))
            txn.put(_key("num_world_states"), msgpack.packb(len(world_states)))
            for index, (step, step_meta) in enumerate(zip(model_steps, step_metadata)):
                txn.put(_key("model", index, "meta"),
                        msgpack.packb(step_meta, use_bin_type=True))
                for name in array_names:
                    if name in step:
                        txn.put(_key("model", index, name),
                                pack_array(np.asarray(step[name])))
            txn.put(_key("num_model_steps"), msgpack.packb(len(model_steps)))

    def close(self):
        self._env.close()
