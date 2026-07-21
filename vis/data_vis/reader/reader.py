"""场景读取器：合并单场景的 LMDB（深度/语义lidar/标注/元数据）与 mp4（RGB）为逐帧数据。

模块: vis/data_vis/reader/reader.py
依赖: cv2, lmdb, msgpack, numpy, collector.writer(unpack_array), vis.data_vis.reader.checks.reader_checks
读取配置: —（场景目录由调用方传入；样式参数不在本层）
对外接口:
    - SceneReader(scene_dir)
        .meta -> dict                 # 场景级元数据（内外参/静态框/相机名/视频引用等）
        .num_frames -> int
        .camera_names -> list[str]
        .available -> dict[str,bool]  # 各模态是否实际落盘：rgb/depth/semantic/optical_flow/lidar
        .rgb(i, camera) -> np.ndarray # 只解码指定相机 RGB，不读取 LMDB 大数组
        .frame(i, modalities=None) -> dict  # 标注 + 所选大数组模态；None 解码全部，传子集只解码所需
        .frame_meta(i) -> dict        # 仅逐帧元数据（ego/bboxes/交通灯…），不解码 RGB/不取大数组
        .close()
    - list_scenes(root) -> list[Path] # root 下的 scene_* 目录（按名排序）
说明: RGB 随机读用 cv2.VideoCapture（顺序播放走 read()，跳帧才 set POS_FRAMES，规避 hevc 频繁 seek）；
      每路视频仅保留最近一帧，使驾驶双帧数据的重叠历史帧无需回跳重解码，缓存大小恒定。
      数组解码复用 collector.writer.unpack_array，确保与写入端的 (dtype,shape,bytes) 格式单一来源、无损还原；
      为此把采集模块根加入 sys.path（vis 是其数据的消费者）。各传感器模态由采集端开关决定是否存在，故构造时
      探测首帧实际落盘的模态（available），frame() 只返回存在的模态；旧场景缺交通灯状态时返回空列表。
"""

import sys
from pathlib import Path

import cv2
import lmdb
import msgpack
import numpy as np

# 复用写入端的数组解包，避免在 vis 侧重写一份 (dtype,shape,bytes) 解析（DRY，规范 §8）
_COLLECTOR_ROOT = Path(__file__).resolve().parents[3] / "data" / "carla_data_collector"
if str(_COLLECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_COLLECTOR_ROOT))
from collector.writer import unpack_array  # noqa: E402  # pyright: ignore[reportMissingImports]  （依赖上面的 sys.path 引导）

from vis.data_vis.reader.checks.reader_checks import check_scene_dir, check_frame_index


def list_scenes(root):
    """返回 root 下所有 scene_* 子目录（按名排序），供选择默认场景。"""
    root = Path(root)
    return sorted(p for p in root.glob("scene_*") if p.is_dir()) if root.is_dir() else []


class _Mp4Reader:
    """单个相机 mp4 的逐帧随机读取：顺序读高效，跳帧才 seek。"""

    def __init__(self, path):
        self._cap = cv2.VideoCapture(str(path))
        self._next = 0  # 下一次 read() 将返回的帧序号
        self._last_idx = None
        self._last_frame = None

    def at(self, idx):
        """返回第 idx 帧 BGR 图（解码失败返回 None）。"""
        if idx == self._last_idx:
            return self._last_frame
        if idx != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
        ok, frame = self._cap.read()
        self._next = idx + 1 if ok else idx
        self._last_idx = idx if ok else None
        self._last_frame = frame if ok else None
        return self._last_frame

    def close(self):
        self._cap.release()
        self._last_frame = None


class SceneReader:
    def __init__(self, scene_dir):
        scene_dir = Path(scene_dir)
        check_scene_dir(scene_dir)
        self._env = lmdb.open(str(scene_dir / "lmdb"), readonly=True, subdir=True, lock=False)
        with self._env.begin() as txn:
            self.meta = msgpack.unpackb(txn.get(b"meta"), raw=False)
            self.num_frames = msgpack.unpackb(txn.get(b"num_frames"))
        self.camera_names = self.meta["camera_names"]
        # RGB 仅在采集开启时落 mp4；video_files 为空即该场景无 RGB，不建解码器
        video_files = self.meta.get("video_files", {})
        self._videos = {cam: _Mp4Reader(scene_dir / video_files[cam])
                        for cam in self.camera_names if cam in video_files}
        # 各模态由采集端开关决定，探测首帧首相机的实际落盘键，frame() 据此只取存在的模态
        self.available = self._detect_available(bool(self._videos))

    def _detect_available(self, has_rgb):
        """探测本场景实际包含哪些模态（首帧首相机为准），兼容任意采集开关组合。"""
        cam0 = self.camera_names[0]
        with self._env.begin() as txn:
            present = {m: txn.get(self._key(0, m, cam0)) is not None
                       for m in ("depth", "semantic", "optical_flow")}
            present["lidar"] = txn.get(self._key(0, "lidar")) is not None
        present["rgb"] = has_rgb
        return present

    def _cam_arrays(self, txn, i, modality, wanted):
        """读某模态在第 i 帧的逐相机数组（模态未落盘或未被 wanted 选中则空 dict，跳过解码开销）。"""
        if not self.available[modality] or (wanted is not None and modality not in wanted):
            return {}
        return {cam: unpack_array(txn.get(self._key(i, modality, cam))) for cam in self.camera_names}

    def frame_meta(self, i):
        """只读第 i 帧的逐帧元数据（ego/bboxes/交通灯等），不解码 RGB/不取大数组。

        供需要跨帧读取轻量状态（如未来 ego 位姿构造轨迹 GT）的下游避免逐帧建 VideoCapture。
        """
        check_frame_index(i, self.num_frames)
        with self._env.begin() as txn:
            return msgpack.unpackb(txn.get(self._key(i, "meta")), raw=False)

    def rgb(self, i, camera):
        """只读取指定帧/相机的 RGB，供时序模型避免为历史帧解码全部监督模态。"""
        check_frame_index(i, self.num_frames)
        return self._videos[camera].at(i)

    def frame(self, i, modalities=None):
        """读取第 i 帧的标注与所选大数组模态，组装为一个 dict（缺失/未选模态为空/None）。

        modalities=None 解码全部已落盘大数组（默认，保持既有行为）；传入模态名集合（depth/semantic/
        optical_flow/lidar 的任意子集）时只解码其中的大数组，供只需部分模态的下游跳过无用解码与分配开销。
        rgb 与逐帧元数据（ego/bboxes/交通灯）恒返回，不受 modalities 影响。
        """
        check_frame_index(i, self.num_frames)
        wanted = None if modalities is None else set(modalities)
        with self._env.begin() as txn:
            fmeta = msgpack.unpackb(txn.get(self._key(i, "meta")), raw=False)
            depth = self._cam_arrays(txn, i, "depth", wanted)
            semantic = self._cam_arrays(txn, i, "semantic", wanted)
            optical_flow = self._cam_arrays(txn, i, "optical_flow", wanted)
            want_lidar = self.available["lidar"] and (wanted is None or "lidar" in wanted)
            lidar = unpack_array(txn.get(self._key(i, "lidar"))) if want_lidar else None
        rgb = {cam: v.at(i) for cam, v in self._videos.items()}
        return {"rgb": rgb, "depth": depth, "semantic": semantic, "optical_flow": optical_flow,
                "lidar": lidar, "ego": fmeta["ego"], "bboxes": fmeta["bboxes"],
                "traffic_light_states": fmeta.get("traffic_light_states", []), "meta": fmeta}

    @staticmethod
    def _key(*parts):
        return "/".join(str(p) for p in parts).encode("utf-8")

    def close(self):
        for v in self._videos.values():
            v.close()
        self._env.close()
