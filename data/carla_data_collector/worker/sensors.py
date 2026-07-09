"""传感器阵列：逐视角按开关创建 RGB/Depth/语义/光流相机、语义分割 Lidar、碰撞传感器。

模块: worker/sensors.py
依赖: queue, carla, worker.geometry, worker.sensors_checks
读取配置: 由 SensorRig 接收 cameras_cfg(width/height/modalities/rig，其中 rig 含各相机 fov) 与 lidar_cfg(enabled,...)；自身不读 config
对外接口:
    - SensorRig(world, ego, cameras_cfg, lidar_cfg)
        .gather(frame_id, timeout_s) -> dict[str, carla 传感器数据]  # 严格收齐本帧
        .collided -> bool
        .extrinsics -> dict[str, list[6]]    # 各相机相对 ego 外参
        .lidar_extrinsic -> list[3]
        .sync_keys -> list[str]
        .destroy() -> None
说明: 相机模态（rgb/depth/semantic/optical_flow）与 lidar 各由 config 开关决定是否创建，关闭即不入 gather、不落盘。
      同一 rig 视角下被启用的各模态相机共享尺寸/FOV/位姿，故像素天然对齐；不同视角可用不同 FOV。
      每个同步传感器用独立 Queue 接收；gather 按 frame_id 收齐当前帧、丢弃陈旧帧、超前即报错，实现
      「收齐本帧所有传感器才推进」的严格同步（Design 同步要求）。碰撞传感器只置标志、不参与 gather。
"""

import queue

import carla

from worker.geometry import make_transform
from worker.sensors_checks import check_ego, check_no_future_frame


class SensorRig:
    def __init__(self, world, ego, cameras_cfg, lidar_cfg):
        check_ego(ego)
        self._world = world
        self._ego = ego
        self._cameras_cfg = cameras_cfg
        self._lidar_cfg = lidar_cfg
        self._actors = []
        self._queues = {}          # key -> Queue：参与同步 gather 的传感器
        self._collided = False
        self._extrinsics = {}
        self._lidar_extrinsic = [lidar_cfg.x, lidar_cfg.y, lidar_cfg.z]
        self._build()

    # ---------- 构建 ----------

    def _attach(self, blueprint, transform, key):
        sensor = self._world.spawn_actor(blueprint, transform, attach_to=self._ego)
        q = queue.Queue()
        sensor.listen(q.put)       # 回调仅入队，gather 再按帧取，避免回调里做重活
        self._actors.append(sensor)
        self._queues[key] = q

    # 相机模态 -> (蓝图类型, config 开关字段)；光流与深度同法，仅蓝图与下游解码不同
    _CAMERA_MODALITIES = (
        ("rgb", "sensor.camera.rgb"),
        ("depth", "sensor.camera.depth"),
        ("semantic", "sensor.camera.semantic_segmentation"),
        ("optical_flow", "sensor.camera.optical_flow"),
    )

    def _build(self):
        bl = self._world.get_blueprint_library()
        cam = self._cameras_cfg

        # 仅创建开关为真的相机模态；蓝图在各视角间复用（spawn 时快照属性，故循环内改 FOV 安全）
        enabled = [(name, bl.find(bp_type)) for name, bp_type in self._CAMERA_MODALITIES
                   if getattr(cam.modalities, name)]
        for _, bp in enabled:
            bp.set_attribute("image_size_x", str(cam.width))
            bp.set_attribute("image_size_y", str(cam.height))
        for c in cam.rig:              # 同视角各启用模态共享 FOV 与位姿，保证像素逐点对齐
            pose = [c.x, c.y, c.z, c.roll, c.pitch, c.yaw]
            self._extrinsics[c.name] = pose
            transform = make_transform(pose)
            for name, bp in enabled:
                bp.set_attribute("fov", str(c.fov))
                self._attach(bp, transform, name + "/" + c.name)

        if self._lidar_cfg.enabled:
            self._build_lidar(bl)

        col_bp = bl.find("sensor.other.collision")
        collision = self._world.spawn_actor(col_bp, carla.Transform(), attach_to=self._ego)
        collision.listen(self._on_collision)
        self._actors.append(collision)  # 不入 _queues：碰撞是事件型，不参与同步收齐

    def _build_lidar(self, bl):
        ld = self._lidar_cfg
        lidar_bp = bl.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("channels", str(ld.channels))
        lidar_bp.set_attribute("range", str(ld.range_m))
        lidar_bp.set_attribute("points_per_second", str(ld.points_per_second))
        lidar_bp.set_attribute("rotation_frequency", str(ld.rotation_frequency))
        lidar_bp.set_attribute("upper_fov", str(ld.upper_fov))
        lidar_bp.set_attribute("lower_fov", str(ld.lower_fov))
        self._attach(lidar_bp, make_transform(self._lidar_extrinsic + [0.0, 0.0, 0.0]), "lidar")

    def _on_collision(self, _event):
        self._collided = True

    # ---------- 采集 ----------

    def gather(self, frame_id, timeout_s):
        """阻塞收齐 frame_id 这一帧的全部同步传感器数据，返回 {key: 数据}。"""
        return {key: self._pull_matching(q, frame_id, timeout_s)
                for key, q in self._queues.items()}

    def _pull_matching(self, q, frame_id, timeout_s):
        """从单个传感器队列取出 frame==frame_id 的那帧；陈旧帧丢弃，超前帧视为时序错乱。"""
        while True:
            data = q.get(timeout=timeout_s)
            if data.frame == frame_id:
                return data
            check_no_future_frame(data.frame, frame_id)  # 超前即报错
            # 否则 data.frame < frame_id：上一轮残留的陈旧帧，丢弃后继续取

    @property
    def collided(self):
        return self._collided

    @property
    def extrinsics(self):
        return self._extrinsics

    @property
    def lidar_extrinsic(self):
        return self._lidar_extrinsic

    @property
    def sync_keys(self):
        return list(self._queues.keys())

    def destroy(self):
        """停止并销毁全部传感器 actor（场景结束清理）。"""
        for actor in self._actors:
            actor.stop()
            actor.destroy()
        self._actors = []
        self._queues = {}
