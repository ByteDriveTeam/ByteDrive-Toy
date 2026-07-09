"""单场景严格同步采集循环：逐帧收齐传感器、交通灯状态、共享内存数据与帧索引。

模块: worker/collect.py
依赖: carla, numpy, common.protocol, common.shm, worker.annotations, worker.geometry, worker.collect_checks
读取配置: 由调用方接收 simulation(warmup_ticks/fixed_delta) 与 collection(max_frames/capture_every_n)，自身不读 config
对外接口:
    - prepare_drive(world, agent, sim_cfg, destination)
        -> (reachable, static_bboxes, traffic_lights, traffic_light_metadata)
    - collect_chunk(world, ego, agent, rig, crowd, traffic_lights, allocator, collection_cfg,
                    timeout_s, counters) -> (status, frames)
说明: 大块张量（RGB/Depth/语义图/光流/语义Lidar）写入 arena（仅记 offset/size/shape/dtype）；小元数据（位姿/控制/
      包围框/交通灯状态）内联进帧索引随控制管道回传。RGB 与 Depth 均存 BGR 三通道（丢 alpha 省内存），depth 的
      编码值交由 collector 解码；语义图只取标签所在的 R 通道存单通道 uint8（标签已是最终值，无需解码）；
      光流存 (H,W,2) float32 运动矢量、无需解码。实际采集哪些模态由 config 开关决定，本文件按 sample 中的 key 自适应。
      一次行驶被切成多段：每填满一次 arena 即返回 partial（本段落盘后复用同一世界续采）；counters 跨段
      累计帧数与 tick，故总帧上限与采样节拍按整次行驶统计。碰撞→当前未落盘段丢弃、行驶结束（Design ④）。
"""

import carla
import numpy as np

from common import protocol as P
from common.shm import ArenaFull
from worker import annotations
from worker.geometry import transform_to_list
from worker.collect_checks import check_destination

_LIDAR_ITEMSIZE = np.dtype(P.SEMANTIC_LIDAR_DTYPE).itemsize
_TRAFFIC_LIGHT_STATE_NAMES = {
    int(carla.TrafficLightState.Red): "red",
    int(carla.TrafficLightState.Yellow): "yellow",
    int(carla.TrafficLightState.Green): "green",
    int(carla.TrafficLightState.Off): "off",
    int(carla.TrafficLightState.Unknown): "unknown",
}


def _camera_bgr(image):
    """carla 相机图像 -> (bytes, shape, dtype)。丢弃 alpha，保留 BGR 三通道。"""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    bgr = np.ascontiguousarray(arr[:, :, :3])
    return bgr.tobytes(), [image.height, image.width, 3], "uint8"


def _semantic_tags(image):
    """carla 语义分割图 -> (bytes, shape, dtype)。标签编码在 R 通道，取单通道 uint8（H,W）省内存。"""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    tags = np.ascontiguousarray(arr[:, :, 2])  # BGRA 中索引 2 为 R，即 CityScapes 语义标签
    return tags.tobytes(), [image.height, image.width], "uint8"


def _lidar_blob(measurement):
    """语义 Lidar -> (bytes, shape, dtype_token)。原样转发结构化字节，无损。"""
    raw = bytes(measurement.raw_data)
    return raw, [len(raw) // _LIDAR_ITEMSIZE], "semantic_lidar"


def _optical_flow(image):
    """光流相机 -> (bytes, shape, dtype)。每像素 (vx,vy) float32，原样存 (H,W,2) 供下游直接消费。"""
    arr = np.frombuffer(image.raw_data, dtype=np.float32).reshape(image.height, image.width, 2)
    return np.ascontiguousarray(arr).tobytes(), [image.height, image.width, 2], "float32"


def _encode_sensor(key, data):
    """按传感器类别选择编码：lidar 结构化、semantic 单通道标签、光流双通道浮点、其余相机 BGR 三通道。"""
    if key == "lidar":
        return _lidar_blob(data)
    if key.startswith("semantic/"):
        return _semantic_tags(data)
    if key.startswith("optical_flow/"):
        return _optical_flow(data)
    return _camera_bgr(data)


def _store_frame(allocator, sample):
    """把一帧全部传感器写入 arena，返回 {key: blob 描述}。"""
    blobs = {}
    for key, data in sample.items():
        raw, shape, dtype = _encode_sensor(key, data)
        offset, size = allocator.put(raw)  # 越界抛 ArenaFull，由上层捕获
        blobs[key] = {P.BLOB_OFFSET: offset, P.BLOB_SIZE: size,
                      P.BLOB_SHAPE: shape, P.BLOB_DTYPE: dtype}
    return blobs


def _ego_state(ego):
    """主车状态：位姿、速度、当次控制量。"""
    velocity = ego.get_velocity()
    control = ego.get_control()
    return {
        "transform": transform_to_list(ego.get_transform()),
        "velocity": [velocity.x, velocity.y, velocity.z],
        "control": {"throttle": control.throttle, "steer": control.steer,
                    "brake": control.brake, "reverse": control.reverse},
    }


def _traffic_light_metadata(traffic_lights):
    """生成交通灯静态元数据；触发区位置转换到世界坐标，供下游计算主车距离。"""
    metadata = []
    for light in traffic_lights:
        transform = light.get_transform()
        trigger = light.trigger_volume
        trigger_location = carla.Location(
            x=trigger.location.x, y=trigger.location.y, z=trigger.location.z)
        transform.transform(trigger_location)
        metadata.append({
            "id": light.id,
            "transform": transform_to_list(transform),
            "trigger_location": [trigger_location.x, trigger_location.y, trigger_location.z],
            "trigger_extent": [trigger.extent.x, trigger.extent.y, trigger.extent.z],
        })
    return metadata


def _traffic_light_states(traffic_lights):
    """读取当前仿真帧内全部交通灯状态，保持按 actor ID 排序的稳定顺序。"""
    state_codes = [int(light.state) for light in traffic_lights]
    return [{"id": light.id, "state": _TRAFFIC_LIGHT_STATE_NAMES.get(code, "unknown"),
             "state_code": code}
            for light, code in zip(traffic_lights, state_codes)]


def prepare_drive(world, agent, sim_cfg, destination):
    """一次行驶的准备：预热、设目标、可达性判定、采静态框与交通灯。仅在 start_scene 调一次。

    参数:
        world/agent: 已布置好的世界与行为体
        sim_cfg:     simulation 配置片段（warmup_ticks 等）
        destination: carla.Location 目标点
    返回:
        (reachable, static_bboxes, traffic_lights, traffic_light_metadata)：不可达时后三项为空列表
    """
    check_destination(destination)

    # 预热：交通流与物理稳定后再采（预热帧产生的陈旧传感器数据会在首次 gather 时被丢弃）
    for _ in range(sim_cfg.warmup_ticks):
        world.tick()

    agent.set_destination(destination)
    if agent.done():
        return False, [], [], []  # 规划不出路线，交由 collector 换组合/种子
    traffic_lights = sorted(world.get_actors().filter("*traffic_light*"), key=lambda light: light.id)
    return (True, annotations.static_bboxes(world), traffic_lights,
            _traffic_light_metadata(traffic_lights))


def collect_chunk(world, ego, agent, rig, crowd, traffic_lights, allocator, collection_cfg,
                  timeout_s, counters):
    """采集一段（填满一次 arena 为止）。counters 跨段累计，使总帧上限/采样节拍按整次行驶统计。

    参数:
        world/ego/agent/rig: 已布置好的世界、主车、行为体、传感器阵列
        crowd:     行人群管理器（WalkerCrowd），逐 tick 重派到达者目标使行人持续漫游
        traffic_lights: 场景准备阶段缓存的全部交通灯 actor（按 ID 排序）
        allocator: 写入共享内存的顺序分配器（调用前须已 reset，本段写满即返回 partial）
        collection_cfg: collection 配置片段（max_frames_per_scene 为整次行驶总帧上限 / capture_every_n_ticks）
        timeout_s: 单次 gather 等待传感器的超时
        counters:  {"total": 已落帧数, "tick_idx": 已 tick 数}，由调用方跨段持有并就地更新
    返回:
        (status, frames)：status 见 protocol.STATUS_*；frames 为本段帧索引列表
    """
    frames = []
    stride = collection_cfg.capture_every_n_ticks

    while counters["total"] < collection_cfg.max_frames_per_scene and not agent.done():
        crowd.retarget_arrived()  # 到达目标的行人重派新目标，避免后半程站住不动
        # 同步模式：先下控制、再 tick，使本帧物理按该控制推进
        ego.apply_control(agent.run_step())
        frame_id = world.tick()
        counters["tick_idx"] += 1
        sample = rig.gather(frame_id, timeout_s)  # 严格收齐本帧所有传感器

        if rig.collided:
            return P.STATUS_COLLISION, frames  # 当前未落盘段丢弃、行驶结束（Design ④）
        if (counters["tick_idx"] - 1) % stride != 0:
            continue  # 未到采样间隔：已 gather 维持同步，但不落帧

        try:
            blobs = _store_frame(allocator, sample)
        except ArenaFull:
            # arena 满：本段到此为止，由 collector 落盘后续采下一段。
            # 触发越界的这一帧已 tick/gather 但只部分写入，直接丢弃（下一段从新 tick 重采）。
            return P.STATUS_PARTIAL, frames

        frames.append({
            "frame_id": frame_id,
            "sim_time": world.get_snapshot().timestamp.elapsed_seconds,
            "ego": _ego_state(ego),
            "blobs": blobs,
            "bboxes": annotations.dynamic_bboxes(world, ego.id),
            "traffic_light_states": _traffic_light_states(traffic_lights),
        })
        counters["total"] += 1

    # 循环正常退出：到达终点（done）或达整次行驶总帧上限
    return (P.STATUS_OK if agent.done() else P.STATUS_MAX_FRAMES), frames
