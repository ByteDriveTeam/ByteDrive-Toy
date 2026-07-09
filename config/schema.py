"""配置的类型定义与加载期校验（参数约束的唯一来源）。

模块: config/schema.py
依赖: dataclasses, typing
读取配置: —（本文件定义配置结构本身，不读取具体键）
对外接口:
    - build_config(raw: dict) -> Config        # 由原始 dict 构造强类型配置对象
    - validate_config(cfg: Config) -> None     # 加载期一次性校验，非法即抛 AssertionError
说明: 用 dataclass 而非 pydantic，保证 Py3.7 与 Py3.12 双解释器均可导入（worker 侧虽不读
      config 文件，但两端共享同一份 schema 以理解下发的配置结构）。校验集中于此，运行期
      实现文件不再重复（规范 §7.3）。
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, List, get_type_hints


# ---------- 数据结构 ----------

@dataclass
class WorkerCfg:
    python_exe: str
    carla_host: str
    carla_port: int
    startup_timeout_s: float
    command_timeout_s: float


@dataclass
class IpcCfg:
    arena_name: str
    arena_size_mb: int
    slot_count: int


@dataclass
class SimulationCfg:
    map: str
    fixed_delta_seconds: float
    warmup_ticks: int


@dataclass
class RouteCfg:
    min_distance_m: float
    max_distance_m: float
    max_scenes: int
    queue_seed: int


@dataclass
class TrafficCfg:
    num_vehicles: int
    num_walkers: int
    vehicle_filter: str
    walker_filter: str
    walker_running_pct: float
    walker_arrival_radius_m: float
    tm_port: int


@dataclass
class WeatherCfg:
    randomize: bool


@dataclass
class EgoCfg:
    behavior: str
    vehicle_filter: str


@dataclass
class CameraCfg:
    name: str
    fov: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class ModalitiesCfg:
    rgb: bool
    depth: bool
    semantic: bool
    optical_flow: bool


@dataclass
class CamerasCfg:
    width: int
    height: int
    modalities: ModalitiesCfg
    rig: List[CameraCfg]


@dataclass
class LidarCfg:
    enabled: bool
    channels: int
    range_m: float
    points_per_second: int
    rotation_frequency: float
    upper_fov: float
    lower_fov: float
    x: float
    y: float
    z: float


@dataclass
class CollectionCfg:
    max_frames_per_scene: int
    capture_every_n_ticks: int


@dataclass
class CollisionCfg:
    max_retries_per_route: int


@dataclass
class OutputCfg:
    root: str
    lmdb_map_size_gb: int
    video_codec: str
    video_crf: int
    video_fps: int


@dataclass
class CarlaCollectorCfg:
    worker: WorkerCfg
    ipc: IpcCfg
    simulation: SimulationCfg
    route: RouteCfg
    traffic: TrafficCfg
    weather: WeatherCfg
    ego: EgoCfg
    cameras: CamerasCfg
    lidar: LidarCfg
    collection: CollectionCfg
    collision: CollisionCfg
    output: OutputCfg


@dataclass
class DataVisDisplayCfg:
    scale: float
    play_fps: int
    window_name: str


@dataclass
class DataVisTrafficLightsCfg:
    nearest_count: int


@dataclass
class DataVisBBoxCfg:
    thickness: int
    max_distance_m: float
    draw_static: bool
    colors: Dict[str, List[int]]   # 语义 -> BGR；Dict 字段经 _from_dict 原样透传


@dataclass
class DataVisDepthCfg:
    max_display_m: float
    colormap: str


@dataclass
class DataVisOpticalFlowCfg:
    max_flow: float


@dataclass
class DataVisBevCfg:
    range_m: float
    size_px: int
    point_radius: int
    color_by: str
    bg: List[int]


@dataclass
class DataVisLidarCfg:
    max_points_draw: int


@dataclass
class DataVisCfg:
    scene_root: str
    display: DataVisDisplayCfg
    traffic_lights: DataVisTrafficLightsCfg
    bbox: DataVisBBoxCfg
    depth: DataVisDepthCfg
    optical_flow: DataVisOpticalFlowCfg
    bev: DataVisBevCfg
    lidar: DataVisLidarCfg


# ---------- model —— 网络结构参数（model/ 只读，不含默认值声明） ----------

@dataclass
class TargetPointEmbeddingCfg:
    coordinate_dim: int
    grid_height: int
    grid_width: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    vector_order: str
    vector_transform: str
    feature_channels: int
    conv1_kernel_size: List[int]
    conv1_stride: List[int]
    conv1_padding: List[int]
    conv2_kernel_size: List[int]
    conv2_stride: List[int]
    conv2_padding: List[int]
    downsample_kernel_size: List[int]
    downsample_stride: List[int]
    downsample_padding: List[int]
    output_height: int
    output_width: int
    goal_token_count: int
    hidden_dim: int
    flatten_order: str
    dtype: str


@dataclass
class ModelCfg:
    target_point_embedding: TargetPointEmbeddingCfg


@dataclass
class Config:
    carla_collector: CarlaCollectorCfg
    data_vis: DataVisCfg
    model: ModelCfg


# ---------- 由 dict 构造 ----------

def _from_dict(cls, data):
    """递归地把 dict 实例化为 dataclass；list[dataclass] 字段按元素类型展开。

    之所以手写而非用第三方(dacite)：避免给双解释器再引入依赖；结构固定、规模小。
    """
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        ftype = hints[f.name]
        value = data[f.name]  # 缺键即 KeyError，等价于「配置不完整」的硬失败
        origin = getattr(ftype, "__origin__", None)
        if origin in (list, List):
            (elem_type,) = ftype.__args__
            kwargs[f.name] = [_from_dict(elem_type, v) for v in value]
        else:
            kwargs[f.name] = _from_dict(ftype, value)
    return cls(**kwargs)


def build_config(raw):
    """由 yaml 解析出的原始 dict 构造强类型 Config。"""
    return _from_dict(Config, raw)


# ---------- 加载期校验（每处标注校验对象，规范 §7.2）----------

# 默认设计约定的视角集合与分辨率/FOV，用于「与 Design 默认一致性」软校验
EXPECTED_RIG_NAMES = {"front", "back", "left", "right", "front_left", "front_right"}


def validate_config(cfg):
    """对 Config 做一次性合法性校验，非法立即抛错（加载期拦截）。"""
    cc = cfg.carla_collector

    # 校验对象: ipc.arena_size_mb / slot_count —— 容量与帧槽须为正
    # slot_count 为废弃预留字段（BumpAllocator 不再按槽切分，max_frames_per_scene 改为跨段累计），
    # 故不再与 max_frames_per_scene 比较，仅保留为正的下限校验。
    assert cc.ipc.arena_size_mb > 0, "ipc.arena_size_mb 必须 > 0"
    assert cc.ipc.slot_count > 0, "ipc.slot_count 必须 > 0"

    # 校验对象: simulation.fixed_delta_seconds / warmup_ticks
    assert cc.simulation.fixed_delta_seconds > 0, "simulation.fixed_delta_seconds 必须 > 0"
    assert cc.simulation.warmup_ticks >= 0, "simulation.warmup_ticks 必须 >= 0"
    # 校验对象: simulation.map —— 仅支持 Opt 地图（规避静态车辆图层 API 问题）
    assert cc.simulation.map.endswith("_Opt"), "simulation.map 必须是 Opt 地图（以 _Opt 结尾）"

    # 校验对象: route 距离区间 —— max 必须严格大于 min 且均为正
    assert 0 < cc.route.min_distance_m < cc.route.max_distance_m, \
        "route 需满足 0 < min_distance_m < max_distance_m"
    assert cc.route.max_scenes >= 0, "route.max_scenes 必须 >= 0（0 表示全部）"

    # 校验对象: traffic 数量与奔跑比例
    assert cc.traffic.num_vehicles >= 0 and cc.traffic.num_walkers >= 0, \
        "traffic.num_vehicles / num_walkers 必须 >= 0"
    assert 0.0 <= cc.traffic.walker_running_pct <= 1.0, \
        "traffic.walker_running_pct 必须在 [0,1]"
    # 校验对象: traffic.walker_arrival_radius_m —— 到达判定半径必须为正
    assert cc.traffic.walker_arrival_radius_m > 0, "traffic.walker_arrival_radius_m 必须 > 0"

    # 校验对象: ego.behavior —— 取值受 BehaviorAgent 支持集合限制
    assert cc.ego.behavior in ("cautious", "normal", "aggressive"), \
        "ego.behavior 仅支持 cautious/normal/aggressive"

    # 校验对象: cameras —— 分辨率为正，rig 非空，各相机 FOV 合法且名称唯一
    assert cc.cameras.width > 0 and cc.cameras.height > 0, "cameras.width/height 必须 > 0"
    assert len(cc.cameras.rig) > 0, "cameras.rig 至少需要一个相机"
    assert all(0 < camera.fov < 180 for camera in cc.cameras.rig), \
        "cameras.rig 中每个相机的 fov 必须在 (0,180)"
    rig_names = [c.name for c in cc.cameras.rig]
    assert len(rig_names) == len(set(rig_names)), "cameras.rig 相机 name 必须唯一"
    # 校验对象: cameras.modalities + lidar.enabled —— 至少启用一种传感器，否则一帧无任何数据可采
    mods = cc.cameras.modalities
    assert any((mods.rgb, mods.depth, mods.semantic, mods.optical_flow, cc.lidar.enabled)), \
        "至少需启用一种传感器（cameras.modalities.* 或 lidar.enabled）"

    # 校验对象: lidar —— 关键参数为正、FOV 上沿高于下沿
    assert cc.lidar.channels > 0 and cc.lidar.points_per_second > 0, \
        "lidar.channels / points_per_second 必须 > 0"
    assert cc.lidar.range_m > 0, "lidar.range_m 必须 > 0"
    assert cc.lidar.upper_fov > cc.lidar.lower_fov, "lidar.upper_fov 必须 > lower_fov"

    # 校验对象: collection —— 帧数上限与采样间隔为正
    assert cc.collection.max_frames_per_scene > 0, "collection.max_frames_per_scene 必须 > 0"
    assert cc.collection.capture_every_n_ticks >= 1, "collection.capture_every_n_ticks 必须 >= 1"

    # 校验对象: collision.max_retries_per_route
    assert cc.collision.max_retries_per_route >= 0, "collision.max_retries_per_route 必须 >= 0"

    # 校验对象: output —— LMDB 容量、CRF、帧率为正
    assert cc.output.lmdb_map_size_gb > 0, "output.lmdb_map_size_gb 必须 > 0"
    assert 0 <= cc.output.video_crf <= 51, "output.video_crf 必须在 [0,51]"
    assert cc.output.video_fps > 0, "output.video_fps 必须 > 0"

    _validate_data_vis(cfg.data_vis)
    _validate_model(cfg.model)


# ---------- model 侧加载期校验（枚举与形状推导的单一来源，规范 §7.3）----------

# 目标点嵌入层受支持的枚举取值（须与 model/target_point_embedding.py 的分支一致）
_TPE_VECTOR_ORDERS = {"grid_minus_target", "target_minus_grid"}
_TPE_VECTOR_TRANSFORMS = {"symlog"}
_TPE_FLATTEN_ORDERS = {"channel_height_width"}
_TPE_DTYPES = {"float32"}


def _validate_model(model):
    """校验对象: cfg.model —— 网络结构参数。"""
    _validate_target_point_embedding(model.target_point_embedding)


def _validate_target_point_embedding(tpe):
    """校验对象: cfg.model.target_point_embedding —— 目标点嵌入层结构参数。"""
    # 校验对象: coordinate_dim —— 固定为 2（ego 平面 [x, y]）
    assert tpe.coordinate_dim == 2, "model.target_point_embedding.coordinate_dim 必须为 2"
    # 校验对象: 正整数尺寸字段
    for name in ("grid_height", "grid_width", "feature_channels",
                 "output_height", "output_width", "goal_token_count", "hidden_dim"):
        assert getattr(tpe, name) > 0, "model.target_point_embedding.{} 必须 > 0".format(name)
    # 校验对象: 栅格边界 —— min 必须严格小于 max
    assert tpe.x_min_m < tpe.x_max_m, "model.target_point_embedding 需满足 x_min_m < x_max_m"
    assert tpe.y_min_m < tpe.y_max_m, "model.target_point_embedding 需满足 y_min_m < y_max_m"
    # 校验对象: 枚举字段 —— 取值受实现分支支持集合限制
    assert tpe.vector_order in _TPE_VECTOR_ORDERS, \
        "model.target_point_embedding.vector_order 仅支持 {}".format(sorted(_TPE_VECTOR_ORDERS))
    assert tpe.vector_transform in _TPE_VECTOR_TRANSFORMS, \
        "model.target_point_embedding.vector_transform 仅支持 {}".format(sorted(_TPE_VECTOR_TRANSFORMS))
    assert tpe.flatten_order in _TPE_FLATTEN_ORDERS, \
        "model.target_point_embedding.flatten_order 仅支持 {}".format(sorted(_TPE_FLATTEN_ORDERS))
    assert tpe.dtype in _TPE_DTYPES, \
        "model.target_point_embedding.dtype 仅支持 {}".format(sorted(_TPE_DTYPES))
    # 校验对象: 卷积核/步长为正、padding 非负（均为 2 元 [H, W]）
    for name in ("conv1_kernel_size", "conv1_stride", "conv2_kernel_size",
                 "conv2_stride", "downsample_kernel_size", "downsample_stride"):
        _check_2d_int(getattr(tpe, name), "model.target_point_embedding." + name, allow_zero=False)
    for name in ("conv1_padding", "conv2_padding", "downsample_padding"):
        _check_2d_int(getattr(tpe, name), "model.target_point_embedding." + name, allow_zero=True)
    # 校验对象: 卷积链推导的输出空间尺寸须与 output_height/width 一致（避免展平维度错配）
    shape = (tpe.grid_height, tpe.grid_width)
    shape = _conv2d_out(shape, tpe.conv1_kernel_size, tpe.conv1_stride, tpe.conv1_padding)
    shape = _conv2d_out(shape, tpe.conv2_kernel_size, tpe.conv2_stride, tpe.conv2_padding)
    shape = _conv2d_out(shape, tpe.downsample_kernel_size, tpe.downsample_stride, tpe.downsample_padding)
    assert shape == (tpe.output_height, tpe.output_width), \
        "model.target_point_embedding 卷积链推导尺寸 {} 与 output_height/width {} 不一致".format(
            shape, (tpe.output_height, tpe.output_width))


def _check_2d_int(values, name, allow_zero):
    """校验对象: 2 元整数配置项（卷积核/步长/padding）—— 长度、类型与下限。"""
    assert len(values) == 2, "{} 必须为 2 元列表，实际为 {}".format(name, values)
    lower_ok = (lambda v: v >= 0) if allow_zero else (lambda v: v > 0)
    assert all(isinstance(v, int) and not isinstance(v, bool) and lower_ok(v) for v in values), \
        "{} 每项必须为{}整数".format(name, "非负" if allow_zero else "正")


def _conv2d_out(shape, kernel, stride, padding):
    """由输入尺寸与卷积参数推导输出 [H, W]；非正即判为非法配置。"""
    h = (shape[0] + 2 * padding[0] - kernel[0]) // stride[0] + 1
    w = (shape[1] + 2 * padding[1] - kernel[1]) // stride[1] + 1
    assert h > 0 and w > 0, "卷积配置产生非正输出尺寸: {}".format((h, w))
    return (h, w)


# data_vis 着色支持的 OpenCV colormap 名（须与 vis/data_vis/draw.py 的映射表一致）
_DATA_VIS_COLORMAPS = {"turbo", "jet", "magma", "viridis", "plasma", "inferno"}


def _validate_data_vis(dv):
    """校验对象: cfg.data_vis —— 数据可视化样式参数（尺寸/量程/枚举/颜色三元组）。"""
    # 校验对象: data_vis.display —— 缩放与播放帧率为正
    assert dv.display.scale > 0, "data_vis.display.scale 必须 > 0"
    assert dv.display.play_fps > 0, "data_vis.display.play_fps 必须 > 0"

    # 校验对象: data_vis.traffic_lights.nearest_count —— 0 表示仅显示状态统计
    assert dv.traffic_lights.nearest_count >= 0, "data_vis.traffic_lights.nearest_count 必须 >= 0"

    # 校验对象: data_vis.bbox —— 线宽/距离为正，颜色为合法 BGR 三元组
    assert dv.bbox.thickness >= 1, "data_vis.bbox.thickness 必须 >= 1"
    assert dv.bbox.max_distance_m > 0, "data_vis.bbox.max_distance_m 必须 > 0"
    assert all(_is_bgr(c) for c in dv.bbox.colors.values()), \
        "data_vis.bbox.colors 每个值须是 0..255 的 BGR 三元组"

    # 校验对象: data_vis.depth —— 量程为正、colormap 受支持
    assert dv.depth.max_display_m > 0, "data_vis.depth.max_display_m 必须 > 0"
    assert dv.depth.colormap in _DATA_VIS_COLORMAPS, \
        "data_vis.depth.colormap 须取值 {}".format(sorted(_DATA_VIS_COLORMAPS))

    # 校验对象: data_vis.optical_flow.max_flow —— 满亮度幅值阈值；0 表示按帧自适应
    assert dv.optical_flow.max_flow >= 0, "data_vis.optical_flow.max_flow 必须 >= 0（0=自适应）"

    # 校验对象: data_vis.bev —— 尺寸/半径/着色方式/背景色
    assert dv.bev.range_m > 0, "data_vis.bev.range_m 必须 > 0"
    assert dv.bev.size_px > 0, "data_vis.bev.size_px 必须 > 0"
    assert dv.bev.point_radius >= 1, "data_vis.bev.point_radius 必须 >= 1"
    assert dv.bev.color_by in ("tag", "height"), "data_vis.bev.color_by 仅支持 tag/height"
    assert _is_bgr(dv.bev.bg), "data_vis.bev.bg 须是 0..255 的 BGR 三元组"

    # 校验对象: data_vis.lidar.max_points_draw —— BEV 绘制点数上限为正
    assert dv.lidar.max_points_draw > 0, "data_vis.lidar.max_points_draw 必须 > 0"


def _is_bgr(c):
    """是否为合法 BGR 颜色：长度 3、各分量 0..255。"""
    return len(c) == 3 and all(0 <= v <= 255 for v in c)
