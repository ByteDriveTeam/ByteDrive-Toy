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
    stem_channels: int
    stem_kernel_size: List[int]
    stem_stride: List[int]
    stem_padding: List[int]
    num_residual_blocks: int
    output_channels: int
    output_height: int
    output_width: int


@dataclass
class DinoV3BackboneCfg:
    model_dir: str
    patch_size: int
    hidden_dim: int
    num_register_tokens: int


@dataclass
class TemporalTrunkCfg:
    in_channels: int
    channels: int
    num_blocks: int
    temporal_kernel: int
    spatial_kernel: int
    expansion: int


@dataclass
class HeadsCfg:
    reduce_channels: int
    up_channels: List[int]
    num_classes: int
    semantic_out: int
    flow_out: int
    depth_out: int


@dataclass
class PhysicsCfg:
    symlog_scale: float
    depth_max_m: float
    flow_dt_s: float
    flow_ndc_pixel_scale: List[float]
    semantic_ignore_index: int


@dataclass
class ModelCfg:
    dinov3_backbone: DinoV3BackboneCfg
    temporal_trunk: TemporalTrunkCfg
    heads: HeadsCfg
    physics: PhysicsCfg
    target_point_embedding: TargetPointEmbeddingCfg


@dataclass
class DatasetCfg:
    scene_root: str
    camera: str
    window_size: int
    window_stride: int
    dino_mean: List[float]
    dino_std: List[float]


@dataclass
class DataCfg:
    dataset: DatasetCfg


@dataclass
class LossWeightsCfg:
    semantic: float
    depth: float
    depth_range: float
    flow: float


@dataclass
class TrainCfg:
    device: str
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    log_every: int
    ckpt_dir: str
    resume: bool
    loss_weights: LossWeightsCfg


@dataclass
class PredVisCfg:
    checkpoint: str
    scene: str
    max_windows: int
    save_dir: str
    show_ground_truth: bool
    display_scale: float
    depth_colormap: str
    depth_max_display_m: float
    flow_max_display: float


@dataclass
class Config:
    carla_collector: CarlaCollectorCfg
    data_vis: DataVisCfg
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    pred_vis: PredVisCfg


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
    _validate_data(cfg.data)
    _validate_train(cfg.train)
    _validate_pred_vis(cfg.pred_vis)


# ---------- model 侧加载期校验（枚举与形状推导的单一来源，规范 §7.3）----------

# 目标点嵌入层受支持的枚举取值（须与 model/target_point_embedding.py 的分支一致）
_TPE_VECTOR_ORDERS = {"grid_minus_target", "target_minus_grid"}
_TPE_VECTOR_TRANSFORMS = {"symlog"}


def _validate_model(model):
    """校验对象: cfg.model —— 网络结构参数。"""
    _validate_dinov3_backbone(model.dinov3_backbone)
    _validate_temporal_trunk(model.temporal_trunk)
    _validate_heads(model.heads)
    _validate_physics(model.physics)
    _validate_target_point_embedding(model.target_point_embedding)


def _validate_dinov3_backbone(bb):
    """校验对象: cfg.model.dinov3_backbone —— 骨干结构参数。"""
    assert bb.patch_size > 0, "model.dinov3_backbone.patch_size 必须 > 0"
    assert bb.hidden_dim > 0, "model.dinov3_backbone.hidden_dim 必须 > 0"
    assert bb.num_register_tokens >= 0, "model.dinov3_backbone.num_register_tokens 必须 >= 0"


def _validate_temporal_trunk(tt):
    """校验对象: cfg.model.temporal_trunk —— 输入投影 + 3D ConvNeXt 主干参数。"""
    for name in ("in_channels", "channels", "num_blocks", "temporal_kernel", "spatial_kernel", "expansion"):
        assert getattr(tt, name) > 0, "model.temporal_trunk.{} 必须 > 0".format(name)
    # 深度可分离卷积须能对称 padding 保持时序/空间尺寸，故核为奇数
    assert tt.temporal_kernel % 2 == 1, "model.temporal_trunk.temporal_kernel 必须为奇数"
    assert tt.spatial_kernel % 2 == 1, "model.temporal_trunk.spatial_kernel 必须为奇数"
    # 逐点膨胀后须能整除回原通道（1×1×1 升维 channels·expansion 再降回 channels，天然成立），
    # 且膨胀通道须为正
    assert tt.channels * tt.expansion > 0, "model.temporal_trunk 膨胀通道必须 > 0"


def _validate_heads(hd):
    """校验对象: cfg.model.heads —— 三头解码结构参数。"""
    assert hd.reduce_channels > 0, "model.heads.reduce_channels 必须 > 0"
    assert len(hd.up_channels) > 0, "model.heads.up_channels 至少一级"
    # 每级像素洗牌前 Conv2d 升到 C_out·4，PixelShuffle(2) 折回 C_out：C_out 须为正整数即可
    assert all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in hd.up_channels), \
        "model.heads.up_channels 每级必须为正整数"
    for name in ("num_classes", "semantic_out", "flow_out", "depth_out"):
        assert getattr(hd, name) > 0, "model.heads.{} 必须 > 0".format(name)
    # 语义头输出通道即类别数，须一致（下游 CE 依赖）
    assert hd.semantic_out == hd.num_classes, \
        "model.heads.semantic_out 必须等于 num_classes"


def _validate_physics(ph):
    """校验对象: cfg.model.physics —— 物理量监督口径参数。"""
    assert ph.symlog_scale > 0, "model.physics.symlog_scale 必须 > 0"
    assert ph.depth_max_m > 0, "model.physics.depth_max_m 必须 > 0"
    assert ph.flow_dt_s > 0, "model.physics.flow_dt_s 必须 > 0"
    assert len(ph.flow_ndc_pixel_scale) == 2, \
        "model.physics.flow_ndc_pixel_scale 必须为 2 元 [sx, sy]"
    # semantic_ignore_index 允许为负（如 -100 表示不忽略任何类），故不校验符号


def _validate_data(data):
    """校验对象: cfg.data —— 数据加载参数。"""
    ds = data.dataset
    assert ds.window_size > 0, "data.dataset.window_size 必须 > 0"
    assert ds.window_stride > 0, "data.dataset.window_stride 必须 > 0"
    assert len(ds.dino_mean) == 3 and len(ds.dino_std) == 3, \
        "data.dataset.dino_mean/std 必须为 3 通道"
    assert all(s > 0 for s in ds.dino_std), "data.dataset.dino_std 每通道必须 > 0"


def _validate_train(train):
    """校验对象: cfg.train —— 训练超参数。"""
    for name in ("epochs", "batch_size", "num_workers", "log_every"):
        assert getattr(train, name) >= (1 if name != "num_workers" else 0), \
            "train.{} 取值非法".format(name)
    assert train.lr > 0, "train.lr 必须 > 0"
    assert train.weight_decay >= 0, "train.weight_decay 必须 >= 0"
    assert train.grad_clip_norm >= 0, "train.grad_clip_norm 必须 >= 0（0 表示不裁剪）"


def _validate_target_point_embedding(tpe):
    """校验对象: cfg.model.target_point_embedding —— 目标点嵌入层结构参数。"""
    # 校验对象: coordinate_dim —— 固定为 2（ego 平面 [x, y]）
    assert tpe.coordinate_dim == 2, "model.target_point_embedding.coordinate_dim 必须为 2"
    # 校验对象: 正整数尺寸字段
    for name in ("grid_height", "grid_width", "stem_channels", "num_residual_blocks",
                 "output_channels", "output_height", "output_width"):
        assert getattr(tpe, name) > 0, "model.target_point_embedding.{} 必须 > 0".format(name)
    # 校验对象: stem_channels —— 残差块瓶颈 mid=channels/2，须为偶数
    assert tpe.stem_channels % 2 == 0, "model.target_point_embedding.stem_channels 必须为偶数"
    # 校验对象: 栅格边界 —— min 必须严格小于 max
    assert tpe.x_min_m < tpe.x_max_m, "model.target_point_embedding 需满足 x_min_m < x_max_m"
    assert tpe.y_min_m < tpe.y_max_m, "model.target_point_embedding 需满足 y_min_m < y_max_m"
    # 校验对象: 枚举字段 —— 取值受实现分支支持集合限制
    assert tpe.vector_order in _TPE_VECTOR_ORDERS, \
        "model.target_point_embedding.vector_order 仅支持 {}".format(sorted(_TPE_VECTOR_ORDERS))
    assert tpe.vector_transform in _TPE_VECTOR_TRANSFORMS, \
        "model.target_point_embedding.vector_transform 仅支持 {}".format(sorted(_TPE_VECTOR_TRANSFORMS))
    # 校验对象: 降采样卷积核/步长为正、padding 非负（均为 2 元 [H, W]）
    _check_2d_int(tpe.stem_kernel_size, "model.target_point_embedding.stem_kernel_size", allow_zero=False)
    _check_2d_int(tpe.stem_stride, "model.target_point_embedding.stem_stride", allow_zero=False)
    _check_2d_int(tpe.stem_padding, "model.target_point_embedding.stem_padding", allow_zero=True)
    # 校验对象: 降采样卷积推导的输出空间尺寸须与 output_height/width 一致（残差块与 1×1 卷积保持尺寸不变）
    shape = _conv2d_out((tpe.grid_height, tpe.grid_width),
                        tpe.stem_kernel_size, tpe.stem_stride, tpe.stem_padding)
    assert shape == (tpe.output_height, tpe.output_width), \
        "model.target_point_embedding 降采样推导尺寸 {} 与 output_height/width {} 不一致".format(
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


def _validate_pred_vis(pv):
    """校验对象: cfg.pred_vis —— 感知模型预测可视化参数。"""
    assert pv.max_windows >= 0, "pred_vis.max_windows 必须 >= 0（0=全部）"
    assert pv.display_scale > 0, "pred_vis.display_scale 必须 > 0"
    assert pv.depth_colormap in _DATA_VIS_COLORMAPS, \
        "pred_vis.depth_colormap 须取值 {}".format(sorted(_DATA_VIS_COLORMAPS))
    assert pv.depth_max_display_m > 0, "pred_vis.depth_max_display_m 必须 > 0"
    assert pv.flow_max_display >= 0, "pred_vis.flow_max_display 必须 >= 0（0=自适应）"
