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
class BevGeometryCfg:
    """BEV（ego 前向单目）几何：坐标量程、工作网格分辨率、视场角、z 采样。

    这是驾驶系统 BEV 几何的**单一来源**：目标点嵌入（初始查询）、frustum 视场掩码、
    数据侧场 GT 栅格化都从这里取量程，避免多处各写一份坐标范围。
    """
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height: int          # Hb：BEV 工作网格前向(x)格数（= 初始查询分辨率）
    width: int           # Wb：BEV 工作网格左右(y)格数
    fov_deg: float       # 前向视场角（视场内监督）
    z_min_m: float       # 目标点嵌入垂直采样下界
    z_max_m: float
    z_step_m: float


@dataclass
class QueryEmbeddingCfg:
    """初始 BEV 查询嵌入（由 target_point_embedding 产）的模块参数（几何取自 BevGeometryCfg）。"""
    coord_symlog_scale: float   # symlog(坐标)·scale 归一到[-1,1]
    mlp_hidden: int             # 逐 cell 列 MLP 隐藏维
    vector_order: str           # 目标点相对向量方向


@dataclass
class FrustumCfg:
    """深度 frustum 位置编码参数（每 patch 中心+四角×深度采样的候选 3D 坐标）。"""
    depth_min_m: float
    depth_max_m: float
    step_near_m: float          # 近处深度采样步长
    step_far_m: float           # 远处深度采样步长（步长在量程内线性增长）
    coord_symlog_scale: float
    mlp_hidden: int


@dataclass
class DrivingAttentionCfg:
    num_heads: int
    mlp_ratio: int


@dataclass
class BevEncoderCfg:
    cross_layers: int              # BEV 查询 ← 图像特征 交叉注意力层数
    num_convnext_blocks: int
    convnext_spatial_kernel: int
    convnext_expansion: int


@dataclass
class FieldsCfg:
    reduce_channels: int           # 上采样前 1×1 压缩到的通道
    up_channels: List[int]         # 各级 2× 像素洗牌输出通道（Hb·2^L = 场分辨率）
    feature_channels: int          # 共享上采样输出特征通道（再由各场 1×1 头解码）


@dataclass
class TrajectoryCfg:
    num_modes: int                 # 前向扇区数 = 多模态轨迹条数
    num_waypoints: int             # 每条轨迹航点数 T_wp
    token_mlp_hidden: int          # 扇区 Token 输入编码 MLP 隐藏维
    cross_layers: int              # Token ← BEV 特征 交叉注意力层数
    self_layers: int               # Token 自注意力层数
    num_heads: int
    velocity_norm_mps: float       # 自车速度归一尺度（m/s）


@dataclass
class DrivingCfg:
    """驾驶系统（复用感知主干 → BEV → 三场 + 多模态轨迹）网络参数。"""
    work_dim: int                  # 工作维 D（neck 融合输出、注意力、BEV 全程）
    freeze_perception: bool        # 是否冻结感知主干（复用其预训练表征）
    neck_num_residual_blocks: int  # driving_neck 融合后 2D 残差块层数
    bev: BevGeometryCfg
    query: QueryEmbeddingCfg
    frustum: FrustumCfg
    attention: DrivingAttentionCfg
    bev_encoder: BevEncoderCfg
    fields: FieldsCfg
    trajectory: TrajectoryCfg


@dataclass
class DinoV3BackboneCfg:
    model_dir: str
    patch_size: int
    hidden_dim: int
    num_register_tokens: int
    feature_layers: List[int]


@dataclass
class FeatureTrunkCfg:
    channels: int
    num_blocks: int


@dataclass
class HeadsCfg:
    reduce_channels: int
    up_channels: List[int]
    num_classes: int
    semantic_out: int
    depth_out: int


@dataclass
class PhysicsCfg:
    symlog_scale: float
    depth_max_m: float
    semantic_ignore_index: int


@dataclass
class ModelCfg:
    dinov3_backbone: DinoV3BackboneCfg
    feature_trunk: FeatureTrunkCfg
    heads: HeadsCfg
    physics: PhysicsCfg
    driving: DrivingCfg


@dataclass
class DatasetCfg:
    scene_root: str
    camera: str
    dino_mean: List[float]
    dino_std: List[float]


@dataclass
class DrivingDatasetCfg:
    """驾驶数据集参数（几何/K/场分辨率取自 model.driving，避免重复声明）。"""
    scene_root: str
    camera: str
    map_dir: str                  # HD 地图目录
    map_name_template: str        # 地图文件名模板，如 "{map}_HD_map.npz"
    dist_sigma_m: float           # 轨迹分布场高斯软标签标准差（米）
    lane_half_width_m: float      # 车道中心线缓冲半宽（栅格可行驶区域用）


@dataclass
class DataCfg:
    dataset: DatasetCfg
    driving: DrivingDatasetCfg


@dataclass
class LossWeightsCfg:
    semantic: float
    depth: float
    depth_grad: float
    depth_range: float


@dataclass
class DrivingLossWeightsCfg:
    trajectory: float             # WTA 多模态轨迹回归
    confidence: float             # 模态置信度分类
    distribution: float           # 轨迹分布场
    risk: float                   # 风险场
    drivable: float               # 可行驶区域场


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
    perception_lr_scale: float     # 驾驶训练时感知子模块（融合+trunk+双头）相对 lr 的缩放（DINOv3 仍冻结）
    loss_weights: LossWeightsCfg
    driving_loss_weights: DrivingLossWeightsCfg


@dataclass
class PredVisCfg:
    checkpoint: str
    scene: str
    max_frames: int
    save_dir: str
    show_ground_truth: bool
    display_scale: float
    depth_colormap: str
    depth_max_display_m: float
    depth_min_display_m: float
    depth_log: bool


@dataclass
class DrivingVisCfg:
    checkpoint: str
    scene: str
    max_frames: int
    save_dir: str
    show_ground_truth: bool
    display_scale: float
    field_colormap: str            # 风险/可行驶/分布场热力图 colormap
    depth_colormap: str
    depth_max_display_m: float
    depth_min_display_m: float
    depth_log: bool


@dataclass
class Config:
    carla_collector: CarlaCollectorCfg
    data_vis: DataVisCfg
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    pred_vis: PredVisCfg
    driving_vis: DrivingVisCfg


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
    _validate_driving_vis(cfg.driving_vis)


# ---------- model 侧加载期校验（枚举与形状推导的单一来源，规范 §7.3）----------

# 目标点相对向量方向枚举（须与 model/target_point_embedding.py 的分支一致）
_TPE_VECTOR_ORDERS = {"grid_minus_target", "target_minus_grid"}


def _validate_model(model):
    """校验对象: cfg.model —— 网络结构参数。"""
    _validate_dinov3_backbone(model.dinov3_backbone)
    _validate_feature_trunk(model.feature_trunk)
    _validate_heads(model.heads)
    _validate_physics(model.physics)
    _validate_driving(model.driving)


def _validate_dinov3_backbone(bb):
    """校验对象: cfg.model.dinov3_backbone —— 骨干结构参数。"""
    assert bb.patch_size > 0, "model.dinov3_backbone.patch_size 必须 > 0"
    assert bb.hidden_dim > 0, "model.dinov3_backbone.hidden_dim 必须 > 0"
    assert bb.num_register_tokens >= 0, "model.dinov3_backbone.num_register_tokens 必须 >= 0"
    # 融合层索引取自 output_hidden_states（含 embedding 于索引 0），须非空且为非负整数
    assert len(bb.feature_layers) > 0, "model.dinov3_backbone.feature_layers 至少选一层"
    assert all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in bb.feature_layers), \
        "model.dinov3_backbone.feature_layers 每项必须为非负整数（hidden_states 索引）"


def _validate_feature_trunk(ft):
    """校验对象: cfg.model.feature_trunk —— 2D 残差块单帧主干参数（输入维已由 feature_fusion 对齐）。"""
    # 2D 瓶颈残差块 C→C/2→C 需二分通道，故 channels 至少为 2
    assert ft.channels >= 2, "model.feature_trunk.channels 必须 >= 2（瓶颈残差块需二分通道）"
    assert ft.num_blocks > 0, "model.feature_trunk.num_blocks 必须 > 0"


def _validate_heads(hd):
    """校验对象: cfg.model.heads —— 三头解码结构参数。"""
    assert hd.reduce_channels > 0, "model.heads.reduce_channels 必须 > 0"
    assert len(hd.up_channels) > 0, "model.heads.up_channels 至少一级"
    # 每级像素洗牌前 Conv2d 升到 C_out·4，PixelShuffle(2) 折回 C_out：C_out 须为正整数即可
    assert all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in hd.up_channels), \
        "model.heads.up_channels 每级必须为正整数"
    for name in ("num_classes", "semantic_out", "depth_out"):
        assert getattr(hd, name) > 0, "model.heads.{} 必须 > 0".format(name)
    # 语义头输出通道即类别数，须一致（下游 CE 依赖）
    assert hd.semantic_out == hd.num_classes, \
        "model.heads.semantic_out 必须等于 num_classes"


def _validate_physics(ph):
    """校验对象: cfg.model.physics —— 物理量监督口径参数。"""
    assert ph.symlog_scale > 0, "model.physics.symlog_scale 必须 > 0"
    assert ph.depth_max_m > 0, "model.physics.depth_max_m 必须 > 0"
    # semantic_ignore_index 允许为负（如 -100 表示不忽略任何类），故不校验符号


def _validate_data(data):
    """校验对象: cfg.data —— 数据加载参数。"""
    ds = data.dataset
    assert len(ds.dino_mean) == 3 and len(ds.dino_std) == 3, \
        "data.dataset.dino_mean/std 必须为 3 通道"
    assert all(s > 0 for s in ds.dino_std), "data.dataset.dino_std 每通道必须 > 0"
    # 校验对象: data.driving —— 高斯软标签/车道半宽为正，地图模板含 {map} 占位
    dr = data.driving
    assert dr.dist_sigma_m > 0 and dr.lane_half_width_m > 0, \
        "data.driving.dist_sigma_m / lane_half_width_m 必须 > 0"
    assert "{map}" in dr.map_name_template, \
        "data.driving.map_name_template 必须含 {map} 占位（按场景地图名解析 HD 地图文件）"


def _validate_train(train):
    """校验对象: cfg.train —— 训练超参数。"""
    for name in ("epochs", "batch_size", "num_workers", "log_every"):
        assert getattr(train, name) >= (1 if name != "num_workers" else 0), \
            "train.{} 取值非法".format(name)
    assert train.lr > 0, "train.lr 必须 > 0"
    assert train.weight_decay >= 0, "train.weight_decay 必须 >= 0"
    assert train.grad_clip_norm >= 0, "train.grad_clip_norm 必须 >= 0（0 表示不裁剪）"
    assert train.perception_lr_scale > 0, "train.perception_lr_scale 必须 > 0（感知子模块相对 lr 缩放）"
    # 校验对象: train.driving_loss_weights —— 各权重非负
    dw = train.driving_loss_weights
    assert all(getattr(dw, n) >= 0 for n in
               ("trajectory", "confidence", "distribution", "risk", "drivable")), \
        "train.driving_loss_weights.* 必须 >= 0"


def _validate_driving(dv):
    """校验对象: cfg.model.driving —— 驾驶系统网络参数。"""
    assert dv.work_dim >= 2 and dv.work_dim % 2 == 0, \
        "model.driving.work_dim 必须为 >=2 的偶数（残差块瓶颈需二分通道）"
    assert dv.neck_num_residual_blocks > 0, "model.driving.neck_num_residual_blocks 必须 > 0"
    _validate_bev_geometry(dv.bev)
    # 校验对象: query —— 尺度为正、MLP 隐藏维为正、向量方向枚举合法
    assert dv.query.coord_symlog_scale > 0 and dv.query.mlp_hidden > 0, \
        "model.driving.query.coord_symlog_scale / mlp_hidden 必须 > 0"
    assert dv.query.vector_order in _TPE_VECTOR_ORDERS, \
        "model.driving.query.vector_order 仅支持 {}".format(sorted(_TPE_VECTOR_ORDERS))
    # 校验对象: frustum —— 深度量程与步长为正、近步长 <= 远步长
    fr = dv.frustum
    assert 0 < fr.depth_min_m < fr.depth_max_m, "model.driving.frustum 需 0 < depth_min_m < depth_max_m"
    assert 0 < fr.step_near_m <= fr.step_far_m, "model.driving.frustum 需 0 < step_near_m <= step_far_m"
    assert fr.coord_symlog_scale > 0 and fr.mlp_hidden > 0, \
        "model.driving.frustum.coord_symlog_scale / mlp_hidden 必须 > 0"
    # 校验对象: attention —— D 须能被头数整除
    assert dv.attention.num_heads > 0 and dv.work_dim % dv.attention.num_heads == 0, \
        "model.driving.attention.num_heads 必须 > 0 且整除 work_dim"
    assert dv.attention.mlp_ratio > 0, "model.driving.attention.mlp_ratio 必须 > 0"
    # 校验对象: bev_encoder —— 层数/核为正
    be = dv.bev_encoder
    assert be.cross_layers > 0 and be.num_convnext_blocks > 0, \
        "model.driving.bev_encoder.cross_layers / num_convnext_blocks 必须 > 0"
    assert be.convnext_spatial_kernel % 2 == 1 and be.convnext_spatial_kernel > 0, \
        "model.driving.bev_encoder.convnext_spatial_kernel 必须为正奇数"
    assert be.convnext_expansion > 0, "model.driving.bev_encoder.convnext_expansion 必须 > 0"
    # 校验对象: fields —— 通道为正、每级像素洗牌通道为正整数
    assert dv.fields.reduce_channels > 0 and dv.fields.feature_channels > 0, \
        "model.driving.fields.reduce_channels / feature_channels 必须 > 0"
    assert len(dv.fields.up_channels) > 0 and all(
        isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in dv.fields.up_channels), \
        "model.driving.fields.up_channels 每级必须为正整数"
    # 校验对象: trajectory —— 模态/航点/层数为正、D 须能被头数整除
    tj = dv.trajectory
    for name in ("num_modes", "num_waypoints", "token_mlp_hidden", "cross_layers", "self_layers"):
        assert getattr(tj, name) > 0, "model.driving.trajectory.{} 必须 > 0".format(name)
    assert tj.num_heads > 0 and dv.work_dim % tj.num_heads == 0, \
        "model.driving.trajectory.num_heads 必须 > 0 且整除 work_dim"
    assert tj.velocity_norm_mps > 0, "model.driving.trajectory.velocity_norm_mps 必须 > 0"


def _validate_bev_geometry(bev):
    """校验对象: cfg.model.driving.bev —— BEV 几何量程/分辨率/视场/z 采样。"""
    assert bev.x_min_m < bev.x_max_m, "model.driving.bev 需满足 x_min_m < x_max_m"
    assert bev.y_min_m < bev.y_max_m, "model.driving.bev 需满足 y_min_m < y_max_m"
    assert bev.height > 0 and bev.width > 0, "model.driving.bev.height/width 必须 > 0"
    assert 0 < bev.fov_deg < 180, "model.driving.bev.fov_deg 必须在 (0,180)"
    assert bev.z_min_m < bev.z_max_m and bev.z_step_m > 0, \
        "model.driving.bev 需满足 z_min_m < z_max_m 且 z_step_m > 0"


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
    assert pv.max_frames >= 0, "pred_vis.max_frames 必须 >= 0（0=全部）"
    assert pv.display_scale > 0, "pred_vis.display_scale 必须 > 0"
    assert pv.depth_colormap in _DATA_VIS_COLORMAPS, \
        "pred_vis.depth_colormap 须取值 {}".format(sorted(_DATA_VIS_COLORMAPS))
    assert pv.depth_max_display_m > 0, "pred_vis.depth_max_display_m 必须 > 0"
    assert 0 < pv.depth_min_display_m < pv.depth_max_display_m, \
        "pred_vis.depth_min_display_m 须 >0 且 < depth_max_display_m（对数量程下限）"


def _validate_driving_vis(dv):
    """校验对象: cfg.driving_vis —— 驾驶模型可视化参数。"""
    assert dv.max_frames >= 0, "driving_vis.max_frames 必须 >= 0（0=全部）"
    assert dv.display_scale > 0, "driving_vis.display_scale 必须 > 0"
    assert dv.field_colormap in _DATA_VIS_COLORMAPS, \
        "driving_vis.field_colormap 须取值 {}".format(sorted(_DATA_VIS_COLORMAPS))
    assert dv.depth_colormap in _DATA_VIS_COLORMAPS, \
        "driving_vis.depth_colormap 须取值 {}".format(sorted(_DATA_VIS_COLORMAPS))
    assert dv.depth_max_display_m > 0, "driving_vis.depth_max_display_m 必须 > 0"
    assert 0 < dv.depth_min_display_m < dv.depth_max_display_m, \
        "driving_vis.depth_min_display_m 须 >0 且 < depth_max_display_m（对数量程下限）"
