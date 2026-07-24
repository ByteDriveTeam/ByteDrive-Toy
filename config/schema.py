"""配置的类型定义与加载期校验（参数约束的唯一来源）。

模块: config/schema.py
依赖: dataclasses, math, typing
读取配置: —（本文件定义配置结构本身，不读取具体键）
对外接口:
    - build_config(raw: dict) -> Config        # 由原始 dict 构造强类型配置对象
    - validate_config(cfg: Config) -> None     # 加载期一次性校验，非法即抛 AssertionError
说明: 用 dataclass 而非 pydantic，保证 Py3.7 与 Py3.12 双解释器均可导入（worker 侧虽不读
      config 文件，但两端共享同一份 schema 以理解下发的配置结构）。校验集中于此，运行期
      实现文件不再重复（规范 §7.3）。
"""

import math
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

    这是驾驶系统 BEV 几何的**单一来源**：纯几何查询嵌入、frustum 视场掩码、
    数据侧场 GT 栅格化都从这里取量程，避免多处各写一份坐标范围。
    """
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height: int          # Hb：BEV 工作网格前向(x)格数（= 初始查询分辨率）
    width: int           # Wb：BEV 工作网格左右(y)格数
    fov_deg: float       # 前向视场角（视场内监督）
    z_min_m: float       # BEV 查询几何嵌入垂直采样下界
    z_max_m: float
    z_step_m: float


@dataclass
class QueryEmbeddingCfg:
    """初始 BEV 纯几何查询嵌入参数（几何取自 BevGeometryCfg）。"""
    coord_symlog_scale: float   # symlog(坐标)·scale 归一到[-1,1]
    mlp_hidden: int             # 逐 cell 列 MLP 隐藏维


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
    temporal_layers: int           # 当前 BEV 查询 ← 上一帧 BEV 交叉注意力层数
    detach_previous: bool          # 是否截断上一帧 BEV 分支梯度
    transformer_layers: int        # BEV Pre-Norm Transformer 层数
    num_register_tokens: int       # BEV 自有无位置寄存器 Token 数
    register_init_std: float       # 寄存器正态初始化标准差
    rope_theta: float              # BEV Patch 二维 RoPE 基频


@dataclass
class FieldsCfg:
    reduce_channels: int           # 上采样前 1×1 压缩到的通道
    up_channels: List[int]         # 各级 2× 像素洗牌输出通道（Hb·2^L = 场分辨率）
    feature_channels: int          # 共享上采样输出特征通道（再由各场 1×1 头解码）


@dataclass
class LaneMapCfg:
    class_names: List[str]         # 0 固定为背景，其余为道路线类别
    reduce_channels: int
    up_channels: List[int]
    feature_channels: int


@dataclass
class TrafficControlCfg:
    state_names: List[str]          # 动态灯色类别；停止线几何由独立二值头预测


@dataclass
class TrajectoryCfg:
    num_modes: int                 # 可学习 Mode Token 数 = 多模态轨迹条数
    num_waypoints: int             # 每条轨迹航点数 T_wp
    planning_dim: int              # 规划分支工作维
    condition_mlp_hidden: int      # 目标点+ego 速度条件编码 MLP 隐藏维
    feature_ffn_hidden: int        # path2 感知特征适配 FFN 隐藏维
    cross_layers: int              # 规划 CTB 数（固定对应第 3、6 层特征）
    self_layers: int               # 规划 CTB 后的 TB 层数
    num_heads: int
    mode_token_init_std: float     # 可学习 Mode Token 随机初始化标准差
    baseline_step_m: float         # 扇区中线基线轨迹航点间距（米）
    symlog_scale: float            # 条件输入（目标点/ego 速度）归一化的 Symlog 缩放（轨迹在物理空间预测）


@dataclass
class BehaviorCfg:
    num_classes: int               # 固定语义顺序的多标签行为类别数


@dataclass
class DrivingCfg:
    """驾驶系统（双帧时序 BEV → 三场 + 独立道路线图 + 多模态轨迹/行为）网络参数。"""
    work_dim: int                  # 工作维 D（neck 融合输出、注意力、BEV 全程）
    freeze_perception: bool        # 是否冻结感知主干（复用其预训练表征）
    neck_num_residual_blocks: int  # driving_neck 融合后 2D 残差块层数
    bev: BevGeometryCfg
    query: QueryEmbeddingCfg
    frustum: FrustumCfg
    attention: DrivingAttentionCfg
    bev_encoder: BevEncoderCfg
    fields: FieldsCfg
    lane_map: LaneMapCfg
    traffic_control: TrafficControlCfg
    trajectory: TrajectoryCfg
    behavior: BehaviorCfg


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
    num_layers: int
    num_heads: int
    mlp_ratio: int
    rope_theta: float


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
class BehaviorTargetCfg:
    stationary_speed_mps: float
    acceleration_threshold_mps2: float
    turn_angle_deg: float
    traffic_light_semantic_tag: int
    traffic_light_match_radius_m: float
    traffic_light_seg_margin_px: int
    traffic_light_min_pixels: int


@dataclass
class LaneMapTargetCfg:
    line_width_m: float
    type_to_class: Dict[str, int]
    unknown_class: int


@dataclass
class TrafficControlTargetCfg:
    route_corridor_m: float         # 路线与停止区相关性判定的走廊半宽
    line_expand_m: float            # 栅格化停止区的额外膨胀半径
    actor_match_radius_m: float     # 地图 ParentActor 与场景交通灯 actor 匹配半径
    stop_margin_m: float            # 红灯轨迹停止安全余量
    reaction_time_s: float          # 可停车性判定反应时间
    comfortable_decel_mps2: float   # 可停车性判定舒适减速度


@dataclass
class DrivingDatasetCfg:
    """驾驶数据集参数（几何/K/场分辨率取自 model.driving，避免重复声明）。"""
    scene_root: str
    camera: str
    map_dir: str                  # HD 地图目录
    map_name_template: str        # 地图文件名模板，如 "{map}_HD_map.npz"
    previous_frame_offset: int    # 时序融合采用的同场景历史帧偏移
    dist_sigma_m: float           # 轨迹分布场高斯软标签标准差（米）
    lane_half_width_m: float      # 车道中心线缓冲半宽（栅格可行驶区域用）
    lane_map: LaneMapTargetCfg
    traffic_control: TrafficControlTargetCfg
    box_min_visible_pixels: int   # box 可见性：反投影后落入 3D 框的最少深度像素数
    target_min_m: float           # 目标点采样距离窗口下界（沿未来轨迹搜近端引导点）
    target_max_m: float           # 目标点采样距离窗口上界
    behavior: BehaviorTargetCfg


@dataclass
class DataCfg:
    scene_cache_size: int
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
    trajectory: float             # 匈牙利匹配多模态轨迹回归
    trajectory_unmatched_weight: float  # 未匹配 Mode 的小权重回归系数
    confidence: float             # 模态置信度分类
    behavior: float               # 行为多标签分类
    distribution: float           # 轨迹分布场
    risk: float                   # 风险场
    drivable: float               # 可行驶区域场
    lane_class: float             # 道路线类别
    lane_class_weights: List[float]  # 类别 CE 权重（顺序对齐 model.driving.lane_map.class_names）
    lane_direction: float         # 道路线有向切向量
    boundary: float               # 轨迹到可行驶区域的越界距离（道路外/可见占用内）
    stop_line: float              # 相关交通灯停止线几何
    traffic_light_state: float    # 停止线区域灯色分类
    stop_crossing: float          # 红灯轨迹越线距离


@dataclass
class TrainCfg:
    device: str
    epochs: int
    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool
    pin_memory: bool
    persistent_workers: bool
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
class LaneMapVisCfg:
    class_colors: List[List[int]]
    arrow_color: List[int]
    arrow_stride_px: int
    arrow_length_px: int
    arrow_thickness: int
    arrow_tip_ratio: float


@dataclass
class TrafficControlVisCfg:
    state_colors: List[List[int]]
    unknown_color: List[int]
    line_threshold: float
    overlay_alpha: float


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
    lane_map: LaneMapVisCfg
    traffic_control: TrafficControlVisCfg


@dataclass
class CloneWorkerCfg:
    python_exe: str
    carla_host: str
    carla_port: int
    startup_timeout_s: float
    command_timeout_s: float


@dataclass
class CloneIpcCfg:
    frame_name: str


@dataclass
class CloneSimulationCfg:
    map: str
    fixed_delta_seconds: float
    warmup_ticks: int
    max_steps: int
    base_seed: int
    random_weather: bool


@dataclass
class CloneRouteCfg:
    min_distance_m: float
    max_distance_m: float
    max_episodes: int
    queue_seed: int
    sampling_resolution_m: float
    target_distance_m: float
    completion_distance_m: float
    progress_search_points: int


@dataclass
class CloneTrafficCfg:
    num_vehicles: int
    vehicle_filter: str
    tm_port: int


@dataclass
class CloneEgoCfg:
    vehicle_filter: str


@dataclass
class CloneCameraCfg:
    width: int
    height: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class CloneInferenceCfg:
    checkpoint: str
    device: str
    min_weight_coverage: float
    confidence_weight: float
    risk_weight: float
    drivable_weight: float
    route_alignment_weight: float
    max_abs_waypoint_m: float


@dataclass
class CloneControlCfg:
    waypoint_dt_s: float
    speed_horizon: int
    min_target_speed_mps: float
    max_target_speed_mps: float
    lookahead_m: float
    wheelbase_m: float
    max_steer_angle_deg: float
    turn_steer_gain: float
    steer_smoothing: float
    longitudinal_kp: float
    longitudinal_ki: float
    longitudinal_kd: float
    integral_limit: float
    max_throttle: float
    max_brake: float
    brake_deadband_mps: float
    behavior_stop_threshold: float
    behavior_stop_indices: List[int]


@dataclass
class CloneSafetyCfg:
    max_route_deviation_m: float
    stuck_speed_mps: float
    stuck_steps: int


@dataclass
class CloneRecordingCfg:
    enabled: bool
    codec: str
    crf: int
    tile_size_px: int


@dataclass
class CloneOutputCfg:
    root: str
    log_every: int


@dataclass
class CloneLoopCfg:
    worker: CloneWorkerCfg
    ipc: CloneIpcCfg
    simulation: CloneSimulationCfg
    route: CloneRouteCfg
    traffic: CloneTrafficCfg
    ego: CloneEgoCfg
    camera: CloneCameraCfg
    inference: CloneInferenceCfg
    control: CloneControlCfg
    safety: CloneSafetyCfg
    recording: CloneRecordingCfg
    output: CloneOutputCfg


@dataclass
class Config:
    carla_collector: CarlaCollectorCfg
    data_vis: DataVisCfg
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    pred_vis: PredVisCfg
    driving_vis: DrivingVisCfg
    clone_loop: CloneLoopCfg


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
    _validate_data(cfg.data, cfg.model.driving.lane_map)
    _validate_train(cfg.train, cfg.model.driving.lane_map)
    _validate_pred_vis(cfg.pred_vis)
    _validate_driving_vis(
        cfg.driving_vis, cfg.model.driving.lane_map, cfg.model.driving.traffic_control)
    _validate_clone_loop(cfg.clone_loop, cfg.model, cfg.data)


def _validate_clone_loop(cl, model, data):
    """校验对象: cfg.clone_loop —— CARLA 闭环运行、推理、控制与安全参数。"""
    assert cl.worker.carla_port > 0 and cl.worker.startup_timeout_s > 0 \
        and cl.worker.command_timeout_s > 0, \
        "clone_loop.worker 端口与超时必须为正"
    assert cl.ipc.frame_name, "clone_loop.ipc.frame_name 不得为空"
    sim = cl.simulation
    assert sim.map.endswith("_Opt"), "clone_loop.simulation.map 必须是 Opt 地图"
    assert sim.fixed_delta_seconds > 0 and sim.warmup_ticks >= 0 and sim.max_steps > 0, \
        "clone_loop.simulation 固定步长/预热/步数取值非法"
    route = cl.route
    assert 0 < route.min_distance_m < route.max_distance_m, \
        "clone_loop.route 需满足 0 < min_distance_m < max_distance_m"
    assert route.max_episodes >= 0 and route.sampling_resolution_m > 0 \
        and route.target_distance_m > 0 and route.completion_distance_m > 0 \
        and route.progress_search_points > 0, "clone_loop.route 参数取值非法"
    assert cl.traffic.num_vehicles >= 0 and cl.traffic.tm_port > 0, \
        "clone_loop.traffic 车辆数必须非负且 TM 端口为正"
    cam = cl.camera
    assert cam.width > 0 and cam.height > 0, \
        "clone_loop.camera 分辨率取值非法"
    assert cam.width % model.dinov3_backbone.patch_size == 0 \
        and cam.height % model.dinov3_backbone.patch_size == 0, \
        "clone_loop.camera 宽高必须被模型 patch_size 整除"
    inf = cl.inference
    assert 0 < inf.min_weight_coverage <= 1 and inf.max_abs_waypoint_m > 0, \
        "clone_loop.inference 权重覆盖率/航点范围取值非法"
    assert all(math.isfinite(value) and value >= 0 for value in (
        inf.confidence_weight, inf.risk_weight, inf.drivable_weight,
        inf.route_alignment_weight)), "clone_loop.inference 各评分权重必须为有限非负数"
    control = cl.control
    assert control.waypoint_dt_s > 0 and control.speed_horizon > 0 \
        and 0 <= control.min_target_speed_mps < control.max_target_speed_mps, \
        "clone_loop.control 航点时间/速度参数取值非法"
    history_steps = control.waypoint_dt_s / sim.fixed_delta_seconds
    assert history_steps >= 1 and abs(history_steps - round(history_steps)) < 1e-6, \
        "clone_loop.control.waypoint_dt_s 必须是仿真固定步长的正整数倍"
    assert control.lookahead_m > 0 and control.wheelbase_m > 0 \
        and 0 < control.max_steer_angle_deg < 90 \
        and 0 < control.turn_steer_gain <= 1, \
        "clone_loop.control 横向控制参数取值非法"
    assert 0 <= control.steer_smoothing < 1 and control.integral_limit >= 0 \
        and 0 < control.max_throttle <= 1 and 0 < control.max_brake <= 1 \
        and control.brake_deadband_mps >= 0, "clone_loop.control PID/执行器参数取值非法"
    assert 0 < control.behavior_stop_threshold < 1 \
        and control.behavior_stop_indices \
        and all(0 <= index < model.driving.behavior.num_classes
                for index in control.behavior_stop_indices), \
        "clone_loop.control 停车行为阈值/索引取值非法"
    safety = cl.safety
    assert safety.max_route_deviation_m > 0 and safety.stuck_speed_mps >= 0 \
        and safety.stuck_steps > 0, "clone_loop.safety 参数取值非法"
    recording = cl.recording
    assert isinstance(recording.enabled, bool), "clone_loop.recording.enabled 必须为布尔值"
    assert recording.codec and 0 <= recording.crf <= 51, \
        "clone_loop.recording.codec 不得为空且 crf 必须在 [0,51]"
    assert recording.tile_size_px > 0 and recording.tile_size_px % 2 == 0, \
        "clone_loop.recording.tile_size_px 必须为正偶数"
    assert cl.output.log_every > 0, "clone_loop.output.log_every 必须 > 0"
    assert len(data.dataset.dino_mean) == 3 and len(data.dataset.dino_std) == 3, \
        "闭环 RGB 归一化要求 data.dataset.dino_mean/std 为三通道"


# ---------- model 侧加载期校验（枚举与形状推导的单一来源，规范 §7.3）----------

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
    """校验对象: cfg.model.feature_trunk —— 完整 DINOv3 序列的 Pre-Norm Transformer 参数。"""
    assert ft.channels > 0, "model.feature_trunk.channels 必须 > 0"
    assert ft.num_layers == 3, "model.feature_trunk.num_layers 必须为 3"
    assert ft.num_heads > 0 and ft.channels % ft.num_heads == 0, \
        "model.feature_trunk.num_heads 必须为正且整除 channels"
    assert (ft.channels // ft.num_heads) % 4 == 0, \
        "model.feature_trunk 每头维度必须被 4 整除（二维 RoPE）"
    assert ft.mlp_ratio > 0 and (ft.channels * ft.mlp_ratio) % 2 == 0, \
        "model.feature_trunk.mlp_ratio 必须 > 0 且 channels·mlp_ratio 为偶数"
    assert math.isfinite(ft.rope_theta) and ft.rope_theta > 0, \
        "model.feature_trunk.rope_theta 必须为有限正数"


def _validate_heads(hd):
    """校验对象: cfg.model.heads —— 三头解码结构参数。"""
    assert hd.reduce_channels > 0, "model.heads.reduce_channels 必须 > 0"
    assert len(hd.up_channels) > 0, "model.heads.up_channels 至少一级"
    # 每级像素洗牌前 1×1 Conv 升到 C_out·4，PixelShuffle(2) 折回 C_out：C_out 须为正整数即可
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


def _validate_data(data, model_lane):
    """校验对象: cfg.data —— 数据加载参数。"""
    assert data.scene_cache_size > 0, "data.scene_cache_size 必须 > 0"
    ds = data.dataset
    assert len(ds.dino_mean) == 3 and len(ds.dino_std) == 3, \
        "data.dataset.dino_mean/std 必须为 3 通道"
    assert all(s > 0 for s in ds.dino_std), "data.dataset.dino_std 每通道必须 > 0"
    # 校验对象: data.driving —— 高斯软标签/车道半宽为正，地图模板含 {map} 占位
    dr = data.driving
    assert dr.previous_frame_offset > 0, "data.driving.previous_frame_offset 必须 > 0"
    assert dr.dist_sigma_m > 0 and dr.lane_half_width_m > 0, \
        "data.driving.dist_sigma_m / lane_half_width_m 必须 > 0"
    lane = dr.lane_map
    assert lane.line_width_m > 0, "data.driving.lane_map.line_width_m 必须 > 0"
    class_ids = list(lane.type_to_class.values()) + [lane.unknown_class]
    assert lane.type_to_class and all(isinstance(i, int) and 0 < i < len(model_lane.class_names)
                                      for i in class_ids), \
        "data.driving.lane_map 类别索引须落在 model.driving.lane_map.class_names 的非背景范围"
    traffic = dr.traffic_control
    assert traffic.route_corridor_m > 0 and traffic.line_expand_m >= 0 \
        and traffic.actor_match_radius_m > 0 and traffic.stop_margin_m >= 0 \
        and traffic.reaction_time_s >= 0 and traffic.comfortable_decel_mps2 > 0, \
        "data.driving.traffic_control 走廊/膨胀/匹配/制动参数取值非法"
    assert dr.box_min_visible_pixels >= 10, \
        "data.driving.box_min_visible_pixels 必须 >= 10"
    assert "{map}" in dr.map_name_template, \
        "data.driving.map_name_template 必须含 {map} 占位（按场景地图名解析 HD 地图文件）"
    assert 0 < dr.target_min_m < dr.target_max_m, \
        "data.driving 需满足 0 < target_min_m < target_max_m（目标点采样窗口）"
    # 校验对象: data.driving.behavior —— 行为标签判定阈值与 Seg 语义参数
    bh = dr.behavior
    assert bh.stationary_speed_mps >= 0, \
        "data.driving.behavior.stationary_speed_mps 必须 >= 0"
    assert bh.acceleration_threshold_mps2 > 0, \
        "data.driving.behavior.acceleration_threshold_mps2 必须 > 0"
    assert 0 < bh.turn_angle_deg < 90, \
        "data.driving.behavior.turn_angle_deg 必须在 (0,90)"
    assert bh.traffic_light_semantic_tag >= 0, \
        "data.driving.behavior.traffic_light_semantic_tag 必须 >= 0"
    assert bh.traffic_light_match_radius_m > 0 and bh.traffic_light_seg_margin_px >= 0 \
        and bh.traffic_light_min_pixels > 0, \
        "data.driving.behavior 交通灯匹配半径/Seg 容差/最少像素数取值非法"


def _validate_train(train, model_lane):
    """校验对象: cfg.train —— 训练超参数。"""
    for name in ("epochs", "batch_size", "num_workers", "log_every"):
        assert getattr(train, name) >= (1 if name != "num_workers" else 0), \
            "train.{} 取值非法".format(name)
    assert train.lr > 0, "train.lr 必须 > 0"
    assert train.weight_decay >= 0, "train.weight_decay 必须 >= 0"
    assert train.grad_clip_norm >= 0, "train.grad_clip_norm 必须 >= 0（0 表示不裁剪）"
    assert train.perception_lr_scale > 0, "train.perception_lr_scale 必须 > 0（感知子模块相对 lr 缩放）"
    assert all(isinstance(getattr(train, name), bool) for name in
               ("shuffle", "drop_last", "pin_memory", "persistent_workers")), \
        "train.shuffle / drop_last / pin_memory / persistent_workers 必须为布尔值"
    # 校验对象: train.driving_loss_weights —— 各权重非负
    dw = train.driving_loss_weights
    assert all(getattr(dw, n) >= 0 for n in
               ("trajectory", "confidence", "behavior", "distribution", "risk", "drivable",
                "lane_class", "lane_direction", "boundary", "stop_line",
                "traffic_light_state", "stop_crossing")), \
        "train.driving_loss_weights.* 必须 >= 0"
    assert 0 < dw.trajectory_unmatched_weight <= 1, \
        "train.driving_loss_weights.trajectory_unmatched_weight 必须在 (0,1]"
    assert len(dw.lane_class_weights) == len(model_lane.class_names) \
        and all(weight > 0 for weight in dw.lane_class_weights), \
        "train.driving_loss_weights.lane_class_weights 须与道路线类别等长且各项 > 0"


def _validate_driving(dv):
    """校验对象: cfg.model.driving —— 驾驶系统网络参数。"""
    assert dv.work_dim >= 2 and dv.work_dim % 2 == 0, \
        "model.driving.work_dim 必须为 >=2 的偶数（残差块瓶颈需二分通道）"
    assert dv.neck_num_residual_blocks > 0, "model.driving.neck_num_residual_blocks 必须 > 0"
    _validate_bev_geometry(dv.bev)
    # 校验对象: query —— 纯几何编码的尺度与 MLP 隐藏维为正
    assert dv.query.coord_symlog_scale > 0 and dv.query.mlp_hidden > 0, \
        "model.driving.query.coord_symlog_scale / mlp_hidden 必须 > 0"
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
    # 校验对象: bev_encoder —— 固定六层、BEV 寄存器和二维 RoPE 参数合法
    be = dv.bev_encoder
    assert be.cross_layers > 0 and be.temporal_layers > 0, \
        "model.driving.bev_encoder.cross_layers / temporal_layers 必须 > 0"
    assert be.transformer_layers == 6, \
        "model.driving.bev_encoder.transformer_layers 必须为 6"
    assert be.num_register_tokens > 0, \
        "model.driving.bev_encoder.num_register_tokens 必须 > 0"
    assert math.isfinite(be.register_init_std) and be.register_init_std > 0, \
        "model.driving.bev_encoder.register_init_std 必须为有限正数"
    assert (dv.work_dim // dv.attention.num_heads) % 4 == 0, \
        "model.driving.bev_encoder 每头维度必须被 4 整除（二维 RoPE）"
    assert math.isfinite(be.rope_theta) and be.rope_theta > 0, \
        "model.driving.bev_encoder.rope_theta 必须为有限正数"
    # 校验对象: fields —— 通道为正、每级像素洗牌通道为正整数
    assert dv.fields.reduce_channels > 0 and dv.fields.feature_channels > 0, \
        "model.driving.fields.reduce_channels / feature_channels 必须 > 0"
    assert len(dv.fields.up_channels) > 0 and all(
        isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in dv.fields.up_channels), \
        "model.driving.fields.up_channels 每级必须为正整数"
    # 校验对象: lane_map —— 独立道路线图类别与上采样结构
    lane = dv.lane_map
    assert len(lane.class_names) >= 2 and lane.class_names[0] == "background", \
        "model.driving.lane_map.class_names 至少含背景+一道路线，且索引 0 必须为 background"
    assert len(lane.class_names) == len(set(lane.class_names)), \
        "model.driving.lane_map.class_names 不得重复"
    assert lane.reduce_channels > 0 and lane.feature_channels > 0, \
        "model.driving.lane_map.reduce_channels / feature_channels 必须 > 0"
    assert len(lane.up_channels) > 0 and all(
        isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in lane.up_channels), \
        "model.driving.lane_map.up_channels 每级必须为正整数"
    assert len(lane.up_channels) == len(dv.fields.up_channels), \
        "model.driving.lane_map 与 fields 上采样级数须一致，保证监督分辨率对齐"
    # 校验对象: traffic_control —— 灯色类别稳定且至少覆盖红灯越线监督所需的 red
    states = dv.traffic_control.state_names
    assert states and len(states) == len(set(states)) and "red" in states, \
        "model.driving.traffic_control.state_names 须非空、不重复且包含 red"
    # 校验对象: trajectory —— 固定 8 Mode、2 个规划 CTB、4 个后续 TB，其余维度与尺度合法
    tj = dv.trajectory
    assert tj.num_modes == 8, "model.driving.trajectory.num_modes 必须为 8"
    assert tj.cross_layers == 2, "model.driving.trajectory.cross_layers 必须为 2（对应第 3/6 层特征）"
    assert tj.self_layers == 4, "model.driving.trajectory.self_layers 必须为 4"
    for name in ("num_waypoints", "planning_dim", "condition_mlp_hidden", "feature_ffn_hidden"):
        assert getattr(tj, name) > 0, "model.driving.trajectory.{} 必须 > 0".format(name)
    assert tj.planning_dim < tj.cross_layers * dv.work_dim, \
        "model.driving.trajectory.planning_dim 必须小于 cross_layers×work_dim（拼接后 1×1 CNN 降维）"
    assert tj.num_heads > 0 and tj.planning_dim % tj.num_heads == 0, \
        "model.driving.trajectory.num_heads 必须 > 0 且整除 planning_dim"
    assert all(math.isfinite(value) and value > 0 for value in (
        tj.mode_token_init_std, tj.baseline_step_m, tj.symlog_scale)), \
        "model.driving.trajectory.mode_token_init_std / baseline_step_m / symlog_scale 必须为有限正数"
    # 校验对象: behavior.num_classes —— 固定对应八类行为语义，避免模型头与标签顺序漂移
    assert dv.behavior.num_classes == 8, \
        "model.driving.behavior.num_classes 必须为 8（固定行为语义顺序）"


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


def _validate_driving_vis(dv, model_lane, model_traffic):
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
    lane = dv.lane_map
    assert len(lane.class_colors) == len(model_lane.class_names) \
        and all(_is_bgr(color) for color in lane.class_colors), \
        "driving_vis.lane_map.class_colors 须与道路线类别等长，且每项为合法 BGR 颜色"
    assert _is_bgr(lane.arrow_color), \
        "driving_vis.lane_map.arrow_color 须为合法 BGR 颜色"
    assert lane.arrow_stride_px > 0 and lane.arrow_length_px > 0 and lane.arrow_thickness > 0, \
        "driving_vis.lane_map 箭头间距、长度和线宽必须 > 0"
    assert 0 < lane.arrow_tip_ratio <= 1, \
        "driving_vis.lane_map.arrow_tip_ratio 必须在 (0,1]"
    traffic = dv.traffic_control
    assert len(traffic.state_colors) == len(model_traffic.state_names) \
        and all(_is_bgr(color) for color in traffic.state_colors), \
        "driving_vis.traffic_control.state_colors 须与灯态类别等长，且每项为合法 BGR 颜色"
    assert _is_bgr(traffic.unknown_color), \
        "driving_vis.traffic_control.unknown_color 须为合法 BGR 颜色"
    assert 0 < traffic.line_threshold < 1, \
        "driving_vis.traffic_control.line_threshold 必须在 (0,1)"
    assert 0 < traffic.overlay_alpha <= 1, \
        "driving_vis.traffic_control.overlay_alpha 必须在 (0,1]"
