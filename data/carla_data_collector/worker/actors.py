"""主车、交通流与行人的生成与销毁。

模块: worker/actors.py
依赖: random, carla, agents.navigation.behavior_agent, worker.geometry, worker.actors_checks
读取配置: 由各函数接收标量（ego.behavior/vehicle_filter、traffic.*），自身不读 config
对外接口:
    - spawn_ego(world, vehicle_filter, behavior, start_pose6) -> (ego, agent)
    - spawn_traffic_vehicles(client, world, tm, num, vehicle_filter) -> list[int]
    - spawn_walkers(client, world, num, walker_filter, running_pct, arrival_radius) -> WalkerCrowd
    - WalkerCrowd: .retarget_arrived() / .walker_ids / .controller_ids
    - destroy_scene_actors(client, world, ego, vehicle_ids, walker_ids, controller_ids) -> None
说明: 主车挂 BehaviorAgent（Design ①）；交通流交 TrafficManager 自动驾驶；行人生成点取自
      world.get_random_location_from_navigation()，保证落在行人可行走范围（Design ②）。
      批量生成用 client.apply_batch_sync，比逐个 spawn 高效且更接近官方 generate_traffic 范式。
      行人由 WalkerCrowd 管理：到达目标后重设新导航点，使其全程持续漫游（否则到点即停）。
"""

import random

import carla

from agents.navigation.behavior_agent import BehaviorAgent
from worker.geometry import make_transform
from worker.actors_checks import check_blueprints

_SpawnActor = carla.command.SpawnActor
_SetAutopilot = carla.command.SetAutopilot
_FutureActor = carla.command.FutureActor
_DestroyActor = carla.command.DestroyActor


def spawn_ego(world, vehicle_filter, behavior, start_pose6):
    """在起点生成主车并挂 BehaviorAgent。返回 (ego, agent)。"""
    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    check_blueprints(blueprints, vehicle_filter)
    bp = blueprints[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(bp, make_transform(start_pose6))
    world.tick()  # 等主车落位后再交给 agent，避免初始位姿未就绪
    return ego, BehaviorAgent(ego, behavior=behavior)


def spawn_traffic_vehicles(client, world, tm, num, vehicle_filter):
    """用 TrafficManager 接管的自动驾驶车填充交通流，返回成功生成的车辆 id 列表。"""
    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    check_blueprints(blueprints, vehicle_filter)
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    def _build(transform):
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        return _SpawnActor(bp, transform).then(_SetAutopilot(_FutureActor, True, tm.get_port()))

    batch = [_build(t) for t in spawn_points[:num]]
    return [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]


def _walker_speed(bp, running_pct):
    """按奔跑比例选取行走/奔跑速度（recommended_values: [_, walk, run]）。"""
    if not bp.has_attribute("speed"):
        return 0.0
    vals = bp.get_attribute("speed").recommended_values
    walk = vals[1] if len(vals) > 1 else "1.4"
    run = vals[2] if len(vals) > 2 else walk
    return float(run if random.random() < running_pct else walk)


class WalkerCrowd:
    """管理行人 AI 控制器：记录每个行人当前目标，到达后重派新导航点使其持续漫游。

    walker 与其控制器成对保存（不依赖 controller.parent），到达判定用行人当前位置到目标的距离。
    """

    def __init__(self, world, arrival_radius):
        self._world = world
        self._arrival_radius = arrival_radius
        self._pairs = []          # [(walker_actor, controller_actor)]
        self._targets = {}        # controller.id -> carla.Location 当前目标
        self.walker_ids = []
        self.controller_ids = []

    def add(self, walker_id, controller_id):
        """登记一对 walker/controller 并下发其初始目标。"""
        walker = self._world.get_actor(walker_id)
        controller = self._world.get_actor(controller_id)
        self._pairs.append((walker, controller))
        self.walker_ids.append(walker_id)
        self.controller_ids.append(controller_id)
        self._retarget(controller)

    def _retarget(self, controller):
        """给一个控制器派一个新的导航网格随机点为目标。"""
        loc = self._world.get_random_location_from_navigation()
        if loc is not None:
            controller.go_to_location(loc)
            self._targets[controller.id] = loc

    def retarget_arrived(self):
        """对已到达（或尚无目标）的行人重派目标；在采集循环里逐 tick 调用。

        含副作用且需逐个读取行人位置、下发命令，无更清晰的向量化写法，故用 for（规范 §9）。
        """
        for walker, controller in self._pairs:
            target = self._targets.get(controller.id)
            if target is None or walker.get_location().distance(target) < self._arrival_radius:
                self._retarget(controller)


def spawn_walkers(client, world, num, walker_filter, running_pct, arrival_radius):
    """在导航网格上生成行人并绑定 AI 控制器，返回管理其漫游的 WalkerCrowd。"""
    blueprints = world.get_blueprint_library().filter(walker_filter)
    check_blueprints(blueprints, walker_filter)

    # 1) 生成点全部取自导航网格随机点 —— 保证行人落在可行走范围（Design ②）
    nav_points = [world.get_random_location_from_navigation() for _ in range(num)]
    spawns = [carla.Transform(loc) for loc in nav_points if loc is not None]

    # 2) 批量生成 walker，并记录各自速度
    chosen = [random.choice(blueprints) for _ in spawns]
    speeds = [_walker_speed(bp, running_pct) for bp in chosen]
    for bp in chosen:
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
    results = client.apply_batch_sync([_SpawnActor(bp, sp) for bp, sp in zip(chosen, spawns)], True)
    walkers = [(r.actor_id, s) for r, s in zip(results, speeds) if not r.error]

    # 3) 批量生成 AI 控制器；保留 walker 与控制器均成功的配对
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    ctrl_results = client.apply_batch_sync(
        [_SpawnActor(controller_bp, carla.Transform(), wid) for wid, _ in walkers], True)
    pairs = [(wid, r.actor_id, s) for (wid, s), r in zip(walkers, ctrl_results) if not r.error]
    world.tick()  # 控制器就位后再启动

    # 4) 启动控制器、设速度，并交由 WalkerCrowd 下发/维护目标
    crowd = WalkerCrowd(world, arrival_radius)
    for walker_id, controller_id, speed in pairs:
        controller = world.get_actor(controller_id)
        controller.start()
        controller.set_max_speed(speed)
        crowd.add(walker_id, controller_id)
    return crowd


def destroy_scene_actors(client, world, ego, vehicle_ids, walker_ids, controller_ids):
    """销毁本场景全部 actor：先停行人控制器，再批量销毁，避免悬挂控制器报错。"""
    for controller in world.get_actors(controller_ids):
        controller.stop()
    ego_ids = [ego.id] if ego is not None else []
    client.apply_batch([_DestroyActor(x)
                        for x in controller_ids + walker_ids + vehicle_ids + ego_ids])
