# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import copy
import math
import numpy as np
import carla
from common.navigation_command import RECOVER_LEFT, RECOVER_RIGHT
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal
from agents.navigation.behavior_prediction import BayesianActorTracker
from agents.navigation.static_obstacles import StaticObstacleIndex
from common.obstacle_geometry import projected_extents
from agents.navigation.risk_field import evaluate_route_risk, plan_front_vehicle_speed
from agents.navigation.longitudinal_planner import plan_longitudinal_speed
from agents.navigation.speed_tracking import allow_coasting, limit_launch_throttle, waiting_coast_speed
from agents.navigation.maneuver_safety import waiting_annotation
from agents.navigation.route_recovery import project_route, sample_route, temporary_route, minimum_recovery_distance
from agents.navigation.route_recovery import moving_escape_route
from agents.navigation.maneuver_safety import assess_maneuver, parallel_hold_path, choose_candidate
from agents.navigation.moving_conflicts import moving_threats, swept_motion_check, pedestrian_blocks_route
from agents.navigation.maneuver_tracking import pursuit_steering, track_committed_path, wheelbase_from_positions
from agents.navigation.maneuver_lanes import adjacent_candidates, check_path_geometry, check_opposing_sight

from agents.tools.misc import get_speed, positive, compute_distance

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. It follows
    lane changes only when they are part of the assigned navigation route.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    _STATE_IDS = {
        "CRUISE": 0, "FOLLOW": 1, "YIELD": 2, "STOPPING": 3,
        "LANE_CHANGE": 4, "APPROACH_STOP": 5, "CREEP": 6,
        "AVOID_LEFT": 7, "AVOID_RIGHT": 8,
        "RECOVER_LEFT": 9, "RECOVER_RIGHT": 10,
    }
    _REASON_IDS = {
        "cruise": 0, "follow_vehicle": 1, "predicted_conflict": 2,
        "pedestrian_risk": 3, "red_light": 4, "yellow_light": 5,
        "comfortable_braking": 6, "lane_change_safe": 7,
        "emergency_stop": 8, "red_light_approach": 9,
        "yellow_light_approach": 10, "red_light_creep": 11,
        "curve_speed": 12, "lane_centering": 13,
        "obstacle_avoid_left": 14, "obstacle_avoid_right": 15,
        "route_recover_left": 16, "route_recover_right": 17,
    }

    def __init__(self, vehicle, behavior='normal', opt_dict=None, map_inst=None,
                 grp_inst=None, behavior_config=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        config = dict(behavior_config or {})
        planner_options = dict(opt_dict or {})
        control_dt = (vehicle.get_world().get_settings().fixed_delta_seconds or .05)
        planner_options.setdefault("longitudinal_control_dict", {
            "K_P": float(config.get("control_longitudinal_kp", .12)),
            "K_I": float(config.get("control_longitudinal_ki", .02)),
            "K_D": 0., "dt": float(control_dt)})
        planner_options.setdefault("base_min_distance",
                                   float(config.get("planner_base_min_distance_m", 1.5)))
        planner_options.setdefault("distance_ratio",
                                   float(config.get("planner_distance_ratio", 0.25)))
        planner_options.setdefault("continuous_route_tracking",
                                   bool(config.get("continuous_route_tracking", True)))
        super().__init__(vehicle, opt_dict=planner_options, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious.create()

        elif behavior == 'normal':
            self._behavior = Normal.create()

        elif behavior == 'aggressive':
            self._behavior = Aggressive.create()
        else:
            raise ValueError("behavior must be cautious, normal, or aggressive")

        self._predictive_enabled = bool(config.get("predictive_enabled", True))
        self._prediction_horizon_s = float(config.get("prediction_horizon_s", 3.0))
        self._prediction_step_s = float(config.get("prediction_step_s", 0.25))
        self._prediction_max_actors = int(config.get("prediction_max_actors", 16))
        self._reaction_time_s = float(config.get("reaction_time_s", 0.7))
        self._comfort_decel_mps2 = float(config.get("comfort_decel_mps2", 3.0))
        self._max_accel_mps2 = float(config.get("max_accel_mps2", 2.2))
        self._max_jerk_mps3 = float(config.get("max_jerk_mps3", 4.0))
        self._longitudinal_horizon_s = float(
            config.get("longitudinal_horizon_s", 3.0))
        self._longitudinal_step_s = float(
            config.get("longitudinal_step_s", 0.25))
        self._longitudinal_preview_s = float(
            config.get("longitudinal_preview_s", 0.5))
        self._route_plan_horizon_m = float(config.get("route_plan_horizon_m", 40.0))
        self._risk_lateral_margin_m = float(config.get("risk_lateral_margin_m", 0.35))
        self._front_min_gap_m = float(config.get("front_min_gap_m", 3.0))
        self._front_confidence_z = float(config.get("front_confidence_z", 1.0))
        self._front_lookahead_s = float(config.get("front_lookahead_s", 1.5))
        self._front_emergency_decel_mps2 = float(
            config.get("front_emergency_decel_mps2", 8.0))
        self._risk_slow_threshold = float(config.get("risk_slow_threshold", 0.18))
        self._risk_yield_threshold = float(config.get("risk_yield_threshold", 0.42))
        self._risk_stop_threshold = float(config.get("risk_stop_threshold", 0.78))
        self._creep_speed_mps = float(config.get("creep_speed_mps", 1.0))
        self._stop_line_margin_m = float(config.get("stop_line_margin_m", 3.5))
        self._corner_lateral_accel_mps2 = float(
            config.get("corner_lateral_accel_mps2", 2.0))
        self._corner_preview_m = float(config.get("corner_preview_m", 30.0))
        self._turn_alignment_speed_mps = float(
            config.get("turn_alignment_speed_mps", 4.0))
        self._turn_alignment_threshold_deg = float(config.get("turn_alignment_threshold_deg", 5.0))
        self._lane_centering_speed_mps = float(
            config.get("lane_centering_speed_mps", 5.0))
        self._obstacle_bypass_enabled = bool(config.get("obstacle_bypass_enabled", True))
        self._moving_evasion_enabled = bool(config.get('moving_evasion_enabled', True))
        self._moving_evasion_horizon_s = float(config.get('moving_evasion_horizon_s', 6.))
        self._moving_evasion_margin_m = float(config.get('moving_evasion_margin_m', .6))
        self._moving_evasion_max_speed_mps = float(config.get('moving_evasion_max_speed_mps', 5.))
        self._moving_evasion_max_offset_m = float(config.get('moving_evasion_max_offset_m', 3.5))
        self._moving_evasion_entry_m = float(config.get('moving_evasion_entry_m', 14.))
        self._moving_evasion_tracking_reserve_m = float(config.get('moving_evasion_tracking_reserve_m', .4))
        self._moving_evasion_hard_mode_probability = float(config.get('moving_evasion_hard_mode_probability', .25))
        self._moving_event = None
        self._pending_moving_id = None
        self._pedestrian_yield_ids = set()
        self._pedestrian_clear_since = None
        self._moving_path = None
        self._moving_recovery = False
        self._moving_vehicle_geometry = None
        self._moving_admission = False
        self._obstacle_min_distance_m = float(config.get("obstacle_min_distance_m", 14.0))
        self._obstacle_max_distance_m = float(config.get("obstacle_max_distance_m", 32.0))
        self._obstacle_approach_m = float(config.get("obstacle_approach_m", 18.0))
        self._obstacle_clear_m = float(config.get("obstacle_clear_m", 7.0))
        self._obstacle_stationary_time_s = float(config.get("obstacle_stationary_time_s", 2.0))
        self._obstacle_entry_clear_m = float(config.get("obstacle_entry_clear_m", 7.0))
        self._maneuver_prediction_s = float(config.get("maneuver_prediction_s", 3.0))
        self._maneuver_min_gap_m = float(config.get("maneuver_min_gap_m", 3.0))
        self._maneuver_rear_ttc_s = float(config.get("maneuver_rear_ttc_s", 4.0))
        self._maneuver_conflict_probability = float(config.get("maneuver_conflict_probability", .2))
        self._maneuver_clear_time_s = float(config.get("maneuver_clear_time_s", .5))
        self._maneuver_speed_mps = float(config.get("maneuver_speed_mps", 2.5))
        self._maneuver_allow_opposite = bool(config.get("maneuver_allow_opposite", True))
        self._maneuver_oncoming_speed_mps = float(config.get("maneuver_oncoming_speed_mps", 15.0))
        self._maneuver_return_buffer_s = float(config.get("maneuver_return_buffer_s", 4.0))
        self._maneuver_max_curvature = float(config.get("maneuver_max_curvature", .18))
        self._control_coast_speed_mps = float(config.get("control_coast_speed_mps", 3.))
        self._control_coast_decel_mps2 = float(config.get("control_coast_decel_mps2", .25))
        self._control_launch_throttle_rate = float(config.get("control_launch_throttle_rate", .6))
        self._control_launch_throttle_max = float(config.get("control_launch_throttle_max", .15))
        self._navigation_goal_approach_s = float(config.get('navigation_goal_approach_s', 3.))
        self._navigation_goal_stop_buffer_m = float(config.get('navigation_goal_stop_buffer_m', .5))
        self._stationary_since = {}
        self._static_obstacle_margin_m = float(config.get('static_obstacle_margin_m', .6))
        self._static_obstacle_in_lane_enabled = bool(config.get('static_obstacle_in_lane_enabled', True))
        self._static_obstacle_boundary_margin_m = float(config.get('static_obstacle_boundary_margin_m', .15))
        self._static_obstacles = (StaticObstacleIndex(
            self._world, float(config.get('static_obstacle_radius_m', 60.)),
            float(config.get('static_obstacle_min_height_m', .08)))
            if config.get('static_obstacles_enabled', True) else None)
        self._maneuver_clear_since = {}
        self._maneuver_last_check = {}
        self._maneuver_candidates = []
        self._maneuver_safety = None
        self._maneuver_wait_distance_m = None
        self._recovery_goal_world = None
        self._route_recovery_m = float(config.get("route_recovery_m", 14.0))
        self._route_recovery_trigger_m = float(config.get("route_recovery_trigger_m", 0.9))
        self._tracker = BayesianActorTracker(
            horizon_s=self._prediction_horizon_s,
            step_s=self._prediction_step_s,
            process_noise=float(config.get("prediction_process_noise", 1.5)),
            measurement_noise=float(config.get("prediction_measurement_noise", 0.25)),
            max_stale_s=float(config.get("prediction_max_stale_s", 1.0)),
            multi_intent_enabled=bool(config.get(
                "prediction_multi_intent_enabled", True)),
            intent_map_radius_m=float(config.get(
                "prediction_intent_map_radius_m", 60.0)),
            intent_brake_mps2=float(config.get(
                "prediction_intent_brake_mps2", 2.0)),
            intent_accelerate_mps2=float(config.get(
                "prediction_intent_accelerate_mps2", 1.2)),
            intent_lane_change_s=float(config.get(
                "prediction_intent_lane_change_s", 3.0)),
        )
        self._state = "CRUISE"
        self._state_since = 0.0
        self._last_light_distance_m = float("inf")
        self._last_light_id = None
        self._active_stop_light_id = None
        self._active_stop_waypoint = None
        self._light_stop_cache = {}
        self._creep_light_id = None
        self._last_target_speed_mps = 0.0
        self._last_target_accel_mps2 = 0.0
        self._last_risk = {}
        self._last_predictions = []
        self._last_longitudinal_plan = None
        self._pedestrian_braking_plan = None
        self._maneuver_obstacle_id = None
        self._maneuver_obstacle_world = None
        self._maneuver_side = 0
        self._maneuver_lane_offset_m = 0.
        self._recovery_side = 0
        self._recovery_committed = False
        self._active_maneuver = None
        self._active_route_points = None
        self._last_snapshot = self._empty_snapshot("cruise")

    @staticmethod
    def _vec2(value):
        return [float(value.x), float(value.y)]

    @staticmethod
    def _transform_payload(transform):
        return [float(transform.location.x), float(transform.location.y),
                float(transform.location.z), float(transform.rotation.roll),
                float(transform.rotation.pitch), float(transform.rotation.yaw)]

    def _empty_snapshot(self, reason):
        reason = str(reason)
        return {
            "schema_version": 1,
            "decision_frame_id": None,
            "result_frame_id": None,
            "decision_sim_time": None,
            "result_sim_time": None,
            "state": self._state,
            "state_id": self._STATE_IDS.get(self._state, -1),
            "front_obstacle_plan": dict(getattr(self, '_last_front_plan', {}) or {}),
            "state_since_sim_time": float(self._state_since),
            "state_duration_s": 0.0,
            "reason": reason,
            "reason_code": self._REASON_IDS.get(reason, -1),
            "reason_codes": [self._REASON_IDS.get(reason, -1)],
            "navigation": {},
            "decision_ego_transform": None,
            "route_plan": {"coordinate_frame": "world", "truncated": False,
                            "total_queue_points": 0, "points": []},
            "target_speed_mps": 0.0,
            "target_acceleration_mps2": 0.0,
            "speed_limit_mps": 0.0,
            "traffic_light_distance_m": None,
            "traffic_light_id": None,
            "risk": dict(self._last_risk),
            "prediction": {"horizon_s": self._prediction_horizon_s,
                           "step_s": self._prediction_step_s,
                           "tracked_actor_count": 0, "valid_track_count": 0,
                           "actors": []},
            "lane_change": {"available_left": False, "available_right": False,
                            "cooldown_s": 0.0},
            "longitudinal_plan": None,
            "maneuver_plan": None,
        }

    def _route_plan_snapshot(self, ego_transform):
        plan = list(self._local_planner.get_plan())
        points = []
        travelled = 0.0
        previous = ego_transform.location
        for waypoint, option in plan:
            location = waypoint.transform.location
            travelled += location.distance(previous)
            points.append({
                "location_world": [float(location.x), float(location.y), float(location.z)],
                "rotation_world": [float(waypoint.transform.rotation.roll),
                                    float(waypoint.transform.rotation.pitch),
                                    float(waypoint.transform.rotation.yaw)],
                "road_option": int(option),
                "road_id": int(waypoint.road_id),
                "section_id": int(waypoint.section_id),
                "lane_id": int(waypoint.lane_id),
                "lane_type": str(waypoint.lane_type),
                "is_junction": bool(waypoint.is_junction),
                "arc_m": float(travelled),
            })
            previous = location
        return {
            "coordinate_frame": "world",
            "truncated": False,
            "total_queue_points": len(plan),
            "points": points,
        }

    def get_decision_snapshot(self):
        """Return a detached, msgpack-safe snapshot of the last control decision."""
        return copy.deepcopy(self._last_snapshot)

    def get_behavior_state(self):
        return self.get_decision_snapshot()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        velocity = self._vehicle.get_velocity()
        self._speed = math.hypot(velocity.x, velocity.y) * 3.6
        self._speed_limit = max(float(self._vehicle.get_speed_limit()), 0.0)
        self._local_planner.set_speed(max(self._speed_limit, self._min_speed))
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def _set_state(self, state, now):
        if state != self._state:
            self._state = state
            self._state_since = float(now)

    def _actor_predictions(self, timestamp):
        actors = list(self._world.get_actors().filter("*vehicle*"))
        actors.extend(list(self._world.get_actors().filter("*walker.pedestrian*")))
        self._last_predictions = (self._tracker.update(
            actors, timestamp, ego_id=self._vehicle.id,
            road_map=self._map, ego_position=self._vehicle.get_location())
            if self._predictive_enabled else [])
        if self._static_obstacles is not None:
            self._last_predictions.extend(self._static_obstacles.predictions(
                self._vehicle, self._prediction_horizon_s, self._prediction_step_s))
        return self._last_predictions

    def _route_points(self):
        if self._active_route_points is not None:
            return self._active_route_points
        points = [self._vehicle.get_location()]
        travelled = 0.0
        for waypoint, _ in list(self._local_planner.get_plan()):
            next_location = waypoint.transform.location
            travelled += points[-1].distance(next_location)
            points.append(next_location)
            if travelled >= self._route_plan_horizon_m:
                break
        return [[float(point.x), float(point.y)] for point in points]

    def _assigned_centerline(self):
        tracker = self._local_planner._route_tracker
        if tracker is not None:
            return tracker.local_points()
        plan = list(self._local_planner.get_plan())
        if len(plan) < 2:
            return None
        first = plan[0][0].transform
        forward = first.get_forward_vector()
        location = first.location
        points = [[location.x - 8.0 * forward.x,
                   location.y - 8.0 * forward.y]]
        points.extend([[waypoint.transform.location.x,
                        waypoint.transform.location.y]
                       for waypoint, _ in plan])
        return points

    def _maneuver_gap(self, centerline, path, ego_xy, target_offset, key, now):
        bounds = self._vehicle.bounding_box.extent
        waypoint = list(self._local_planner.get_plan())[0][0]
        in_lane = (path['phase'] == 'avoid' and
                   abs(target_offset) + bounds.y + self._static_obstacle_boundary_margin_m <= waypoint.lane_width * .5)
        receiving = (self._physical_adjacent_lane(waypoint, 1 if target_offset > 0. else -1,
                                                self._maneuver_allow_opposite)
                     if abs(target_offset) > .1 and not in_lane else waypoint)
        receiving = receiving if receiving is not None else waypoint
        route_forward = waypoint.transform.get_forward_vector()
        lane_forward = receiving.transform.get_forward_vector()
        same_direction = route_forward.x * lane_forward.x + route_forward.y * lane_forward.y > .8
        reservation = (max(path["return_distance_m"], 0.) / max(min(
            self._speed / 3.6, self._maneuver_speed_mps), 2.) + self._maneuver_return_buffer_s)
        forward = self._vehicle.get_transform().get_forward_vector()
        tangent = project_route(centerline, ego_xy)[2]
        aligned = abs(forward.x * tangent[0] + forward.y * tangent[1])
        crossed = abs(forward.x * tangent[1] - forward.y * tangent[0])
        moving_check = bool(getattr(self, '_moving_admission', False) or getattr(self, '_moving_event', None))
        result = assess_maneuver(
            centerline, path["points"], self._last_predictions, ego_xy, self._speed / 3.6,
            (bounds.x, bounds.y), target_offset,
            horizon_s=self._maneuver_prediction_s, minimum_gap_m=self._maneuver_min_gap_m,
            rear_ttc_s=self._maneuver_rear_ttc_s,
            probability_threshold=self._maneuver_conflict_probability,
            lateral_margin_m=self._risk_lateral_margin_m,
            receiving_width_m=receiving.lane_width,
            current_half_width_m=bounds.y * aligned + bounds.x * crossed,
            reservation_s=reservation, oncoming_speed_mps=self._maneuver_oncoming_speed_mps,
            opposing_lane=not same_direction, physical_probability=moving_check)
        source_lane = waypoint if path["phase"] == "avoid" else self._map.get_waypoint(self._vehicle.get_location())
        if path["phase"] == "recover":
            # The centre may already be back in the assigned lane while the
            # body still straddles its old lane. Keep that source corridor
            # available until the return is complete.
            ego_lateral = project_route(centerline, ego_xy)[1]
            route_center = waypoint.transform.location
            for adjacent in (waypoint.get_left_lane(), waypoint.get_right_lane()):
                if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
                    continue
                location = adjacent.transform.location
                offset = ((location.x - route_center.x) * route_forward.y -
                          (location.y - route_center.y) * route_forward.x)
                if offset * ego_lateral > .1:
                    source_lane = adjacent
                    break
        geometry = check_path_geometry(self._map, centerline, path, source_lane, receiving, bounds)
        if geometry.get("max_curvature", 0.) > self._maneuver_max_curvature:
            geometry.update(safe=False, kind="curvature_limit")
        result["geometry"] = geometry
        result["same_direction"] = same_direction
        result['in_lane'] = bool(in_lane)
        result["safe"] = result["safe"] and geometry["safe"]
        if moving_check:
            if (self._static_obstacles is not None and path['return_distance_m'] + bounds.x +
                    self._moving_evasion_margin_m > self._static_obstacles.radius_m):
                geometry.update(safe=False, kind='static_observation_horizon')
                result['safe'] = False
            requested = max(self._maneuver_speed_mps, self._speed / 3.6 + .5,
                            (self._moving_event or {}).get('execution_speed_mps', 0.))
            maximum = min(self._moving_evasion_max_speed_mps,
                math.sqrt(self._corner_lateral_accel_mps2 / max(geometry.get('max_curvature', 0.), .001)))
            speeds = list(dict.fromkeys([min(requested, maximum), maximum,
                                         min(self._maneuver_speed_mps, maximum)]))
            speed_checks = []
            for execution_speed in speeds:
                motion = swept_motion_check(path['points'], self._last_predictions,
                    self._speed / 3.6, (bounds.x, bounds.y), execution_speed,
                    decel_mps2=self._comfort_decel_mps2,
                    horizon_s=self._moving_evasion_horizon_s,
                    margin_m=self._moving_evasion_margin_m,
                    tracking_margin_m=self._moving_evasion_tracking_reserve_m,
                    ego_heading=math.radians(self._vehicle.get_transform().rotation.yaw),
                    settled_follower_ids=result.get('settled_follower_ids', ()),
                    hard_mode_probability=self._moving_evasion_hard_mode_probability)
                speed_checks.append(motion)
                if motion['safe']:
                    break
            result['speed_candidates'] = speed_checks
            result['motion_check'] = motion
            result['execution_speed_mps'] = execution_speed
            result['blocking_actors'].extend(motion['blocking_actors'])
            result['safe'] = result['safe'] and motion['safe']
        if not same_direction and result["safe"]:
            sight_distance = (self._maneuver_oncoming_speed_mps + self._maneuver_speed_mps) * reservation + self._maneuver_min_gap_m
            result["sight"] = check_opposing_sight(self._world, receiving, sight_distance,
                                                  self._vehicle.get_transform(), bounds)
            result["safe"] = result["safe"] and result["sight"]["safe"]
        result["cost"] = (10. * result["max_conflict_probability"] +
                          5. * geometry.get("max_curvature", 0.) +
                          .02 * path["return_distance_m"] +
                          1. / max(result["min_gap_m"] or 100., 1.))
        # A stale successful check is not continuous clearance after a gap in
        # evaluation, such as an obstacle moving and stopping again.
        if now - self._maneuver_last_check.get(key, now) > .25:
            self._maneuver_clear_since.pop(key, None)
        self._maneuver_last_check[key] = now
        if not result["safe"]:
            self._maneuver_clear_since.pop(key, None)
        else:
            self._maneuver_clear_since.setdefault(key, now)
        result["clear_duration_s"] = max(0., now - self._maneuver_clear_since.get(key, now))
        straddling_receiver = abs(target_offset - project_route(centerline, ego_xy)[1]) < (
            receiving.lane_width * .5 + bounds.y * aligned + bounds.x * crossed - .2)
        continuing = bool(path['phase'] == 'recover' and self._recovery_committed and straddling_receiver)
        result['continuing_committed_merge'] = bool(result['safe'] and continuing)
        result["ready"] = bool(result["safe"] and (
            result["clear_duration_s"] >= self._maneuver_clear_time_s or continuing or
            moving_check))
        result['immediate_moving_admission'] = bool(result['ready'] and moving_check)
        result['admission_status'] = ('ready' if result['ready'] else
                                      'confirming' if result['safe'] else 'blocked')
        if result['ready'] and path['phase'] == 'recover':
            self._recovery_committed = True
        return result

    @staticmethod
    def _physical_adjacent_lane(waypoint, side, allow_opposite=False):
        """Map world-space left/right to an admissible physical adjacent lane."""
        return next((item["waypoint"] for item in adjacent_candidates(waypoint, allow_opposite)
                     if item["side"] == side), None)

    def _update_maneuver(self):
        """Plan a temporary bypass or return path without changing navigation."""
        self._active_maneuver = None
        self._active_route_points = None
        self._maneuver_safety = None
        self._maneuver_wait_distance_m = None
        self._moving_admission = False
        if not self._obstacle_bypass_enabled:
            return
        centerline = self._assigned_centerline()
        if centerline is None:
            return
        ego = self._vehicle.get_location()
        ego_xy = [float(ego.x), float(ego.y)]
        now = float(self._world.get_snapshot().timestamp.elapsed_seconds)
        seen = set()
        for prediction in self._last_predictions:
            seen.add(prediction.actor_id)
            if prediction.actor_type == "vehicle" and prediction.speed_mps <= .45:
                self._stationary_since.setdefault(prediction.actor_id, now)
            else:
                self._stationary_since.pop(prediction.actor_id, None)
        self._stationary_since = {key: value for key, value in self._stationary_since.items() if key in seen}
        projected_ego = project_route(centerline, ego_xy)
        if projected_ego is None:
            return
        ego_arc, lateral, tangent = projected_ego
        transform = self._vehicle.get_transform()
        forward = transform.get_forward_vector()
        bounds = self._vehicle.bounding_box
        # Project the whole body onto the route normal; heading errors make
        # the footprint wider than extent.y even before the centre drifts.
        body_half_width = (
            bounds.extent.y * abs(forward.x * tangent[0] + forward.y * tangent[1]) +
            bounds.extent.x * abs(forward.x * tangent[1] - forward.y * tangent[0]))
        route_waypoint = list(self._local_planner.get_plan())[0][0]
        lane_width = max(float(route_waypoint.lane_width), 2.5)
        if self._maneuver_obstacle_world is not None:
            tracked_obstacle = next((item for item in self._last_predictions
                                     if item.actor_id == self._maneuver_obstacle_id), None)
            projected_obstacle = project_route(
                centerline, self._maneuver_obstacle_world)
            moving_passed = bool(self._moving_event is not None and tracked_obstacle is not None and
                project_route(centerline, tracked_obstacle.current_position or tracked_obstacle.means[0])[0] - ego_arc <
                -bounds.extent.x - tracked_obstacle.extent_xy[0] - 2.)
            if (projected_obstacle is None or projected_obstacle[0] - ego_arc < -5.0 or
                    moving_passed or
                    (self._moving_event is None and tracked_obstacle is not None and tracked_obstacle.speed_mps > .8)):
                self._maneuver_obstacle_id = None
                self._maneuver_obstacle_world = None
                self._moving_recovery = self._moving_event is not None
                self._moving_event = None
                self._moving_path = None
                if abs(lateral) > .2:
                    self._recovery_side = -1 if lateral > 0. else 1
                    self._recovery_committed = False
                    self._recovery_goal_world = None
        if (self._maneuver_obstacle_world is None and not route_waypoint.is_junction and
                self._direction not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT)):
            candidates = []
            threats = (moving_threats(centerline, self._last_predictions, ego_xy,
                self._speed / 3.6, (bounds.extent.x, bounds.extent.y),
                self._moving_evasion_horizon_s, self._moving_evasion_margin_m)
                if self._moving_evasion_enabled else [])
            threat = threats[0] if threats else None
            if threat is not None:
                self._pending_moving_id = threat['actor'].actor_id
                candidates.append((0., threat['actor']))
                self._moving_admission = True
            elif self._pending_moving_id is not None:
                pending = next((item for item in self._last_predictions
                                if item.actor_id == self._pending_moving_id), None)
                projected = project_route(centerline, pending.current_position or pending.means[0]) if pending is not None else None
                if (pending is None or projected[0] < ego_arc - 5. or
                        abs(projected[1]) > body_half_width + pending.extent_xy[1] + self._moving_evasion_margin_m):
                    self._pending_moving_id = None
                elif pending.actor_type == 'vehicle' and pending.speed_mps <= .45:
                    # A threatening oncoming car stopping is not a newly
                    # discovered parked car: retain the stop/gap decision
                    # instead of launching during the stationary timer.
                    candidates.append((0., pending))
            for prediction in self._last_predictions:
                if prediction.actor_type not in ("vehicle", "static") or \
                        prediction.speed_mps > 0.45 or not len(prediction.means) or \
                        (prediction.actor_type != 'static' and
                         now - self._stationary_since.get(prediction.actor_id, now) < self._obstacle_stationary_time_s):
                    continue
                projected = project_route(centerline, prediction.means[0])
                if projected is None:
                    continue
                arc, actor_lateral, _ = projected
                obstacle_width = (projected_extents(prediction.static_box, projected[2])[1]
                                  if prediction.actor_type == 'static' else prediction.extent_xy[1])
                distance = arc - ego_arc
                if (self._obstacle_min_distance_m <= distance <=
                        self._obstacle_max_distance_m and
                        abs(actor_lateral) <=
                        max(float(self._vehicle.bounding_box.extent.y), 0.5) +
                        float(obstacle_width) + (self._static_obstacle_margin_m
                                                 if prediction.actor_type == 'static' else .25)):
                    candidates.append((distance, prediction))
            # Waiting traffic at a route signal remains a normal stop, not a
            # parked obstacle to pass.
            if candidates and self._traffic_light_target(route_waypoint)[0] is not None:
                candidates = [item for item in candidates if item[1].actor_type == 'static' or
                              (threat is not None and item[1].actor_id == threat['actor'].actor_id)]
            if candidates:
                _, obstacle = min(candidates, key=lambda item: item[0])
                moving = bool(threat is not None and obstacle.actor_id == threat['actor'].actor_id)
                self._moving_admission = moving
                obstacle_xy = threat['anchor'] if moving else obstacle.means[0].tolist()
                obstacle_arc = project_route(centerline, obstacle_xy)[0]
                queued = False
                for other in self._last_predictions:
                    if (other.actor_id == obstacle.actor_id or
                            other.actor_type != "vehicle" or not len(other.means)):
                        continue
                    projected_other = project_route(centerline, other.means[0])
                    if (not moving and obstacle.actor_type != 'static' and projected_other is not None and
                            2.0 < projected_other[0] - obstacle_arc < 14.0 and
                            abs(projected_other[1]) < lane_width * 0.6):
                        queued = True
                        break
                if not queued:
                    attempts = []
                    lanes = adjacent_candidates(route_waypoint, self._maneuver_allow_opposite)
                    if moving:
                        lanes = [dict(lane, offset_m=math.copysign(min(abs(lane['offset_m']), self._moving_evasion_max_offset_m),
                                                                 lane['offset_m'])) for lane in lanes]
                        if obstacle.actor_type == 'pedestrian':
                            # A crossing pedestrian may only need a partial
                            # lateral escape. Reserve the same complete return
                            # and both lanes; do not insist on reaching the
                            # adjacent centre before considering a safe path.
                            lanes += [dict(lane, offset_m=lane['offset_m'] * fraction)
                                      for lane in list(lanes) for fraction in (.5, .75)]
                    if (obstacle.actor_type == 'static' or moving) and self._static_obstacle_in_lane_enabled:
                        projected = project_route(centerline, obstacle_xy)
                        width = (projected_extents(obstacle.static_box, projected[2])[1]
                                 if obstacle.actor_type == 'static' else obstacle.extent_xy[1])
                        max_offset = lane_width * .5 - body_half_width - self._static_obstacle_boundary_margin_m
                        for offset in (projected[1] + width + body_half_width + self._static_obstacle_margin_m,
                                       projected[1] - width - body_half_width - self._static_obstacle_margin_m):
                            if .05 < abs(offset) <= max_offset:
                                lanes.append({'side': 1 if offset > 0. else -1,
                                              'offset_m': offset, 'same_direction': True})
                    for lane in lanes:
                        side = lane["side"]
                        candidate = temporary_route(
                            centerline, ego_xy, side, abs(lane["offset_m"]), obstacle_position=obstacle_xy,
                            approach_m=self._obstacle_approach_m, clear_m=self._obstacle_clear_m,
                            recovery_m=self._route_recovery_m, entry_clear_m=self._obstacle_entry_clear_m,
                            ego_heading=math.radians(transform.rotation.yaw))
                        if moving:
                            candidate = moving_escape_route(centerline, ego_xy, side, abs(lane['offset_m']),
                                obstacle_xy, math.radians(transform.rotation.yaw),
                                entry_m=(self._moving_evasion_entry_m if threat['ttc_s'] >= 4. else
                                    minimum_recovery_distance(lane['offset_m'] - lateral, bounds.extent.x,
                                        self._moving_evasion_entry_m, self._maneuver_max_curvature)),
                                clear_m=self._obstacle_clear_m, recovery_m=self._route_recovery_m)
                        if candidate is None:
                            continue
                        safety = self._maneuver_gap(centerline, candidate, ego_xy, lane["offset_m"],
                                                    ("avoid", side, obstacle.actor_id, round(abs(lane['offset_m']), 2)), now)
                        attempts.append(dict(safety, side=side, offset_m=lane["offset_m"],
                                             committed_path=candidate if moving else None))
                    chosen = choose_candidate(attempts)
                    self._maneuver_candidates = attempts
                    if chosen is not None:
                        self._maneuver_side = chosen["side"]
                        self._maneuver_lane_offset_m = abs(chosen["offset_m"])
                        self._maneuver_obstacle_id = obstacle.actor_id
                        self._maneuver_obstacle_world = obstacle_xy
                        if moving:
                            self._moving_event = {key: value for key, value in threat.items() if key != 'actor'}
                            self._moving_event['actor_id'] = obstacle.actor_id
                            self._moving_event['execution_speed_mps'] = chosen['execution_speed_mps']
                            self._moving_path = copy.deepcopy(chosen['committed_path'])
                    self._maneuver_safety = ({"phase": "avoid", "candidates": attempts,
                                             "waiting_for_gap": self._maneuver_obstacle_world is None}
                                            if attempts else None)
                    if self._maneuver_obstacle_world is None and attempts:
                        # Wait before the minimum bypass entry distance, so a
                        # later opening still leaves room for a smooth detour.
                        self._maneuver_wait_distance_m = max(0., obstacle_arc - ego_arc -
                                                              self._obstacle_min_distance_m - self._front_min_gap_m)
        obstacle = self._maneuver_obstacle_world
        if obstacle is None:
            ego_waypoint = self._map.get_waypoint(ego)
            if self._recovery_side:
                if abs(lateral) < 0.2 or lateral * self._recovery_side >= 0.0:
                    self._moving_recovery = False
                    self._recovery_side = 0
                    self._recovery_committed = False
                    self._maneuver_side = 0
                    self._recovery_goal_world = None
                    return
            else:
                self._recovery_committed = False
                if (ego_waypoint is None or ego_waypoint.is_junction or
                        route_waypoint.is_junction or
                        self._direction in (RoadOption.CHANGELANELEFT,
                                            RoadOption.CHANGELANERIGHT) or
                        abs(lateral) < min(self._route_recovery_trigger_m,
                            max(0.15, lane_width * 0.5 -
                                body_half_width - 0.2))):
                    self._maneuver_side = 0
                    return
                self._recovery_side = -1 if lateral > 0.0 else 1
            side = self._recovery_side
        else:
            self._recovery_side = 0
            self._recovery_committed = False
            self._recovery_goal_world = None
            side = self._maneuver_side
        end_position = sample_route(centerline, 1e9)[0]
        end_arc = project_route(centerline, end_position)[0]
        # Navigation endpoints are not physical road boundaries. Reserving
        # three metres inside an already short final segment can make an
        # otherwise feasible return exceed the curvature limit indefinitely.
        recovery_distance = min(self._route_recovery_m, max(3., end_arc - ego_arc - .5))
        if obstacle is None:
            recovery_distance = min(recovery_distance, minimum_recovery_distance(
                lateral, bounds.extent.x, self._route_recovery_m, self._maneuver_max_curvature))
        if obstacle is None and self._recovery_goal_world is not None:
            recovery_distance = max(3., project_route(centerline, self._recovery_goal_world)[0] - ego_arc)
        path = temporary_route(
            centerline, ego_xy, side, self._maneuver_lane_offset_m if obstacle is not None else lane_width,
            obstacle_position=obstacle,
            lookahead_m=max(5.0, self._speed / 3.6 * 0.8 + 4.0),
            approach_m=self._obstacle_approach_m,
            clear_m=self._obstacle_clear_m,
            recovery_m=recovery_distance, entry_clear_m=self._obstacle_entry_clear_m,
            ego_heading=math.radians(transform.rotation.yaw))
        if path is None:
            return
        if obstacle is not None and self._moving_path is not None:
            # Follow the admitted world-space escape. Rebuilding the initial
            # correction at every tick moves its full-offset point away from
            # ego and can postpone the escape until after the encounter.
            path = copy.deepcopy(self._moving_path)
            progress = project_route(path['points'], ego_xy)[0]
            original = path['points']
            end = project_route(original, original[-1])[0]
            path['points'] = track_committed_path(original, ego_xy, math.radians(transform.rotation.yaw))
            path['target_world_xy'] = sample_route(original,
                min(end, progress + max(5., self._speed / 3.6 * .8 + 4.)))[0].tolist()
            path['ego_route_offset_m'] = lateral
            path['obstacle_route_distance_m'] = project_route(centerline, obstacle)[0] - ego_arc
        target_offset = side * self._maneuver_lane_offset_m if obstacle is not None else 0.
        key = ((path['phase'], side, self._maneuver_obstacle_id, round(self._maneuver_lane_offset_m, 2))
               if obstacle is not None else (path['phase'], side, None))
        safety = self._maneuver_gap(centerline, path, ego_xy, target_offset, key, now)
        if obstacle is not None and self._moving_event is not None and not safety['ready']:
            # Tracking error can make a short connector to the old reference
            # infeasible. Replan a bounded, smooth remaining lateral move;
            # shortening it with the remaining offset avoids a receding goal.
            residual = target_offset - lateral
            entry = minimum_recovery_distance(residual, bounds.extent.x, self._moving_evasion_entry_m, self._maneuver_max_curvature)
            repair = moving_escape_route(centerline, ego_xy, side, abs(target_offset),
                obstacle, math.radians(transform.rotation.yaw), entry_m=entry,
                clear_m=self._obstacle_clear_m, recovery_m=recovery_distance)
            repair_safety = self._maneuver_gap(centerline, repair, ego_xy, target_offset, key, now)
            if repair_safety['ready']:
                path, safety = repair, repair_safety
                path['replanned_from_tracking_error'] = True
                self._moving_path = copy.deepcopy(path)
        if obstacle is not None and not safety["same_direction"] and not safety["safe"] and abs(lateral) > .2:
            # A previously admitted opposing-lane reservation can be revoked.
            # Prefer a validated return over waiting indefinitely in that lane.
            abort_side = -1 if lateral > 0. else 1
            abort = temporary_route(centerline, ego_xy, abort_side, lane_width,
                                    recovery_m=recovery_distance)
            if abort is not None:
                abort_safety = self._maneuver_gap(centerline, abort, ego_xy, 0.,
                                                   ("recover", abort_side, None), now)
                if abort_safety["ready"]:
                    path, safety = abort, dict(abort_safety, aborted_opposing=True)
                    obstacle = None
                    self._maneuver_obstacle_id = None
                    self._maneuver_obstacle_world = None
                    self._recovery_side = abort_side
                    side = abort_side
        self._maneuver_safety = dict(safety, phase=path["phase"], waiting_for_gap=not safety["ready"],
                                    candidates=self._maneuver_candidates if obstacle is not None else [])
        if obstacle is None:
            if safety["ready"] and self._recovery_goal_world is None:
                self._recovery_goal_world = sample_route(centerline, ego_arc + recovery_distance)[0].tolist()
            elif not safety["ready"]:
                self._recovery_goal_world = None
            path["recovery_goal_world_xy"] = self._recovery_goal_world
        if not safety["ready"]:
            path["proposed_points"] = path["points"]
            path["points"], path["target_world_xy"] = parallel_hold_path(
                centerline, ego_xy, max(5., self._speed / 3.6 * .8 + 4.))
            path["waiting_for_gap"] = True
            if obstacle is not None:
                self._maneuver_wait_distance_m = max(0., path["obstacle_route_distance_m"] -
                                                      self._obstacle_min_distance_m - self._front_min_gap_m)
            else:
                # Preserve room for the entire return before the navigation
                # endpoint; a blocked recovery must not drive past its goal.
                self._maneuver_wait_distance_m = max(0., end_arc - ego_arc -
                                                      self._route_recovery_m - 3.)
        path["obstacle_actor_id"] = self._maneuver_obstacle_id
        self._active_maneuver = path
        path['moving_conflict'] = copy.deepcopy(self._moving_event)
        self._active_route_points = path["points"]

    def _risk_assessment(self, timestamp, update_predictions=True):
        predictions = (self._actor_predictions(timestamp) if update_predictions
                       else self._last_predictions)
        risk = evaluate_route_risk(
            self._route_points(), predictions,
            self._vehicle.get_location(), self._speed / 3.6,
            route_width_m=(max(float(self._vehicle.bounding_box.extent.y), 0.5) +
                           self._risk_lateral_margin_m),
            pedestrian_route_width_m=max(1.6, getattr(self._map.get_waypoint(
                self._vehicle.get_location()), "lane_width", 3.5) * 0.55))
        self._last_risk = risk
        return risk

    def _front_vehicle_plan(self):
        bounds = self._vehicle.bounding_box.extent
        predictions = self._last_predictions
        if self._moving_path_ready():
            predictions = [item for item in predictions if item.actor_id != self._maneuver_obstacle_id]
        return plan_front_vehicle_speed(
            self._route_points(), predictions,
            self._vehicle.get_location(), self._speed / 3.6,
            bounds.x, bounds.y, self._risk_lateral_margin_m,
            self._reaction_time_s, self._comfort_decel_mps2,
            minimum_gap_m=self._front_min_gap_m,
            confidence_z=self._front_confidence_z,
            lookahead_s=self._front_lookahead_s,
            emergency_decel_mps2=self._front_emergency_decel_mps2)

    def _moving_path_ready(self):
        safety = self._maneuver_safety or {}
        return bool(getattr(self, '_moving_event', None) and self._active_maneuver is not None and
                    safety.get('ready') and (safety.get('motion_check') or {}).get('safe'))

    def _plan_target_speed(self, target_speed_kmh, constraints, dt_s):
        dt_s = max(float(dt_s), 1e-3)
        actual_speed = self._speed / 3.6
        policy_speed = max(float(target_speed_kmh) / 3.6, 0.0)
        # Target acceleration is not measured vehicle inertia. Clear a stale
        # brake command when the policy permits acceleration. A stop outside
        # the planning horizon plus its terminal braking distance is too far
        # away to justify continuing to brake now.
        horizon_speed = (actual_speed +
                         self._max_accel_mps2 * self._longitudinal_horizon_s)
        distant_stop_m = (
            actual_speed * self._longitudinal_horizon_s +
            0.5 * self._max_accel_mps2 * self._longitudinal_horizon_s ** 2 +
            horizon_speed ** 2 / (2.0 * self._comfort_decel_mps2) + 2.0)
        constraint_allows_accel = all(
            (distance > distant_stop_m if limit <= 0.5 else
             math.sqrt(limit * limit + 2.0 * self._comfort_decel_mps2 *
                       max(distance, 0.0)) > actual_speed + 0.25)
            for distance, limit in constraints)
        release_brake = (policy_speed > actual_speed + 0.25 and
                         constraint_allows_accel and
                         self._last_target_accel_mps2 < 0.0)
        previous_accel = (0.0 if release_brake
                          else self._last_target_accel_mps2)
        plan = plan_longitudinal_speed(
            actual_speed, previous_accel, policy_speed, constraints,
            horizon_s=self._longitudinal_horizon_s,
            step_s=min(self._longitudinal_step_s, max(float(dt_s), 0.01)),
            preview_s=self._longitudinal_preview_s,
            max_accel_mps2=self._max_accel_mps2,
            comfort_decel_mps2=self._comfort_decel_mps2,
            max_jerk_mps3=self._max_jerk_mps3)
        preview_speed = plan["target_speed_mps"]
        previous_speed = max(self._last_target_speed_mps, 0.0)
        if abs(previous_speed - actual_speed) > max(1.0, 0.5 * actual_speed):
            previous_speed = actual_speed
        if release_brake:
            previous_speed = max(previous_speed, actual_speed)
        # Preview speed already integrates future acceleration. Differentiating
        # it again can cause runaway commands with delayed vehicle response.
        desired_accel = float(plan["target_acceleration_mps2"])
        jerk_step = self._max_jerk_mps3 * dt_s
        command_accel = min(max(
            desired_accel, previous_accel - jerk_step),
            previous_accel + jerk_step)
        command_speed = max(previous_speed + command_accel * dt_s, 0.0)
        plan["preview_speed_mps"] = preview_speed
        plan["planned_first_acceleration_mps2"] = plan[
            "target_acceleration_mps2"]
        plan["target_speed_mps"] = command_speed
        plan["target_acceleration_mps2"] = command_accel
        self._last_longitudinal_plan = plan
        self._last_target_speed_mps = command_speed
        self._last_target_accel_mps2 = command_accel
        return command_speed * 3.6

    def _navigation_goal_distance(self):
        if not getattr(self._local_planner, '_stop_waypoint_creation', True):
            return None
        plan = list(self._local_planner.get_plan())
        if not plan:
            return 0.
        ego = self._vehicle.get_location()
        endpoint = plan[-1][0].transform
        if ego.distance(endpoint.location) > 80.:
            return None
        if len(plan) == 1:
            forward = endpoint.get_forward_vector()
            return max(0., (endpoint.location.x - ego.x) * forward.x +
                          (endpoint.location.y - ego.y) * forward.y)
        points = [[waypoint.transform.location.x, waypoint.transform.location.y]
                  for waypoint, _ in plan]
        arc, _, tangent = project_route(points, [ego.x, ego.y])
        nearest = sample_route(points, arc)[0]
        along = arc + float(np.dot(np.asarray([ego.x, ego.y]) - nearest, tangent))
        length = float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())
        return max(0., length - along)

    @staticmethod
    def _lane_available(waypoint, side):
        if waypoint is None:
            return False
        marking = (waypoint.left_lane_marking if side == "left"
                   else waypoint.right_lane_marking)
        adjacent = (waypoint.get_left_lane() if side == "left"
                    else waypoint.get_right_lane())
        allowed = ((carla.LaneChange.Left, carla.LaneChange.Both) if side == "left"
                   else (carla.LaneChange.Right, carla.LaneChange.Both))
        return bool(marking is not None and marking.lane_change in allowed and
                    adjacent is not None and adjacent.lane_type == carla.LaneType.Driving
                    and waypoint.lane_id * adjacent.lane_id > 0)

    def _build_snapshot(self, reason, target_speed_kmh, timestamp, control=None):
        transform = self._vehicle.get_transform()
        plan = getattr(self, "_decision_route_plan", None)
        if plan is None:
            plan = self._route_plan_snapshot(transform)
        option = getattr(self, "_decision_navigation_option", self._direction)
        target_waypoint = getattr(self, "_decision_target_waypoint",
                                  self._local_planner.target_waypoint)
        current_waypoint = self._map.get_waypoint(transform.location)
        snapshot = self._empty_snapshot(reason)
        ego_location = transform.location
        nearest = sorted(self._last_predictions, key=lambda item: (
            (float(item.means[0, 0]) - ego_location.x) ** 2 +
            (float(item.means[0, 1]) - ego_location.y) ** 2))[:self._prediction_max_actors]
        snapshot.update({
            "decision_sim_time": float(timestamp),
            "state": self._state,
            "state_id": self._STATE_IDS.get(self._state, -1),
            "state_since_sim_time": float(self._state_since),
            "state_duration_s": float(max(timestamp - self._state_since, 0.0)),
            "reason": str(reason),
            "reason_code": self._REASON_IDS.get(str(reason), -1),
            "reason_codes": [self._REASON_IDS.get(str(reason), -1)],
            "target_speed_mps": float(max(target_speed_kmh, 0.0) / 3.6),
            "target_acceleration_mps2": float(self._last_target_accel_mps2),
            "speed_limit_mps": float(self._speed_limit / 3.6),
            "traffic_light_distance_m": (
                float(self._last_light_distance_m)
                if math.isfinite(self._last_light_distance_m) else None),
            "traffic_light_id": self._last_light_id,
            "route_plan": plan,
            "decision_ego_transform": self._transform_payload(transform),
            "prediction": {
                "horizon_s": self._prediction_horizon_s,
                "step_s": self._prediction_step_s,
                "tracked_actor_count": self._tracker.track_count,
                "valid_track_count": len(self._last_predictions),
                "actors": [{
                    "actor_id": item.actor_id, "actor_type": item.actor_type,
                    "observation_count": item.observation_count,
                    "speed_mps": item.speed_mps,
                    "extent_xy_m": list(item.extent_xy),
                    "times_s": item.times.tolist(),
                    "mean_world_xy": item.means.tolist(),
                    "current_position_world_xy": item.current_position,
                    "observed_velocity_world_xy": item.current_velocity,
                    "observed_heading_world_xy": item.current_heading_xy,
                    "static_box": item.static_box,
                    "covariance_world_xy": item.covariances.tolist(),
                    "intent_modes": [{
                        "intent": mode.name,
                        "probability": mode.probability,
                        "mean_world_xy": mode.means.tolist(),
                        "covariance_world_xy": mode.covariances.tolist(),
                    } for mode in item.modes],
                } for item in nearest],
            },
            "lane_change": {
                "available_left": self._lane_available(current_waypoint, "left"),
                "available_right": self._lane_available(current_waypoint, "right"),
                "cooldown_s": 0.0,
            },
            "longitudinal_plan": self._last_longitudinal_plan,
            "pedestrian_braking_plan": getattr(self, '_pedestrian_braking_plan', None),
            "maneuver_plan": copy.deepcopy(self._active_maneuver),
            "maneuver_safety": copy.deepcopy(self._maneuver_safety),
        })
        if option is not None:
            snapshot["navigation"] = {
                "road_option": int(option),
                "incoming_road_option": int(self._incoming_direction)
                if self._incoming_direction is not None else None,
                "target_waypoint": self._transform_payload(target_waypoint.transform)
                if target_waypoint is not None else None,
                "source": "local_planner",
                "assigned_route_index": (len(self._local_planner._initial_route['points']) -
                    len(self._local_planner.get_plan())) if getattr(self._local_planner, '_initial_route', None) else None,
                "recovery_command": 0,
            }
            if (self._active_maneuver is not None and
                    self._active_maneuver["phase"] == "recover"):
                snapshot["navigation"]["recovery_command"] = (
                    RECOVER_RIGHT if self._active_maneuver["side"] < 0
                    else RECOVER_LEFT)
        if control is not None:
            snapshot["applied_control"] = {
                "throttle": float(control.throttle), "steer": float(control.steer),
                "brake": float(control.brake), "hand_brake": bool(control.hand_brake),
                "reverse": bool(control.reverse),
            }
        self._last_snapshot = snapshot
        return snapshot

    def _finish_control(self, control, reason, timestamp, target_speed_kmh=None):
        if getattr(self, '_pedestrian_braking_plan', None) is not None:
            self._pedestrian_braking_plan['yield_actor_ids'] = sorted(self._pedestrian_yield_ids)
            self._pedestrian_braking_plan['corridor_clear_since_sim_time'] = self._pedestrian_clear_since
        moving_tracking = (getattr(self, '_moving_event', None) is not None or
            (getattr(self, '_moving_recovery', False) and self._active_maneuver is not None and
             self._active_maneuver['phase'] == 'recover'))
        if moving_tracking and self._active_maneuver is not None:
            if self._moving_vehicle_geometry is None:
                physics = self._vehicle.get_physics_control()
                wheels = physics.wheels
                # CARLA wheel positions can be expressed in world axes.
                # Taking only X collapses the wheelbase on north/south roads.
                wheelbase = wheelbase_from_positions([(wheel.position.x, wheel.position.y)
                                                     for wheel in wheels])
                angle = math.radians(max(wheel.max_steer_angle for wheel in wheels))
                self._moving_vehicle_geometry = (max(wheelbase, 1.), max(angle, .3),
                    [(point.x, point.y) for point in physics.steering_curve])
            wheelbase, angle, curve = self._moving_vehicle_geometry
            multiplier = float(np.interp(self._speed, [p[0] for p in curve], [p[1] for p in curve])) if curve else 1.
            pose = self._vehicle.get_transform()
            control.steer = pursuit_steering(self._active_maneuver['points'],
                [pose.location.x, pose.location.y], math.radians(pose.rotation.yaw),
                self._speed / 3.6, wheelbase, angle, multiplier, self._vehicle.get_control().steer)
            self._active_maneuver['tracking'] = dict(controller='curvature_pursuit',
                wheelbase_m=wheelbase, maximum_wheel_angle_rad=angle, steering_multiplier=multiplier,
                assumed_tracking_reserve_m=self._moving_evasion_tracking_reserve_m)
            self._local_planner._vehicle_controller.past_steering = control.steer
        target = (self._local_planner._target_speed if target_speed_kmh is None
                  else target_speed_kmh)
        # Only final committed states advance the exported state timer.
        # Intermediate FOLLOW/CRUISE assignments within this tick must not
        # reset a continuous avoidance/recovery episode on every frame.
        previous = getattr(self, '_last_snapshot', {})
        if previous.get('decision_sim_time') is None:
            self._state_since = timestamp
        elif previous.get('state') == self._state:
            self._state_since = previous.get('state_since_sim_time', self._state_since)
        self._build_snapshot(reason, target, timestamp, control)
        return control

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        while leaving the assigned navigation route unchanged.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

        if vehicle_state and vehicle is not None:
            # BasicAgent's angular branch returns distance between ego-front
            # and target-rear transforms. run_step subtracts body extents once,
            # so normalize here instead of subtracting both cars twice.
            distance = self._vehicle.get_location().distance(vehicle.get_location())
        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]
        # BasicAgent treats an empty list as a request to scan all vehicles.
        if not walker_list:
            return False, None, -1

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1e-3, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = max(0.0, min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                max(0.0, self._speed_limit - self._behavior.speed_lim_dist)]))
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = max(0.0, min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                max(0.0, self._speed_limit - self._behavior.speed_lim_dist)]))
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = max(0.0, min([
                self._behavior.max_speed,
                max(0.0, self._speed_limit - self._behavior.speed_lim_dist)]))
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def _predictive_target_speed(self, risk):
        """Translate route risk into an anticipatory speed target in km/h."""
        target = min(self._behavior.max_speed,
                     max(0.0, self._speed_limit - self._behavior.speed_lim_dist))
        max_risk = float(risk.get("max_risk", 0.0))
        ttc = float(risk.get("expected_ttc_s", float("inf")))
        if max_risk >= self._risk_stop_threshold or (ttc < 1.0 and max_risk > 0.55):
            return 0.0, "predicted_conflict"
        if max_risk >= self._risk_yield_threshold or (ttc < 2.0 and max_risk > 0.35):
            return min(target, max(self._min_speed, self._speed * 0.45)), "predicted_conflict"
        if max_risk >= self._risk_slow_threshold:
            return min(target, max(self._min_speed, self._speed * 0.75)), "comfortable_braking"
        return target, "cruise"

    def _corner_speed_constraints(self):
        """Return route-relative speed limits for tight bends ahead."""
        points = self._decision_route_plan["points"]
        if getattr(self, "_active_route_points", None):
            points = []
            arc = 0.
            previous = self._active_route_points[0]
            for location in self._active_route_points:
                arc += math.hypot(location[0] - previous[0], location[1] - previous[1])
                points.append({"location_world": location, "arc_m": arc})
                previous = location
        ego_speed = self._speed / 3.6
        buffer_m = ego_speed * self._reaction_time_s + 4.0
        constraints = []
        for index in range(1, len(points) - 1):
            middle = points[index]
            distance = middle["arc_m"]
            if distance > self._corner_preview_m:
                break
            before = points[index - 1]["location_world"]
            center = middle["location_world"]
            after = points[index + 1]["location_world"]
            ax, ay = center[0] - before[0], center[1] - before[1]
            bx, by = after[0] - center[0], after[1] - center[1]
            first = math.hypot(ax, ay)
            second = math.hypot(bx, by)
            chord = math.hypot(after[0] - before[0], after[1] - before[1])
            if min(first, second, chord) < 0.75:
                continue
            curvature = 2.0 * abs(ax * by - ay * bx) / (first * second * chord)
            if curvature < 0.025:
                continue
            corner_speed = math.sqrt(self._corner_lateral_accel_mps2 / curvature)
            approach_m = max(distance - buffer_m, 0.0)
            constraints.append((approach_m, corner_speed))
        return constraints

    def _corner_speed_limit_kmh(self, constraints=None):
        """Brake before a tight planned turn, using its three-point curvature."""
        if constraints is None:
            constraints = self._corner_speed_constraints()
        limit_mps = float("inf")
        for approach_m, corner_speed in constraints:
            allowed = math.sqrt(corner_speed ** 2 +
                                2.0 * self._comfort_decel_mps2 * approach_m)
            limit_mps = min(limit_mps, allowed)
        return limit_mps * 3.6

    def _tracking_speed_limit_kmh(self, ego_waypoint):
        """Hold a moderate speed until the vehicle is aligned with its lane."""
        if self._moving_path_ready():
            # The original lane's heading differs intentionally during an
            # admitted detour. Its generic alignment cap must not invalidate
            # the jointly checked speed profile. Curvature limits are already
            # part of this path's admission and remain applied below.
            return self._maneuver_safety['execution_speed_mps'] * 3.6, 'curve_speed'
        points = self._decision_route_plan["points"]
        if not points:
            return float("inf"), None
        transform = self._vehicle.get_transform()
        target_yaw = points[0]["rotation_world"][2]
        yaw_error = abs((transform.rotation.yaw - target_yaw + 180.0) % 360.0 - 180.0)
        if yaw_error > self._turn_alignment_threshold_deg:
            return self._turn_alignment_speed_mps * 3.6, "curve_speed"
        center = ego_waypoint.transform.location
        offset = math.hypot(transform.location.x - center.x,
                            transform.location.y - center.y)
        if offset > min(0.6, ego_waypoint.lane_width * 0.15):
            return self._lane_centering_speed_mps * 3.6, "lane_centering"
        return float("inf"), None

    def _route_stop_distance(self, stop, ego_waypoint):
        """Project a native stop waypoint onto matching lanes of the queued route."""
        location = self._vehicle.get_location()
        forward = stop.transform.get_forward_vector()
        stop_location = stop.transform.location
        if ((stop_location.x - location.x) * forward.x +
                (stop_location.y - location.y) * forward.y) <= 0.0:
            return None
        stop_key = (stop.road_id, stop.section_id, stop.lane_id)
        previous_xy = (location.x, location.y)
        previous_key = (ego_waypoint.road_id, ego_waypoint.section_id,
                        ego_waypoint.lane_id)
        previous_arc = 0.0
        nearest = None
        for point in self._decision_route_plan["points"]:
            current_xy = point["location_world"][:2]
            current_key = (point["road_id"], point["section_id"], point["lane_id"])
            arc = point["arc_m"]
            if stop_key == previous_key or stop_key == current_key:
                dx = current_xy[0] - previous_xy[0]
                dy = current_xy[1] - previous_xy[1]
                length_sq = dx * dx + dy * dy
                if length_sq > 1e-6:
                    along = min(max(((stop_location.x - previous_xy[0]) * dx +
                                     (stop_location.y - previous_xy[1]) * dy) /
                                    length_sq, 0.0), 1.0)
                    lateral = math.hypot(
                        stop_location.x - previous_xy[0] - along * dx,
                        stop_location.y - previous_xy[1] - along * dy)
                    if lateral <= stop.lane_width * 0.5:
                        projected = previous_arc + along * (arc - previous_arc)
                        if nearest is None or projected < nearest:
                            nearest = projected
            previous_xy, previous_key, previous_arc = current_xy, current_key, arc
        return nearest

    def _traffic_light_target(self, ego_waypoint):
        self._last_light_distance_m = float("inf")
        self._last_light_id = None
        if self._ignore_traffic_lights:
            return None, None
        speed = self._speed / 3.6
        threshold = max(self._route_plan_horizon_m,
                        self._base_tlight_threshold + speed * 3.0, 8.0)
        stop_distance = speed * self._reaction_time_s + speed ** 2 / max(
            2.0 * self._comfort_decel_mps2, 1e-3)
        chosen = None
        lights = self._world.get_actors().filter("*traffic_light*")
        ego_location = self._vehicle.get_location()
        for light in lights:
            if light.state not in (carla.TrafficLightState.Red,
                                   carla.TrafficLightState.Yellow):
                continue
            if light.get_location().distance(ego_location) > threshold + 20.0:
                continue
            stops = self._light_stop_cache.get(light.id)
            if stops is None:
                stops = light.get_stop_waypoints()
                self._light_stop_cache[light.id] = stops
            for stop in stops:
                distance = self._route_stop_distance(stop, ego_waypoint)
                if distance is None or distance > threshold:
                    continue
                if light.state == carla.TrafficLightState.Yellow \
                        and distance <= stop_distance + 2.0:
                    continue
                if chosen is None or distance < chosen[0]:
                    chosen = (distance, light, stop)
        if chosen is not None:
            self._last_light_distance_m = chosen[0]
            self._last_light_id = int(chosen[1].id)
            self._active_stop_waypoint = chosen[2]
            return 0.0, ("red_light" if chosen[1].state == carla.TrafficLightState.Red
                         else "yellow_light")
        if self._active_stop_light_id is not None:
            for light in lights:
                if light.id != self._active_stop_light_id or \
                        light.state != carla.TrafficLightState.Red:
                    continue
                stop = self._active_stop_waypoint
                if stop is None:
                    continue
                stop_location = stop.transform.location
                forward = stop.transform.get_forward_vector()
                ahead = ((stop_location.x - ego_location.x) * forward.x +
                         (stop_location.y - ego_location.y) * forward.y)
                distance = stop_location.distance(ego_location)
                if ahead > 0.0 and distance <= threshold + 3.0:
                    self._last_light_distance_m = distance
                    self._last_light_id = int(light.id)
                    return 0.0, "red_light"
        return None, None

    def run_step(self, debug=False):
        """Choose a state and speed before issuing a single control command."""
        world_snapshot = self._world.get_snapshot()
        timestamp = float(world_snapshot.timestamp.elapsed_seconds)
        dt = float(getattr(world_snapshot.timestamp, "delta_seconds", 0.1) or 0.1)
        self._update_information()
        self._last_longitudinal_plan = None
        self._actor_predictions(timestamp)
        self._update_maneuver()
        self._decision_route_plan = self._route_plan_snapshot(self._vehicle.get_transform())
        self._decision_navigation_option = self._direction
        self._decision_target_waypoint = self._local_planner.target_waypoint
        ego_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        risk = self._risk_assessment(timestamp, update_predictions=False)
        moving_ready = self._moving_path_ready()
        if moving_ready:
            # The trigger is governed by the admitted joint brake+steer
            # envelope. Apply ordinary longitudinal risk to every OTHER actor;
            # retain the unfiltered distribution/risk in the expert snapshot.
            risk = evaluate_route_risk(self._route_points(), [item for item in self._last_predictions
                if item.actor_id != self._maneuver_obstacle_id], self._vehicle.get_location(),
                self._speed / 3.6, route_width_m=self._vehicle.bounding_box.extent.y + self._risk_lateral_margin_m)
            self._maneuver_safety['control_risk_excluding_admitted_actor'] = risk
        front_plan = self._front_vehicle_plan()
        self._last_front_plan = dict(front_plan)
        target, reason = self._predictive_target_speed(risk)
        light_target, light_reason = self._traffic_light_target(ego_waypoint)
        if light_target is not None:
            speed_mps = self._speed / 3.6
            stop_distance = speed_mps * self._reaction_time_s + \
                speed_mps ** 2 / max(2.0 * self._comfort_decel_mps2, 1e-3)
            remaining = max(self._last_light_distance_m - 2.0 -
                            speed_mps * self._reaction_time_s, 0.0)
            stop_speed_kmh = math.sqrt(2.0 * self._comfort_decel_mps2 * remaining) * 3.6
            target = min(target, stop_speed_kmh)
            if (self._active_stop_light_id == self._last_light_id or
                    self._last_light_distance_m <= stop_distance + 3.0):
                self._active_stop_light_id = self._last_light_id
                can_creep = (light_reason == "red_light" and
                             self._last_light_distance_m > self._stop_line_margin_m + 1.0 and
                             risk.get("max_risk", 0.0) < self._risk_yield_threshold and
                             (self._creep_light_id == self._last_light_id or
                              (speed_mps < 0.3 and
                               risk.get("max_risk", 0.0) < self._risk_slow_threshold)))
                if can_creep:
                    self._creep_light_id = self._last_light_id
                    target = min(target, self._creep_speed_mps * 3.6)
                    self._last_target_speed_mps = min(
                        self._last_target_speed_mps, self._creep_speed_mps)
                    self._last_target_accel_mps2 = min(
                        self._last_target_accel_mps2, 0.0)
                    reason = "red_light_creep"
                    self._set_state("CREEP", timestamp)
                else:
                    self._creep_light_id = None
                    reason = light_reason
                    self._set_state("STOPPING", timestamp)
                    if speed_mps < 0.5 or \
                            self._last_light_distance_m <= self._stop_line_margin_m + 1.0:
                        target = 0.0
            else:
                reason = light_reason + "_approach"
                self._set_state("APPROACH_STOP", timestamp)
            if self._state == "STOPPING" and \
                    self._last_light_distance_m < max(3.0, speed_mps):
                self._last_target_speed_mps = 0.0
                self._last_target_accel_mps2 = 0.0
                return self._finish_control(self.emergency_stop(), reason, timestamp, 0.0)
        else:
            self._active_stop_light_id = None
            self._active_stop_waypoint = None
            self._creep_light_id = None

        pedestrian_gap = float(risk.get("longitudinal_gap_m", float("inf")))
        pedestrian_ttc = float(risk.get("expected_ttc_s", float("inf")))
        speed_mps = self._speed / 3.6
        soft_pedestrian_stop = False
        pedestrians = [item for item in self._last_predictions if item.actor_type == 'pedestrian' and
                       not (moving_ready and item.actor_id == self._maneuver_obstacle_id)]
        if pedestrians:
            bounds = self._vehicle.bounding_box.extent
            self._pedestrian_braking_plan = swept_motion_check(self._route_points(), pedestrians,
                speed_mps, (bounds.x, bounds.y), 0., decel_mps2=self._comfort_decel_mps2,
                horizon_s=self._moving_evasion_horizon_s, margin_m=self._moving_evasion_margin_m,
                allow_unobserved_braking=True)
        else:
            self._pedestrian_braking_plan = None
        bounds = self._vehicle.bounding_box.extent
        location = self._vehicle.get_location()
        threats = moving_threats(self._route_points(), pedestrians,
            (location.x, location.y),
            speed_mps, (bounds.x, bounds.y), self._moving_evasion_horizon_s,
            self._moving_evasion_margin_m)
        # A newly appearing crossing target needs immediate stable yielding.
        # Established tracks continue using their joint speed/escape plan;
        # a six-second threat alone is not an instruction to stop completely.
        self._pedestrian_yield_ids.update(threat['actor'].actor_id for threat in threats
                                         if threat['actor'].observation_count <= 2)
        retained = [item for item in pedestrians if item.actor_id in self._pedestrian_yield_ids and
                    pedestrian_blocks_route(self._route_points(), item, (bounds.x, bounds.y),
                                            self._moving_evasion_margin_m)]
        if retained:
            self._pedestrian_clear_since = None
        elif self._pedestrian_clear_since is None:
            self._pedestrian_clear_since = timestamp
        elif timestamp - self._pedestrian_clear_since >= .3:
            self._pedestrian_yield_ids.clear()
        if self._pedestrian_yield_ids and pedestrians:
            target, reason, soft_pedestrian_stop = 0., 'pedestrian_risk', True
            self._set_state('YIELD', timestamp)
        emergency_distance = (speed_mps * 0.3 + speed_mps ** 2 /
                              max(2.0 * self._comfort_decel_mps2, 1e-3) + 4.0)
        if (risk.get("closest_actor_type") == "pedestrian" and
                risk.get("max_risk", 0.0) >= self._risk_slow_threshold and
                pedestrian_ttc <= 2.0 and
                0.0 <= pedestrian_gap <= emergency_distance):
            self._pedestrian_yield_ids.update(item.actor_id for item in pedestrians
                if item.actor_id == risk.get('closest_actor_id'))
            if not (self._pedestrian_braking_plan or {}).get('safe'):
                self._set_state("STOPPING", timestamp)
                self._last_target_speed_mps = 0.0
                self._last_target_accel_mps2 = 0.0
                return self._finish_control(self.emergency_stop(), "pedestrian_risk", timestamp, 0.0)
            target, reason, soft_pedestrian_stop = 0., 'pedestrian_risk', True
            self._set_state('YIELD', timestamp)

        walker_state, walker, walker_distance = self.pedestrian_avoid_manager(ego_waypoint)
        if walker_state and not (moving_ready and walker.id == self._maneuver_obstacle_id):
            self._pedestrian_yield_ids.add(walker.id)
            gap = walker_distance - max(walker.bounding_box.extent.x,
                                        walker.bounding_box.extent.y) - max(
                                            self._vehicle.bounding_box.extent.x,
                                            self._vehicle.bounding_box.extent.y)
            if gap < self._behavior.braking_distance and not (self._pedestrian_braking_plan or {}).get('safe'):
                self._set_state("STOPPING", timestamp)
                self._last_target_speed_mps = 0.0
                self._last_target_accel_mps2 = 0.0
                return self._finish_control(self.emergency_stop(), "pedestrian_risk",
                                            timestamp, 0.0)
            target = min(target, max(0.0, self._speed * 0.4))
            reason = "pedestrian_risk"
            self._set_state("YIELD", timestamp)
            soft_pedestrian_stop = bool((self._pedestrian_braking_plan or {}).get('safe'))

        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_waypoint)
        if (vehicle_state and self._active_maneuver is not None and
                self._active_maneuver["phase"] == "avoid" and
                vehicle.id == self._maneuver_obstacle_id):
            vehicle_state = False
        if vehicle_state:
            gap = distance - max(vehicle.bounding_box.extent.x,
                                 vehicle.bounding_box.extent.y) - max(
                                     self._vehicle.bounding_box.extent.x,
                                     self._vehicle.bounding_box.extent.y)
            if gap < self._behavior.braking_distance:
                self._set_state("STOPPING", timestamp)
                self._last_target_speed_mps = 0.0
                self._last_target_accel_mps2 = 0.0
                # Holding a stopped car behind traffic is following behavior,
                # not a new emergency braking event.
                stop_reason = ("follow_vehicle" if speed_mps < 0.5
                               else "emergency_stop")
                return self._finish_control(self.emergency_stop(), stop_reason,
                                            timestamp, 0.0)
            follow_speed = min(max(0.0, get_speed(vehicle)),
                               max(0.0, gap / max(self._behavior.safety_time, 0.1) * 3.6))
            target = min(target, follow_speed)
            if light_target is None and reason == "cruise":
                reason = "follow_vehicle"
            if light_target is None:
                self._set_state("FOLLOW", timestamp)
        elif light_target is None and reason in ("predicted_conflict", "pedestrian_risk"):
            self._set_state("YIELD", timestamp)
        elif light_target is None and reason == "comfortable_braking":
            self._set_state("YIELD", timestamp)
        elif light_target is None and self._direction in (
                RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            if reason == "cruise":
                reason = "lane_change_safe"
            self._set_state("LANE_CHANGE", timestamp)
        elif light_target is None:
            self._set_state("CRUISE", timestamp)

        if front_plan["emergency"]:
            self._set_state("STOPPING", timestamp)
            self._last_target_speed_mps = 0.0
            self._last_target_accel_mps2 = 0.0
            stop_reason = ("emergency_stop" if speed_mps >= .5 else
                           "comfortable_braking" if front_plan['kind'] == 'static' else
                           "follow_vehicle" if front_plan["kind"] == "following" else
                           "predicted_conflict")
            return self._finish_control(self.emergency_stop(), stop_reason,
                                        timestamp, 0.0)
        if light_target is None and self._active_maneuver is not None:
            if self._active_maneuver["phase"] == "avoid":
                if self._active_maneuver["side"] > 0:
                    self._set_state("AVOID_LEFT", timestamp)
                    reason = "obstacle_avoid_left"
                else:
                    self._set_state("AVOID_RIGHT", timestamp)
                    reason = "obstacle_avoid_right"
            elif self._active_maneuver["side"] > 0:
                self._set_state("RECOVER_LEFT", timestamp)
                reason = "route_recover_left"
            else:
                self._set_state("RECOVER_RIGHT", timestamp)
                reason = "route_recover_right"
        front_target_kmh = front_plan["speed_cap_mps"] * 3.6
        if front_target_kmh < target:
            target = front_target_kmh
            if light_target is None and reason in (
                    "cruise", "comfortable_braking"):
                if front_plan["kind"] == "crossing":
                    reason = "predicted_conflict"
                    self._set_state("YIELD", timestamp)
                elif front_plan['kind'] == 'static':
                    reason = 'comfortable_braking'
                    self._set_state('APPROACH_STOP', timestamp)
                else:
                    reason = "follow_vehicle"
                    self._set_state("FOLLOW", timestamp)

        if self._incoming_waypoint is not None and self._incoming_waypoint.is_junction \
                and self._incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT):
            target = min(target, max(0.0, self._speed_limit - 5.0))
        speed_constraints = []
        pedestrian_coast_safe = False
        if soft_pedestrian_stop:
            target = 0.
            self._last_target_accel_mps2 = min(self._last_target_accel_mps2, 0.)
            self._last_target_speed_mps = min(self._last_target_speed_mps, speed_mps)
            distances = [project_route(self._route_points(), item.current_position or item.means[0])[0]
                         - self._vehicle.bounding_box.extent.x - item.extent_xy[0] - self._front_min_gap_m
                         for item in pedestrians]
            if distances:
                speed_constraints.append((max(min(distances), 0.), 0.))
            coast = swept_motion_check(self._route_points(), pedestrians, speed_mps,
                (self._vehicle.bounding_box.extent.x, self._vehicle.bounding_box.extent.y), 0.,
                decel_mps2=self._control_coast_decel_mps2,
                horizon_s=self._moving_evasion_horizon_s, margin_m=self._moving_evasion_margin_m,
                allow_unobserved_braking=True)
            pedestrian_coast_safe = coast['safe']
            self._pedestrian_braking_plan['coasting'] = coast
        if self._maneuver_wait_distance_m is not None:
            speed_constraints.append((self._maneuver_wait_distance_m, 0.))
        if light_target is not None and reason != "red_light_creep":
            speed_constraints.append((
                max(self._last_light_distance_m - 2.0, 0.0), 0.0))
        if (front_plan["kind"] == "crossing" and
                front_plan.get("crossing_constraint", True)):
            speed_constraints.append((
                max(float(front_plan["gap_m"]) - self._front_min_gap_m,
                    0.0), 0.0))
        elif front_plan["kind"] in ("following", "static"):
            lead = next((item for item in self._last_predictions
                         if item.actor_id == front_plan["actor_id"]), None)
            if lead is not None and lead.speed_mps < 0.5:
                speed_constraints.append((
                    max(float(front_plan["gap_m"]) - self._front_min_gap_m,
                        0.0), 0.0))
        corner_constraints = self._corner_speed_constraints()
        speed_constraints.extend(corner_constraints)
        corner_target = self._corner_speed_limit_kmh(corner_constraints)
        if corner_target < target:
            target = corner_target
            if reason == "cruise":
                reason = "curve_speed"
        tracking_target, tracking_reason = self._tracking_speed_limit_kmh(ego_waypoint)
        if self._active_maneuver is not None:
            target = min(target, (self._maneuver_safety or {}).get(
                'execution_speed_mps', self._maneuver_speed_mps) * 3.6)
        if tracking_target < target:
            target = tracking_target
            if reason in ("cruise", "curve_speed"):
                reason = tracking_reason
        waiting_coast_limit = None
        blocked_moving_stop = False
        if light_target is None and self._maneuver_safety is not None and \
                self._maneuver_safety.get("waiting_for_gap"):
            if self._maneuver_safety.get("phase") == "avoid":
                # Waiting for lateral admission is not permission to consume
                # the remaining entry corridor. Stay stopped when already
                # waiting, instead of accelerating toward a moving stop point.
                moving_wait = self._moving_admission or self._moving_event is not None
                checks = ([self._maneuver_safety] if 'geometry' in self._maneuver_safety else
                          self._maneuver_safety.get('candidates', []))
                transient = any(check.get('geometry', {}).get('safe') and check.get('blocking_actors') and
                    not any(block['kind'] in ('parked_on_path', 'static_on_path')
                            for block in check.get('blocking_actors', [])) for check in checks)
                target = min(target, self._maneuver_speed_mps * 3.6) if moving_wait and transient else 0.
                blocked_moving_stop = moving_wait and not transient
                if blocked_moving_stop:
                    self._last_target_accel_mps2 = min(self._last_target_accel_mps2, 0.)
                    self._last_target_speed_mps = min(self._last_target_speed_mps, self._speed / 3.6)
            elif self._maneuver_wait_distance_m is not None:
                waiting_coast_limit = waiting_coast_speed(
                    self._maneuver_wait_distance_m, self._control_coast_decel_mps2)
                target = min(target, waiting_coast_limit * 3.6)
            else:
                target = min(target, self._turn_alignment_speed_mps * 3.6)
            waiting_state, reason = waiting_annotation(
                self._maneuver_safety, self._state, reason,
                real_conflict=(float(risk.get('max_risk', 0.)) >= self._risk_yield_threshold or
                               (front_plan['kind'] == 'crossing' and
                                front_plan.get('crossing_constraint', True))))
            self._set_state(waiting_state, timestamp)
        # Route completion is a soft navigation goal, not an unexpected
        # obstacle. Approach it before LocalPlanner empties its queue and
        # otherwise issues a full brake at cruising speed.
        traffic_constraints = list(speed_constraints)
        goal_distance = self._navigation_goal_distance()
        goal_limit = None
        goal_approach_active = False
        if goal_distance is not None:
            stop_distance = max(goal_distance - self._navigation_goal_stop_buffer_m, 0.)
            speed_constraints.append((stop_distance, 0.))
            goal_limit = stop_distance / self._navigation_goal_approach_s
            if goal_limit * 3.6 < target:
                goal_approach_active = True
                target = goal_limit * 3.6
                if reason in ('cruise', 'curve_speed') and self._speed / 3.6 >= goal_limit - .1:
                    reason = 'comfortable_braking'
                    self._set_state('APPROACH_STOP' if goal_limit > 0. else 'STOPPING', timestamp)
        if target <= 0.0 and self._speed / 3.6 < 0.5:
            if goal_distance is not None and goal_distance < 1.:
                self._local_planner.set_speed(0.)
                self._local_planner.run_step(debug=debug)
            self._last_target_speed_mps = 0.0
            self._last_target_accel_mps2 = 0.0
            return self._finish_control(self.emergency_stop(), reason, timestamp, 0.0)
        stop_for_conflict = reason == "predicted_conflict" and target <= 0.0
        policy_speed = target / 3.6
        target = self._plan_target_speed(target, speed_constraints, dt)
        if moving_ready:
            ceiling = max(speed_mps, self._maneuver_safety['execution_speed_mps'])
            target = min(target, ceiling * 3.6)
            self._last_target_speed_mps = target / 3.6
            self._last_longitudinal_plan['target_speed_mps'] = target / 3.6
        self._local_planner.set_speed(target)
        control = self._local_planner.run_step(
            debug=debug,
            target_location_override=(
                self._active_maneuver["target_world_xy"]
                if self._active_maneuver is not None else None))
        if self._state == "STOPPING" or stop_for_conflict or soft_pedestrian_stop:
            control.throttle = 0.0
        if waiting_coast_limit is not None and self._speed / 3.6 >= waiting_coast_limit:
            # The jerk-limited issued target may lag this approaching stop.
            # Release throttle before the available coasting tail is consumed.
            control.throttle = 0.
        if goal_limit is not None and self._speed / 3.6 >= goal_limit:
            control.throttle = 0.
        soft_terminal_stop = (goal_limit is not None and front_plan['kind'] is None and
                              not vehicle_state and light_target is None and
                              self._active_maneuver is None and
                              float(risk.get('max_risk', 0.)) < self._risk_slow_threshold and
                              reason in ('cruise', 'comfortable_braking', 'curve_speed'))
        safe_maneuver_tracking = (self._active_maneuver is not None and
                                  (self._maneuver_safety or {}).get('ready', False) and
                                  front_plan['kind'] is None and not vehicle_state and
                                  light_target is None and
                                  float(risk.get('max_risk', 0.)) < self._risk_slow_threshold)
        moving_wait_coast = False
        if blocked_moving_stop and .3 < self._speed / 3.6 <= self._control_coast_speed_mps:
            stop_check = swept_motion_check(self._route_points(), self._last_predictions,
                self._speed / 3.6, (self._vehicle.bounding_box.extent.x, self._vehicle.bounding_box.extent.y),
                0., decel_mps2=self._control_coast_decel_mps2,
                horizon_s=max(self._moving_evasion_horizon_s, self._speed / 3.6 / self._control_coast_decel_mps2 + 1.),
                margin_m=self._moving_evasion_margin_m,
                ego_heading=math.radians(self._vehicle.get_transform().rotation.yaw))
            moving_wait_coast = stop_check['safe']
            self._maneuver_safety['coasting_stop_check'] = stop_check
        if control.brake > 0. and allow_coasting(
                self._speed / 3.6, target / 3.6, policy_speed, traffic_constraints,
                stop_for_conflict=stop_for_conflict,
                wait_distance=self._maneuver_wait_distance_m,
                coast_speed_mps=self._control_coast_speed_mps,
                coast_decel_mps2=self._control_coast_decel_mps2,
                soft_navigation_goal=soft_terminal_stop,
                safe_maneuver_tracking=safe_maneuver_tracking or pedestrian_coast_safe or moving_wait_coast):
            control.brake = 0.
        if (moving_ready and 0. < control.brake < .1 and
                light_target is None and not vehicle_state and front_plan['kind'] is None and
                speed_mps - target / 3.6 < 1.):
            # Tiny PID tracking brakes can produce a large transient in CARLA
            # during steering. Release only after proving the slower natural
            # deceleration envelope against every actor on the admitted path.
            coast_check = swept_motion_check(self._active_maneuver['points'], self._last_predictions,
                speed_mps, (bounds.x, bounds.y), self._maneuver_safety['execution_speed_mps'],
                decel_mps2=self._control_coast_decel_mps2,
                horizon_s=self._moving_evasion_horizon_s, margin_m=self._moving_evasion_margin_m,
                tracking_margin_m=self._moving_evasion_tracking_reserve_m,
                ego_heading=math.radians(self._vehicle.get_transform().rotation.yaw),
                settled_follower_ids=self._maneuver_safety.get('settled_follower_ids', ()),
                hard_mode_probability=self._moving_evasion_hard_mode_probability)
            self._maneuver_safety['tracking_coast_check'] = coast_check
            if coast_check['safe']:
                control.brake = 0.
        control.throttle = limit_launch_throttle(
            control.throttle, self._vehicle.get_control().throttle, dt,
            self._speed / 3.6, rate_per_s=self._control_launch_throttle_rate,
            low_speed_mps=self._control_coast_speed_mps,
            launch_throttle_max=self._control_launch_throttle_max)
        if waiting_coast_limit is not None:
            # Creep power must shrink with the waiting speed. Repeatedly
            # launching at full low-speed throttle defeats a gentle stop tail.
            control.throttle = min(control.throttle, self._control_launch_throttle_max *
                                   min(waiting_coast_limit / 1.5, 1.))
        if goal_limit is not None:
            if (soft_terminal_stop and self._local_planner.done() and
                    self._speed / 3.6 <= self._control_coast_speed_mps):
                control.throttle = 0.
                control.brake = 0. if self._speed / 3.6 > .3 else self._max_brake
                control.steer = self._vehicle.get_control().steer
        if self._last_longitudinal_plan is not None:
            self._last_longitudinal_plan['navigation_goal_distance_m'] = goal_distance
            self._last_longitudinal_plan['navigation_goal_speed_mps'] = goal_limit
            self._last_longitudinal_plan['navigation_goal_active'] = goal_approach_active
        return self._finish_control(control, reason, timestamp)

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
