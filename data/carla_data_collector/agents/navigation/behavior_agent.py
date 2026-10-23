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
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal
from agents.navigation.behavior_prediction import BayesianActorTracker
from agents.navigation.risk_field import evaluate_route_risk, plan_front_vehicle_speed
from agents.navigation.longitudinal_planner import plan_longitudinal_speed

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
    }
    _REASON_IDS = {
        "cruise": 0, "follow_vehicle": 1, "predicted_conflict": 2,
        "pedestrian_risk": 3, "red_light": 4, "yellow_light": 5,
        "comfortable_braking": 6, "lane_change_safe": 7,
        "emergency_stop": 8, "red_light_approach": 9,
        "yellow_light_approach": 10, "red_light_creep": 11,
        "curve_speed": 12, "lane_centering": 13,
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
        planner_options.setdefault("base_min_distance",
                                   float(config.get("planner_base_min_distance_m", 1.5)))
        planner_options.setdefault("distance_ratio",
                                   float(config.get("planner_distance_ratio", 0.25)))
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
        self._lane_centering_speed_mps = float(
            config.get("lane_centering_speed_mps", 5.0))
        self._tracker = BayesianActorTracker(
            horizon_s=self._prediction_horizon_s,
            step_s=self._prediction_step_s,
            process_noise=float(config.get("prediction_process_noise", 1.5)),
            measurement_noise=float(config.get("prediction_measurement_noise", 0.25)),
            max_stale_s=float(config.get("prediction_max_stale_s", 1.0)),
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
        self._speed = get_speed(self._vehicle)
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
        if not self._predictive_enabled:
            self._last_predictions = []
            return []
        actors = list(self._world.get_actors().filter("*vehicle*"))
        actors.extend(list(self._world.get_actors().filter("*walker.pedestrian*")))
        self._last_predictions = self._tracker.update(
            actors, timestamp, ego_id=self._vehicle.id)
        return self._last_predictions

    def _route_points(self):
        points = [self._vehicle.get_location()]
        travelled = 0.0
        for waypoint, _ in list(self._local_planner.get_plan()):
            next_location = waypoint.transform.location
            travelled += points[-1].distance(next_location)
            points.append(next_location)
            if travelled >= self._route_plan_horizon_m:
                break
        return [[float(point.x), float(point.y)] for point in points]

    def _risk_assessment(self, timestamp):
        predictions = self._actor_predictions(timestamp)
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
        return plan_front_vehicle_speed(
            self._route_points(), self._last_predictions,
            self._vehicle.get_location(), self._speed / 3.6,
            bounds.x, bounds.y, self._risk_lateral_margin_m,
            self._reaction_time_s, self._comfort_decel_mps2,
            minimum_gap_m=self._front_min_gap_m,
            confidence_z=self._front_confidence_z,
            lookahead_s=self._front_lookahead_s,
            emergency_decel_mps2=self._front_emergency_decel_mps2)

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
        desired_accel = min(max((preview_speed - previous_speed) / dt_s,
                                -self._comfort_decel_mps2),
                            self._max_accel_mps2)
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
                    "speed_mps": item.speed_mps,
                    "extent_xy_m": list(item.extent_xy),
                    "times_s": item.times.tolist(),
                    "mean_world_xy": item.means.tolist(),
                    "covariance_world_xy": item.covariances.tolist(),
                } for item in nearest],
            },
            "lane_change": {
                "available_left": self._lane_available(current_waypoint, "left"),
                "available_right": self._lane_available(current_waypoint, "right"),
                "cooldown_s": 0.0,
            },
            "longitudinal_plan": self._last_longitudinal_plan,
        })
        if option is not None:
            snapshot["navigation"] = {
                "road_option": int(option),
                "incoming_road_option": int(self._incoming_direction)
                if self._incoming_direction is not None else None,
                "target_waypoint": self._transform_payload(target_waypoint.transform)
                if target_waypoint is not None else None,
                "source": "local_planner",
            }
        if control is not None:
            snapshot["applied_control"] = {
                "throttle": float(control.throttle), "steer": float(control.steer),
                "brake": float(control.brake), "hand_brake": bool(control.hand_brake),
                "reverse": bool(control.reverse),
            }
        self._last_snapshot = snapshot
        return snapshot

    def _finish_control(self, control, reason, timestamp, target_speed_kmh=None):
        target = (self._local_planner._target_speed if target_speed_kmh is None
                  else target_speed_kmh)
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
        points = self._decision_route_plan["points"]
        if not points:
            return float("inf"), None
        transform = self._vehicle.get_transform()
        target_yaw = points[0]["rotation_world"][2]
        yaw_error = abs((transform.rotation.yaw - target_yaw + 180.0) % 360.0 - 180.0)
        if yaw_error > 12.0:
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
        self._decision_route_plan = self._route_plan_snapshot(self._vehicle.get_transform())
        self._decision_navigation_option = self._direction
        self._decision_target_waypoint = self._local_planner.target_waypoint
        ego_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        risk = self._risk_assessment(timestamp)
        front_plan = self._front_vehicle_plan()
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
        emergency_distance = (speed_mps * 0.3 + speed_mps ** 2 /
                              max(2.0 * self._comfort_decel_mps2, 1e-3) + 4.0)
        if (risk.get("closest_actor_type") == "pedestrian" and
                risk.get("max_risk", 0.0) >= self._risk_slow_threshold and
                pedestrian_ttc <= 2.0 and
                0.0 <= pedestrian_gap <= emergency_distance):
            self._set_state("STOPPING", timestamp)
            self._last_target_speed_mps = 0.0
            self._last_target_accel_mps2 = 0.0
            return self._finish_control(self.emergency_stop(), "pedestrian_risk",
                                        timestamp, 0.0)

        walker_state, walker, walker_distance = self.pedestrian_avoid_manager(ego_waypoint)
        if walker_state:
            gap = walker_distance - max(walker.bounding_box.extent.x,
                                        walker.bounding_box.extent.y) - max(
                                            self._vehicle.bounding_box.extent.x,
                                            self._vehicle.bounding_box.extent.y)
            if gap < self._behavior.braking_distance:
                self._set_state("STOPPING", timestamp)
                self._last_target_speed_mps = 0.0
                self._last_target_accel_mps2 = 0.0
                return self._finish_control(self.emergency_stop(), "pedestrian_risk",
                                            timestamp, 0.0)
            target = min(target, max(0.0, self._speed * 0.4))
            reason = "pedestrian_risk"
            self._set_state("YIELD", timestamp)

        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_waypoint)
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
        elif light_target is None and reason == "predicted_conflict":
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
            return self._finish_control(self.emergency_stop(), "emergency_stop",
                                        timestamp, 0.0)
        front_target_kmh = front_plan["speed_cap_mps"] * 3.6
        if front_target_kmh < target:
            target = front_target_kmh
            if light_target is None and reason in (
                    "cruise", "comfortable_braking"):
                if front_plan["kind"] == "crossing":
                    reason = "predicted_conflict"
                    self._set_state("YIELD", timestamp)
                else:
                    reason = "follow_vehicle"
                    self._set_state("FOLLOW", timestamp)

        if self._incoming_waypoint is not None and self._incoming_waypoint.is_junction \
                and self._incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT):
            target = min(target, max(0.0, self._speed_limit - 5.0))
        speed_constraints = []
        if light_target is not None and reason != "red_light_creep":
            speed_constraints.append((
                max(self._last_light_distance_m - 2.0, 0.0), 0.0))
        if front_plan["kind"] == "crossing":
            speed_constraints.append((
                max(float(front_plan["gap_m"]) - self._front_min_gap_m,
                    0.0), 0.0))
        elif front_plan["kind"] == "following":
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
        if tracking_target < target:
            target = tracking_target
            if reason in ("cruise", "curve_speed"):
                reason = tracking_reason
        if target <= 0.0 and self._speed / 3.6 < 0.5:
            self._last_target_speed_mps = 0.0
            self._last_target_accel_mps2 = 0.0
            return self._finish_control(self.emergency_stop(), reason, timestamp, 0.0)
        stop_for_conflict = reason == "predicted_conflict" and target <= 0.0
        target = self._plan_target_speed(target, speed_constraints, dt)
        self._local_planner.set_speed(target)
        control = self._local_planner.run_step(debug=debug)
        if self._state == "STOPPING" or stop_for_conflict:
            control.throttle = 0.0
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
