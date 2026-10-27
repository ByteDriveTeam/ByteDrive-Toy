"""Small online Bayesian motion predictor for CARLA actors.

The collector runs in the CARLA 0.9.15 Python 3.7 environment, so this module
intentionally uses only NumPy.  It is a constant-acceleration Kalman filter,
not a learned predictor: the posterior covariance grows with the forecast
horizon and is consumed by ``risk_field``.
"""

from dataclasses import dataclass
from collections import deque
import math

import numpy as np
from agents.navigation.intent_prediction import (
    infer_intent_probabilities, predict_intent_modes, motion_lane_context, IntentMode)


@dataclass
class ActorPrediction:
    actor_id: int
    actor_type: str
    times: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    speed_mps: float
    extent_xy: tuple
    modes: tuple = ()
    current_position: tuple = None
    current_velocity: tuple = None
    static_box: dict = None
    observation_count: int = 0
    current_heading_xy: tuple = None

    def for_mode(self, mode):
        """Return a legacy-shaped single hypothesis for risk calculations."""
        return ActorPrediction(
            actor_id=self.actor_id, actor_type=self.actor_type,
            times=self.times, means=mode.means,
            covariances=mode.covariances,
            speed_mps=mode.speed_mps, extent_xy=self.extent_xy,
            current_position=self.current_position,
            current_velocity=self.current_velocity, static_box=self.static_box,
            observation_count=self.observation_count, current_heading_xy=self.current_heading_xy)


class _Track:
    def __init__(self, actor_id, actor_type, process_noise, measurement_noise,
                 extent_xy):
        self.actor_id = int(actor_id)
        self.actor_type = str(actor_type)
        self.process_noise = float(max(process_noise, 1e-6))
        self.measurement_noise = float(max(measurement_noise, 1e-6))
        self.extent_xy = (float(extent_xy[0]), float(extent_xy[1]))
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self.timestamp = None
        self.last_seen = None
        self.last_observed_position = None
        self.last_observed_velocity = None
        self.lane_context = None
        self.intent_active = False
        self.intent_probabilities = {}
        self.motion_acceleration_mps2 = 0.0
        self.lateral_speed_mps = 0.0
        self.observation_count = 0
        self.initialized = False
        self.position_history = deque(maxlen=11)
        self.current_heading_xy = None

    @staticmethod
    def _transition(dt):
        dt = float(max(dt, 1e-3))
        return np.array([
            [1, 0, dt, 0, .5 * dt * dt, 0],
            [0, 1, 0, dt, 0, .5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

    def _process_covariance(self, dt):
        # White jerk noise.  This keeps the posterior smooth while allowing a
        # vehicle to change speed without pretending the future is exact.
        q = self.process_noise
        d = float(max(dt, 1e-3))
        q2 = q * q
        block = np.array([
            [d ** 5 / 20.0, d ** 4 / 8.0, d ** 3 / 6.0],
            [d ** 4 / 8.0, d ** 3 / 3.0, d ** 2 / 2.0],
            [d ** 3 / 6.0, d ** 2 / 2.0, d],
        ]) * q2
        out = np.zeros((6, 6), dtype=np.float64)
        out[np.ix_([0, 2, 4], [0, 2, 4])] = block
        out[np.ix_([1, 3, 5], [1, 3, 5])] = block
        return out

    def update(self, position, velocity, timestamp, lane_context=None, heading_xy=None):
        if heading_xy is not None:
            self.current_heading_xy = tuple(heading_xy)
        position = np.asarray(position, dtype=np.float64)[:2]
        velocity = np.asarray(velocity, dtype=np.float64)[:2]
        timestamp = float(timestamp)
        if not self.initialized or self.timestamp is None or timestamp <= self.timestamp:
            self.position_history.clear()
            self.position_history.append((timestamp, position.copy()))
            self.x[:] = 0.0
            self.x[:2] = position
            self.x[2:4] = velocity
            self.P[:] = np.eye(6, dtype=np.float64)
            self.P[:2, :2] *= self.measurement_noise
            self.P[2:4, 2:4] *= max(self.measurement_noise, 0.25)
            self.P[4:, 4:] *= 4.0
            self.timestamp = timestamp
            self.last_seen = timestamp
            self.last_observed_position = position.copy()
            self.last_observed_velocity = velocity.copy()
            self.lane_context = lane_context
            self.intent_probabilities = {}
            self.motion_acceleration_mps2 = 0.0
            self.lateral_speed_mps = 0.0
            self.observation_count = 1
            self.initialized = True
            return

        dt = min(max(timestamp - self.timestamp, 1e-3), 1.0)
        if self.actor_type == "pedestrian" and self.last_observed_position is not None:
            self.observation_count += 1
            # CARLA walkers can report near-zero actor velocity while their
            # controller moves them across the road. Consecutive ground-truth
            # positions are the reliable motion observation in that case.
            velocity = (position - self.last_observed_position) / dt
            walk_speed = float(np.linalg.norm(velocity))
            if walk_speed > 4.5:
                velocity *= 4.5 / walk_speed
        if self.actor_type == "vehicle":
            old_speed = float(np.linalg.norm(self.last_observed_velocity))
            new_speed = float(np.linalg.norm(velocity))
            acceleration = float(np.clip((new_speed - old_speed) / dt,
                                         -5.0, 4.0))
            self.motion_acceleration_mps2 = (
                0.6 * self.motion_acceleration_mps2 + 0.4 * acceleration)
            if lane_context is not None:
                direction = np.asarray(lane_context["direction"], dtype=np.float64)
                normal = np.asarray([-direction[1], direction[0]])
                observed_lateral = float(np.clip(np.dot(
                    position - self.last_observed_position, normal) / dt,
                    -2.5, 2.5))
                self.lateral_speed_mps = (
                    0.5 * self.lateral_speed_mps + 0.5 * observed_lateral)
            else:
                self.lateral_speed_mps = 0.0
            self.observation_count += 1
            self.intent_probabilities = infer_intent_probabilities(
                self.intent_probabilities, new_speed,
                self.motion_acceleration_mps2, self.lateral_speed_mps,
                lane_context, self.observation_count)
        z = np.concatenate((position, velocity))
        F = self._transition(dt)
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self._process_covariance(dt)
        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 0], H[1, 1], H[2, 2], H[3, 3] = 1.0, 1.0, 1.0, 1.0
        R = np.eye(4, dtype=np.float64) * self.measurement_noise
        R[2:, 2:] *= 0.5
        innovation = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        try:
            gain = np.linalg.solve(S, H.dot(self.P)).T
        except np.linalg.LinAlgError:
            gain = self.P.dot(H.T).dot(np.linalg.pinv(S))
        self.x = self.x + gain.dot(innovation)
        # Simulator positions are exact at the current tick. Anchor the mean
        # to the observation so an old velocity prior cannot leave it behind.
        self.x[:2] = position
        self.x[2:4] = velocity
        if self.actor_type == "pedestrian":
            self.x[4:6] = 0.
        else:
            # Ground-truth simulator velocity is an exact current condition.
            # A long stationary history must not leave a spurious CA transient
            # after an oncoming actor starts moving. Retain the filtered
            # covariance and history-derived acceleration/intent posterior.
            self.position_history.append((timestamp, position.copy()))
            # Fit ground-truth positions over the recent second. Reported
            # physics velocities may oscillate every tick under target-velocity
            # traffic control; differentiating that oscillation invents a
            # sustained acceleration and repeatedly revokes a safe merge.
            if len(self.position_history) >= 3:
                history_times = np.asarray([item[0] - timestamp for item in self.position_history])
                design = np.stack([.5 * history_times**2, history_times, np.ones_like(history_times)], axis=1)
                positions = np.asarray([item[1] for item in self.position_history])
                acceleration = np.linalg.lstsq(design, positions, rcond=None)[0][0]
                norm = float(np.linalg.norm(acceleration))
                self.x[4:6] = acceleration * min(1., 4. / max(norm, 1e-6))
            else:
                self.x[4:6] = 0.
        self.P = (np.eye(6) - gain.dot(H)).dot(self.P)
        self.P = 0.5 * (self.P + self.P.T)
        self.P[np.diag_indices(6)] = np.maximum(self.P[np.diag_indices(6)], 1e-6)
        self.timestamp = timestamp
        self.last_seen = timestamp
        self.last_observed_position = position.copy()
        self.last_observed_velocity = velocity.copy()
        self.lane_context = lane_context

    def predict(self, horizon_s, step_s, multi_intent_enabled=True,
                brake_mps2=2.0, accelerate_mps2=1.2,
                lane_change_s=3.0):
        horizon_s = float(max(horizon_s, 0.0))
        step_s = float(max(step_s, 1e-3))
        count = max(1, int(math.ceil(horizon_s / step_s)))
        times = np.arange(1, count + 1, dtype=np.float64) * step_s
        means, covariances = [], []
        acceleration = self.x[4:6].copy()
        if self.actor_type == 'vehicle' and self.lane_context is not None:
            direction = np.asarray(self.lane_context['direction'])
            normal = np.array([-direction[1], direction[0]])
            if abs(float(np.dot(self.x[2:4], normal))) < .15:
                acceleration -= normal * float(np.dot(acceleration, normal))
        speed = float(np.linalg.norm(self.x[2:4]))
        along_acceleration = float(np.dot(acceleration, self.x[2:4] / max(speed, .01)))
        stopping_time = speed / -along_acceleration if along_acceleration < -.01 else float('inf')
        for time_s in times:
            F = self._transition(time_s)
            moving_time = min(float(time_s), stopping_time)
            displacement = (self.x[2:4] * moving_time + .5 * acceleration * moving_time**2
                            if speed > .15 else np.zeros(2))
            means.append(self.x[:2] + displacement)
            cov = F.dot(self.P).dot(F.T)[:2, :2] + self._process_covariance(time_s)[:2, :2]
            cov = 0.5 * (cov + cov.T)
            cov[np.diag_indices(2)] = np.maximum(cov[np.diag_indices(2)], 1e-4)
            covariances.append(cov)
        means = np.asarray(means, dtype=np.float32)
        covariances = np.asarray(covariances, dtype=np.float32)
        modes = ()
        if multi_intent_enabled and self.actor_type == "vehicle":
            probabilities = self.intent_probabilities or \
                infer_intent_probabilities(
                    {}, float(np.linalg.norm(self.x[2:4])), 0.0, 0.0,
                    self.lane_context, self.observation_count)
            modes = predict_intent_modes(
                self.x[:2], self.x[2:4], times, covariances,
                probabilities, self.lane_context, self.lateral_speed_mps,
                brake_mps2=brake_mps2,
                accelerate_mps2=accelerate_mps2,
                lane_change_s=lane_change_s)
        elif multi_intent_enabled and self.actor_type == 'pedestrian':
            # Preserve alternatives to constant acceleration, especially after
            # an abrupt stop/turn. All hypotheses are conditioned on observed
            # motion; the base posterior remains in the safety calculation.
            velocity = np.asarray(self.last_observed_velocity)
            position = self.x[:2]
            t = times[:, None]
            scenarios = [('walk_continue', .60, 1.), ('walk_stop', .25, 0.),
                         ('walk_faster', .10, 1.4), ('walk_reverse', .05, -.7)]
            modes = tuple(IntentMode(name, weight,
                (position + t * velocity * scale).astype(np.float32),
                covariances.copy(), float(np.linalg.norm(velocity) * abs(scale)))
                for name, weight, scale in scenarios)
        return ActorPrediction(
            actor_id=self.actor_id,
            actor_type=self.actor_type,
            times=times,
            means=means,
            covariances=covariances,
            speed_mps=float(np.linalg.norm(self.x[2:4])),
            extent_xy=self.extent_xy,
            modes=modes,
            current_position=tuple(self.x[:2]),
            # Current gaps use the measured simulator velocity. Keep the KF
            # posterior for future means/modes instead of presenting its lag
            # as an observed catch-up by an already yielding follower.
            current_velocity=tuple(self.last_observed_velocity),
            observation_count=self.observation_count,
            current_heading_xy=self.current_heading_xy,
        )


class BayesianActorTracker:
    """Maintains bounded per-actor posteriors and produces future predictions."""

    def __init__(self, horizon_s=3.0, step_s=0.25, process_noise=1.5,
                 measurement_noise=0.25, max_stale_s=1.0,
                 multi_intent_enabled=True, intent_map_radius_m=60.0,
                 intent_brake_mps2=2.0, intent_accelerate_mps2=1.2,
                 intent_lane_change_s=3.0):
        self.horizon_s = float(horizon_s)
        self.step_s = float(step_s)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_stale_s = float(max_stale_s)
        self.multi_intent_enabled = bool(multi_intent_enabled)
        self.intent_map_radius_m = float(intent_map_radius_m)
        self.intent_brake_mps2 = float(intent_brake_mps2)
        self.intent_accelerate_mps2 = float(intent_accelerate_mps2)
        self.intent_lane_change_s = float(intent_lane_change_s)
        self._tracks = {}

    @staticmethod
    def _actor_type(actor):
        return "pedestrian" if actor.type_id.startswith("walker.") else "vehicle"

    @staticmethod
    def _extent(actor):
        extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
        return (getattr(extent, "x", 1.0), getattr(extent, "y", 0.6))

    @staticmethod
    def _xy(value):
        return [float(value.x), float(value.y)]

    @staticmethod
    def _lane_context(road_map, location):
        waypoint = road_map.get_waypoint(location)
        if waypoint is None or waypoint.is_junction:
            return None
        forward = waypoint.transform.get_forward_vector()
        direction = np.asarray([forward.x, forward.y], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        normal = np.asarray([-direction[1], direction[0]])
        center = waypoint.transform.location
        offset = float(np.dot(
            np.asarray([location.x - center.x, location.y - center.y]),
            normal))

        def accessible(adjacent):
            return bool(adjacent is not None and
                        adjacent.lane_type == waypoint.lane_type and
                        adjacent.lane_id * waypoint.lane_id > 0)

        return {
            "direction": direction,
            "offset_m": offset,
            "lane_width_m": max(float(waypoint.lane_width), 2.0),
            "left": accessible(waypoint.get_left_lane()),
            "right": accessible(waypoint.get_right_lane()),
        }

    def update(self, actors, timestamp, ego_id=None, road_map=None,
               ego_position=None):
        timestamp = float(timestamp)
        seen = set()
        ego_xy = (np.asarray(self._xy(ego_position), dtype=np.float64)
                  if ego_position is not None else None)
        for actor in sorted(actors, key=lambda item: int(item.id)):
            actor_id = int(actor.id)
            if ego_id is not None and actor_id == int(ego_id):
                continue
            if not (actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker.")):
                continue
            track = self._tracks.get(actor_id)
            actor_type = self._actor_type(actor)
            extent = self._extent(actor)
            if track is None or track.actor_type != actor_type:
                track = _Track(actor_id, actor_type, self.process_noise,
                               self.measurement_noise, extent)
                self._tracks[actor_id] = track
            location = actor.get_location()
            position = self._xy(location)
            lane_context = None
            nearby = (ego_xy is None or np.linalg.norm(
                np.asarray(position) - ego_xy) <= self.intent_map_radius_m)
            if (self.multi_intent_enabled and road_map is not None and
                    actor_type == "vehicle" and nearby):
                lane_context = self._lane_context(road_map, location)
            track.extent_xy = extent
            track.intent_active = bool(nearby)
            velocity = self._xy(actor.get_velocity())
            heading = None
            if hasattr(actor, 'get_transform'):
                forward = actor.get_transform().get_forward_vector()
                heading = self._xy(forward)
            track.update(position, velocity, timestamp,
                         lane_context=motion_lane_context(lane_context, velocity), heading_xy=heading)
            seen.add(actor_id)
        for actor_id in list(self._tracks):
            track = self._tracks[actor_id]
            if actor_id not in seen and track.last_seen is not None \
                    and timestamp - track.last_seen > self.max_stale_s:
                del self._tracks[actor_id]
        return [track.predict(
                    self.horizon_s, self.step_s,
                    multi_intent_enabled=(self.multi_intent_enabled and
                                          track.intent_active),
                    brake_mps2=self.intent_brake_mps2,
                    accelerate_mps2=self.intent_accelerate_mps2,
                    lane_change_s=self.intent_lane_change_s)
                for actor_id, track in sorted(self._tracks.items())
                if actor_id in seen and track.initialized]

    @property
    def track_count(self):
        return len(self._tracks)

