"""Small online Bayesian motion predictor for CARLA actors.

The collector runs in the CARLA 0.9.15 Python 3.7 environment, so this module
intentionally uses only NumPy.  It is a constant-acceleration Kalman filter,
not a learned predictor: the posterior covariance grows with the forecast
horizon and is consumed by ``risk_field``.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ActorPrediction:
    actor_id: int
    actor_type: str
    times: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    speed_mps: float
    extent_xy: tuple


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
        self.initialized = False

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

    def update(self, position, velocity, timestamp):
        position = np.asarray(position, dtype=np.float64)[:2]
        velocity = np.asarray(velocity, dtype=np.float64)[:2]
        timestamp = float(timestamp)
        z = np.concatenate((position, velocity))
        if not self.initialized or self.timestamp is None or timestamp <= self.timestamp:
            self.x[:] = 0.0
            self.x[:2] = position
            self.x[2:4] = velocity
            self.P[:] = np.eye(6, dtype=np.float64)
            self.P[:2, :2] *= self.measurement_noise
            self.P[2:4, 2:4] *= max(self.measurement_noise, 0.25)
            self.P[4:, 4:] *= 4.0
            self.timestamp = timestamp
            self.last_seen = timestamp
            self.initialized = True
            return

        dt = min(max(timestamp - self.timestamp, 1e-3), 1.0)
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
        self.P = (np.eye(6) - gain.dot(H)).dot(self.P)
        self.P = 0.5 * (self.P + self.P.T)
        self.P[np.diag_indices(6)] = np.maximum(self.P[np.diag_indices(6)], 1e-6)
        self.timestamp = timestamp
        self.last_seen = timestamp

    def predict(self, horizon_s, step_s):
        horizon_s = float(max(horizon_s, 0.0))
        step_s = float(max(step_s, 1e-3))
        count = max(1, int(math.ceil(horizon_s / step_s)))
        times = np.arange(1, count + 1, dtype=np.float64) * step_s
        means, covariances = [], []
        for time_s in times:
            F = self._transition(time_s)
            means.append(F.dot(self.x)[:2].copy())
            cov = F.dot(self.P).dot(F.T)[:2, :2] + self._process_covariance(time_s)[:2, :2]
            cov = 0.5 * (cov + cov.T)
            cov[np.diag_indices(2)] = np.maximum(cov[np.diag_indices(2)], 1e-4)
            covariances.append(cov)
        return ActorPrediction(
            actor_id=self.actor_id,
            actor_type=self.actor_type,
            times=times,
            means=np.asarray(means, dtype=np.float32),
            covariances=np.asarray(covariances, dtype=np.float32),
            speed_mps=float(np.linalg.norm(self.x[2:4])),
            extent_xy=self.extent_xy,
        )


class BayesianActorTracker:
    """Maintains bounded per-actor posteriors and produces future predictions."""

    def __init__(self, horizon_s=3.0, step_s=0.25, process_noise=1.5,
                 measurement_noise=0.25, max_stale_s=1.0):
        self.horizon_s = float(horizon_s)
        self.step_s = float(step_s)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_stale_s = float(max_stale_s)
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

    def update(self, actors, timestamp, ego_id=None):
        timestamp = float(timestamp)
        seen = set()
        for actor in actors:
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
            track.extent_xy = extent
            track.update(self._xy(actor.get_location()), self._xy(actor.get_velocity()), timestamp)
            seen.add(actor_id)
        for actor_id in list(self._tracks):
            track = self._tracks[actor_id]
            if actor_id not in seen and track.last_seen is not None \
                    and timestamp - track.last_seen > self.max_stale_s:
                del self._tracks[actor_id]
        return [track.predict(self.horizon_s, self.step_s)
                for actor_id, track in self._tracks.items()
                if actor_id in seen and track.initialized]

    @property
    def track_count(self):
        return len(self._tracks)

