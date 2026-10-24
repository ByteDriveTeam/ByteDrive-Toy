"""Deterministic Bayesian intent hypotheses for nearby road users.

Intent probabilities use recent measured acceleration and lane-relative motion.
Each mode retains its own trajectory distribution; downstream risk code must
weight the modes instead of treating their union as one wide Gaussian.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class IntentMode:
    name: str
    probability: float
    means: np.ndarray
    covariances: np.ndarray
    speed_mps: float


def _normal_likelihood(value, center, scale):
    return math.exp(-0.5 * ((float(value) - center) / scale) ** 2)


def infer_intent_probabilities(previous, speed_mps, acceleration_mps2,
                               lateral_speed_mps, lane_context,
                               observation_count):
    """Update a small discrete posterior from motion evidence and lane access."""
    speed = max(float(speed_mps), 0.0)
    base = {"keep": 0.72, "brake": 0.16, "accelerate": 0.12}
    if speed < 1.0:
        base = {"keep": 0.90, "brake": 0.08, "accelerate": 0.02}
    elif lane_context is not None and observation_count >= 3:
        for side in ("left", "right"):
            if lane_context.get(side, False):
                base["change_" + side] = 0.04
                base["keep"] -= 0.04

    evidence = {}
    for name in base:
        prior = 0.65 * float(previous.get(name, base[name])) + 0.35 * base[name]
        if name == "brake":
            likelihood = _normal_likelihood(acceleration_mps2, -2.0, 1.25)
        elif name == "accelerate":
            likelihood = _normal_likelihood(acceleration_mps2, 1.4, 1.25)
        else:
            likelihood = _normal_likelihood(acceleration_mps2, 0.0, 1.1)
        if name == "change_left":
            likelihood *= _normal_likelihood(lateral_speed_mps, 1.0, 0.55)
        elif name == "change_right":
            likelihood *= _normal_likelihood(lateral_speed_mps, -1.0, 0.55)
        elif lane_context is not None:
            likelihood *= _normal_likelihood(lateral_speed_mps, 0.0, 0.5)
        evidence[name] = max(prior * likelihood, 1e-12)

    total = sum(evidence.values())
    probabilities = {name: 0.9 * evidence[name] / total + 0.1 * base[name]
                     for name in base}
    norm = sum(probabilities.values())
    return {name: value / norm for name, value in probabilities.items()}


def _travel(speed, acceleration, time_s):
    speed = max(float(speed), 0.0)
    acceleration = float(acceleration)
    if acceleration < 0.0:
        time_s = min(time_s, speed / max(-acceleration, 1e-6))
    return max(speed * time_s + 0.5 * acceleration * time_s * time_s, 0.0)


def predict_intent_modes(position, velocity, times, base_covariances,
                         probabilities, lane_context, lateral_speed_mps,
                         brake_mps2=2.0, accelerate_mps2=1.2,
                         lane_change_s=3.0):
    """Forecast separate world-space Gaussian trajectories for each intent."""
    position = np.asarray(position, dtype=np.float64)[:2]
    velocity = np.asarray(velocity, dtype=np.float64)[:2]
    speed = float(np.linalg.norm(velocity))
    if lane_context is not None:
        heading = np.asarray(lane_context["direction"], dtype=np.float64)
        heading /= max(float(np.linalg.norm(heading)), 1e-6)
    elif speed > 0.3:
        heading = velocity / speed
    else:
        heading = np.asarray([1.0, 0.0], dtype=np.float64)
    normal = np.asarray([-heading[1], heading[0]])
    forward_speed = max(float(np.dot(velocity, heading)), 0.0)
    lateral_speed = float(np.clip(lateral_speed_mps, -2.0, 2.0))
    lane_width = (float(lane_context.get("lane_width_m", 3.5))
                  if lane_context is not None else 3.5)
    lane_offset = (float(lane_context.get("offset_m", 0.0))
                   if lane_context is not None else 0.0)
    modes = []
    for name, probability in probabilities.items():
        means, covariances = [], []
        if name == "brake":
            acceleration = -max(float(brake_mps2), 0.1)
        elif name == "accelerate":
            acceleration = max(float(accelerate_mps2), 0.1)
        else:
            acceleration = 0.0
        for time_s, base_covariance in zip(times, base_covariances):
            time_s = float(time_s)
            distance = _travel(forward_speed, acceleration, time_s)
            if name.startswith("change_"):
                sign = 1.0 if name == "change_left" else -1.0
                fraction = min(time_s / max(float(lane_change_s), 0.2), 1.0)
                smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                goal = sign * max(lane_width - sign * lane_offset,
                                  0.5 * lane_width)
                lateral = goal * smooth + lateral_speed * time_s * math.exp(
                    -time_s / 0.6)
            else:
                lateral = lateral_speed * 0.6 * (1.0 - math.exp(
                    -time_s / 0.6))
            means.append(position + heading * distance + normal * lateral)
            covariance = np.asarray(base_covariance, dtype=np.float64).copy()
            covariance += np.eye(2) * 0.03 * time_s * time_s
            if name.startswith("change_"):
                covariance += np.outer(normal, normal) * 0.1 * time_s
            covariances.append(covariance)
        modes.append(IntentMode(
            name=name, probability=float(probability),
            means=np.asarray(means, dtype=np.float32),
            covariances=np.asarray(covariances, dtype=np.float32),
            speed_mps=speed))
    return tuple(modes)
