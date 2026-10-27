"""Exclude explicit navigation-end behavior from driving supervision.

The model has no destination input. A terminal navigation stop must therefore
not become an ordinary straight-driving target. Raw scene records are retained.
Scenes without explicit terminal diagnostics keep their previous semantics.
"""

import math


TRAINING_SELECTION = "exclude_explicit_navigation_goal_tail_v1"


def is_navigation_goal_approach(row):
    expert = row.get("expert") or {}
    plan = expert.get("longitudinal_plan") or {}
    if 'navigation_goal_active' in plan:
        return bool(plan['navigation_goal_active'])
    # Read the first goal-approach recordings made before the explicit flag.
    return (plan.get("navigation_goal_distance_m") is not None and
            plan.get("navigation_goal_speed_mps") is not None and
            expert.get("state") in ("APPROACH_STOP", "STOPPING") and
            expert.get("reason") == "comfortable_braking")


def navigation_goal_cutoff(*timelines):
    """Earliest goal-driven decision, using both high- and low-rate records."""
    cutoff = None
    for rows in timelines:
        for row in rows:
            if not is_navigation_goal_approach(row):
                continue
            expert = row.get("expert") or {}
            timestamp = float(expert.get("decision_sim_time", row["sim_time"]))
            if math.isfinite(timestamp):
                cutoff = timestamp if cutoff is None else min(cutoff, timestamp)
    return cutoff


def window_before_navigation_goal(timestamp, horizon_s, cutoff):
    """Reject current/future targets touching the terminal tail, including its boundary."""
    return cutoff is None or float(timestamp) + float(horizon_s) < cutoff - 1e-6


def navigation_training_prefix(kinematics, frames=()):
    """Truncate before interpolation, so no vocabulary window can cross the tail."""
    cutoff = navigation_goal_cutoff(kinematics, frames)
    if cutoff is None:
        return kinematics
    return [row for row in kinematics if float(row["sim_time"]) < cutoff - 1e-6]
