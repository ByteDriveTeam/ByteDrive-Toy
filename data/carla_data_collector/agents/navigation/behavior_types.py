# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Behaviour profiles used by :mod:`behavior_agent`.

The old implementation kept ``tailgate_counter`` as a class attribute.  That
made two agents influence one another.  Profiles are now small per-agent
objects while retaining the public profile names and values.
"""

from dataclasses import dataclass, field


@dataclass
class BehaviorProfile:
    """Mutable per-agent driving profile."""

    max_speed: float
    speed_lim_dist: float
    speed_decrease: float
    safety_time: float
    min_proximity_threshold: float
    braking_distance: float
    tailgate_counter: int = field(default=0)


def _profile(max_speed, speed_lim_dist, speed_decrease, safety_time,
             min_proximity_threshold, braking_distance, tailgate_counter=0):
    return BehaviorProfile(
        max_speed=max_speed,
        speed_lim_dist=speed_lim_dist,
        speed_decrease=speed_decrease,
        safety_time=safety_time,
        min_proximity_threshold=min_proximity_threshold,
        braking_distance=braking_distance,
        tailgate_counter=tailgate_counter,
    )


class Cautious(object):
    """Class for Cautious agent."""
    max_speed, speed_lim_dist, speed_decrease = 40, 6, 12
    safety_time, min_proximity_threshold, braking_distance = 3, 12, 6
    tailgate_counter = 0
    @staticmethod
    def create():
        return _profile(40, 6, 12, 3, 12, 6)


class Normal(object):
    """Class for Normal agent."""
    max_speed, speed_lim_dist, speed_decrease = 50, 3, 10
    safety_time, min_proximity_threshold, braking_distance = 3, 10, 5
    tailgate_counter = 0
    @staticmethod
    def create():
        return _profile(50, 3, 10, 3, 10, 5)


class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed, speed_lim_dist, speed_decrease = 70, 1, 8
    safety_time, min_proximity_threshold, braking_distance = 3, 8, 4
    tailgate_counter = -1
    @staticmethod
    def create():
        return _profile(70, 1, 8, 3, 8, 4, -1)
