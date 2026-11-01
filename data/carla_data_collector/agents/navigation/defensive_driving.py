"""Deterministic route-preserving braking reserves and evidenced cut-in yielding.

RSS-inspired reserve, not an RSS implementation or a formal safety guarantee.
Every cap is advisory to the existing safety/jerk/traffic-control planners.
"""
import math
import numpy as np

from agents.navigation.risk_field import _route_geometry, _project_route_segment, _interval_probability


def reserve_distance(speed, lead_speed, response_s, accel_mps2, brake_mps2, lead_brake_mps2, gap_m):
    after = max(speed, 0.) + accel_mps2 * response_s
    return gap_m + max(0., speed * response_s + .5 * accel_mps2 * response_s**2 +
        after**2 / (2. * brake_mps2) - max(lead_speed, 0.)**2 / (2. * lead_brake_mps2))


def reserve_speed(gap, lead_speed, response_s, accel_mps2, brake_mps2, lead_brake_mps2, gap_m):
    """Invert the monotone braking reserve; no numerical optimizer required."""
    available = max(gap - gap_m, 0.) + max(lead_speed, 0.)**2 / (2. * lead_brake_mps2)
    a = 1. / (2. * brake_mps2)
    b = response_s * (1. + accel_mps2 / brake_mps2)
    c = .5 * accel_mps2 * response_s**2 + (accel_mps2 * response_s)**2 / (2. * brake_mps2)
    return max(0., (-b + math.sqrt(max(0., b*b + 4.*a*(available - c)))) / (2.*a))


class DefensivePlanner:
    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = bool(cfg.get('defensive_enabled', True))
        self.response = float(cfg.get('defensive_response_s', 1.))
        self.headway = float(cfg.get('defensive_headway_s', 1.4))
        self.lead_brake = float(cfg.get('defensive_lead_brake_mps2', 6.))
        self.probability = float(cfg.get('defensive_intent_probability', .15))
        self.inward_speed = float(cfg.get('defensive_inward_speed_mps', .2))
        self.horizon = float(cfg.get('defensive_preview_s', 3.))
        self.uncertainty = float(cfg.get('defensive_uncertainty_margin_m', 2.))
        self.clear_time = float(cfg.get('defensive_clear_time_s', .6))
        self.release_accel = float(cfg.get('defensive_release_accel_mps2', .8))
        self.brake = float(cfg.get('comfort_decel_mps2', 3.))
        self.accel = float(cfg.get('max_accel_mps2', 2.2))
        self.min_gap = float(cfg.get('front_min_gap_m', 3.))
        self.reset()

    def reset(self):
        self.cap = None
        self.actor_id = None
        self.kind = None
        self.clear_since = None
        self.last_time = None
        self.stopping_actor = None
        self.stopping_gap = None

    def update(self, route, predictions, position, speed, extent, timestamp, cruise_speed,
               suspended=False):
        result = dict(enabled=self.enabled, suspended=bool(suspended), active=False,
                      actor_id=None, kind=None, speed_cap_mps=None, candidates=[],
                      release_held=False, clear_since_sim_time=None)
        result['stopping_committed'] = False
        if not self.enabled or suspended:
            self.reset()
            return result
        if self.last_time is not None and timestamp < self.last_time:
            self.reset()
        dt = min(max(timestamp - self.last_time, 0.), .25) if self.last_time is not None else .1
        self.last_time = timestamp
        geometry = _route_geometry(route)
        if geometry is None or len(geometry[0]) < 2:
            self.reset()
            return result
        points, arcs = geometry[:2]
        ego_arc, _, ego_direction = _project_route_segment(position, points, arcs)
        ego_xy = np.asarray(position)[:2]
        candidates = []
        for actor in predictions or ():
            if actor.actor_type != 'vehicle' or not len(actor.means):
                continue
            current = np.asarray(actor.current_position if actor.current_position is not None else actor.means[0])
            velocity = np.asarray(actor.current_velocity if actor.current_velocity is not None else (0., 0.))
            arc, lateral, tangent = _project_route_segment(current, points, arcs)
            longitudinal = float(np.dot(current - ego_xy, ego_direction))
            # Never turn a trailing/side-by-side actor into a lead reserve.
            if longitudinal <= extent[0] + actor.extent_xy[0] or arc <= ego_arc:
                continue
            along = float(np.dot(velocity, tangent))
            if along < -.5:
                continue  # Existing moving-conflict planner owns oncoming traffic.
            normal = np.array([-tangent[1], tangent[0]])
            heading = np.asarray(actor.current_heading_xy if actor.current_heading_xy is not None else tangent)
            heading /= max(float(np.linalg.norm(heading)), .01)
            actor_length = abs(float(np.dot(heading, tangent))) * actor.extent_xy[0] + abs(float(np.dot(heading, normal))) * actor.extent_xy[1]
            actor_width = abs(float(np.dot(heading, normal))) * actor.extent_xy[0] + abs(float(np.dot(heading, tangent))) * actor.extent_xy[1]
            corridor = extent[1] + actor_width + .15
            gap = arc - ego_arc - extent[0] - actor_length
            kind, probability, entry_s = 'lead_reserve', 1., None
            in_route = abs(lateral) < corridor
            if not in_route:
                inward = -math.copysign(1., lateral) * float(np.dot(velocity, normal))
                if actor.observation_count < 3:
                    continue
                modes = [(0., actor.means, actor.covariances)] + [
                    (mode.probability, mode.means, mode.covariances) for mode in actor.modes]
                probability = 0.
                entry_times = []
                for weight, means, covariances in modes:
                    for time_s, mean, covariance in zip(actor.times, means, covariances):
                        if time_s > self.horizon:
                            break
                        station, offset, direction = _project_route_segment(mean, points, arcs)
                        movement = abs(lateral) - abs(offset)
                        if abs(offset) >= corridor or movement < .5 or station <= ego_arc + extent[0]:
                            continue
                        future_normal = np.array([-direction[1], direction[0]])
                        sigma = math.sqrt(max(float(future_normal.dot(covariance).dot(future_normal)), .001))
                        if _interval_probability(offset, sigma, corridor) < .35:
                            continue
                        if inward < self.inward_speed and weight < max(.3, self.probability):
                            continue
                        probability += weight if weight > 0. else (1. if inward >= self.inward_speed else 0.)
                        entry_times.append(float(time_s))
                        break
                if probability < self.probability:
                    continue
                kind, probability, entry_s = 'cut_in_reserve', min(probability, 1.), min(entry_times)
            covariance = np.asarray(actor.covariances[0])
            sigma = math.sqrt(max(float(tangent.dot(covariance).dot(tangent)), 0.))
            margin = min(sigma, self.uncertainty)
            effective_gap = max(gap - margin, 0.)
            lead_speed = max(along, 0.)
            # A supported braking hypothesis can lower the anticipated lead
            # velocity. Do not use covariance alone as evidence of a cut-in.
            for mode in actor.modes:
                if mode.name != 'brake' or mode.probability < self.probability:
                    continue
                i = min(int(np.searchsorted(actor.times, min(1., self.horizon))), len(actor.times)-1)
                future_arc = _project_route_segment(mode.means[i], points, arcs)[0]
                lead_speed = min(lead_speed, max((future_arc - arc) / max(float(actor.times[i]), .1), 0.))
            cap = min(reserve_speed(effective_gap, lead_speed, self.response, self.accel,
                                   self.brake, self.lead_brake, self.min_gap),
                      max(effective_gap - self.min_gap, 0.) / self.headway)
            if cap >= cruise_speed:
                continue
            candidates.append(dict(actor_id=int(actor.actor_id), kind=kind, gap_m=float(gap),
                uncertainty_margin_m=float(margin), supported_probability=float(probability),
                entry_time_s=entry_s, lead_speed_mps=float(lead_speed), speed_cap_mps=float(cap),
                observed_lead_speed_mps=float(along),
                required_reserve_m=float(reserve_distance(speed, lead_speed, self.response,
                    self.accel, self.brake, self.lead_brake, self.min_gap))))
        result['candidates'] = candidates
        chosen = min(candidates, key=lambda item: (item['speed_cap_mps'], item['actor_id'])) if candidates else None
        if chosen is not None:
            raw = chosen['speed_cap_mps']
            stationary_lead = chosen['kind'] == 'lead_reserve' and chosen['observed_lead_speed_mps'] <= .5
            opened_gap = (self.stopping_actor == chosen['actor_id'] and self.stopping_gap is not None and
                          chosen['gap_m'] > self.stopping_gap + 1.5)
            if (stationary_lead and not opened_gap and (self.stopping_actor == chosen['actor_id'] or
                    (chosen['observed_lead_speed_mps'] <= .3 and raw < speed))):
                self.stopping_gap = (min(self.stopping_gap, chosen['gap_m'])
                    if self.stopping_actor == chosen['actor_id'] and self.stopping_gap is not None else chosen['gap_m'])
                self.stopping_actor = chosen['actor_id']
                raw = min(raw, speed)
                result['stopping_committed'] = True
                result['stopping_anchor_gap_m'] = self.stopping_gap
            else:
                self.stopping_actor = None
                self.stopping_gap = None
            self.clear_since = None
            self.cap = raw if self.cap is None else min(raw, self.cap + self.release_accel * dt)
            self.actor_id, self.kind = chosen['actor_id'], chosen['kind']
        elif self.cap is not None:
            self.stopping_actor = None
            self.stopping_gap = None
            if self.clear_since is None:
                self.clear_since = timestamp
            if timestamp - self.clear_since >= self.clear_time:
                self.cap += self.release_accel * dt
                if self.cap >= cruise_speed:
                    self.cap, self.actor_id, self.kind = None, None, None
            result['release_held'] = self.cap is not None
        result.update(active=self.cap is not None, speed_cap_mps=self.cap,
                      actor_id=self.actor_id, kind=self.kind, clear_since_sim_time=self.clear_since)
        return result
