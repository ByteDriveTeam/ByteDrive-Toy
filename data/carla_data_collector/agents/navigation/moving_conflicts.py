"""Route anchored moving-conflict triggers and time-dependent path admission."""
import math
import numpy as np

from agents.navigation.route_recovery import project_route, sample_route
from agents.navigation.maneuver_safety import observed_velocity, _current, _extended_projection, _finite_path_distances


def moving_threats(centerline, predictions, ego_xy, ego_speed, ego_extent,
                   horizon_s=6., margin_m=.6):
    """Trigger from observed motion AND posterior modes, never a risk scalar alone.

    A stable adjacent actor is not a trigger. A pedestrian needs temporal
    overlap with the assigned corridor, including its stopping hypothesis.
    Return the conflict's initial world anchor; execution never moves the
    global route or chases a moving obstacle laterally.
    """
    ego_arc = project_route(centerline, ego_xy)[0]
    threats = []
    for actor in predictions:
        if actor.actor_type not in ('vehicle', 'pedestrian') or not len(actor.means):
            continue
        current = _current(actor)
        arc, lateral, tangent = _extended_projection(centerline, current)
        velocity = observed_velocity(actor)
        along = float(np.dot(velocity, tangent))
        half_length = ego_extent[0] + actor.extent_xy[0]
        half_width = ego_extent[1] + actor.extent_xy[1] + margin_m
        delta = arc - ego_arc
        if delta < -half_length or delta > (max(ego_speed, 2.5) + max(-along, 0.)) * horizon_s + half_length:
            continue
        ttc = max(delta - half_length, 0.) / max(max(ego_speed, 2.5) - along, .5)
        oncoming = actor.actor_type == 'vehicle' and along < -.5
        if oncoming:
            # Actual inward cuts and oncoming-in-lane are supported; merely
            # passing in the adjacent opposing lane remains ordinary driving.
            closing_position = current + velocity * min(ttc, horizon_s)
            future_lateral = project_route(centerline, closing_position)[1]
            conflict = min(abs(lateral), abs(future_lateral)) < half_width and ttc <= horizon_s
        else:
            if actor.actor_type != 'pedestrian':
                continue
            conflict, ttc = False, float('inf')
            hypotheses = [actor.means] + [mode.means for mode in actor.modes if mode.probability >= .1]
            for means in hypotheses:
                for time_s, point in zip(actor.times, means):
                    if time_s > horizon_s:
                        break
                    station, offset, _ = _extended_projection(centerline, point)
                    if (abs(offset) < half_width and
                            abs(station - ego_arc - max(ego_speed, 2.5) * time_s) < half_length + 2.):
                        conflict, ttc = True, min(ttc, float(time_s))
        if not conflict:
            continue
        # Reserve past the initial actor station, including its possible stop.
        # moving_escape_route handles early lateral entry independently.
        anchor_arc = arc
        anchor = sample_route(centerline, anchor_arc)[0].tolist()
        threats.append(dict(actor=actor, ttc_s=float(ttc), kind='oncoming' if oncoming else 'pedestrian',
                            anchor=anchor, distance_m=float(delta)))
    return sorted(threats, key=lambda item: (item['ttc_s'], item['distance_m']))


def pedestrian_blocks_route(path, actor, ego_extent, margin_m=.6, horizon_s=3.):
    """Keep a yield until the corridor clears, independently of ego speed."""
    half_width = ego_extent[1] + max(actor.extent_xy) + margin_m
    half_length = ego_extent[0] + max(actor.extent_xy)
    trajectories = [np.asarray([_current(actor)]), actor.means[actor.times <= horizon_s]]
    trajectories += [mode.means[actor.times <= horizon_s] for mode in actor.modes
                     if mode.probability >= .1]
    return any(station >= -half_length and abs(offset) < half_width
               for trajectory in trajectories for point in trajectory
               for station, offset, _ in [_extended_projection(path, point)])


def swept_motion_check(path, predictions, ego_speed, ego_extent, target_speed,
                       decel_mps2=2., horizon_s=6., margin_m=.6, settled_follower_ids=(),
                       hard_mode_probability=.25, tracking_margin_m=0., ego_heading=None,
                       allow_unobserved_braking=False):
    """Check a brake+steer timing envelope and every supported intent mean.

    Slow/nominal/fast progress bounds cover controller lag; checks use whole
    body support on the local path tangent. Never skip the trigger actor.
    This is additional admission evidence, not a replacement for lane gaps,
    posterior probability, static objects or complete return reservation.
    """
    blockers, minimum = [], float('inf')
    times = np.arange(.0, horizon_s + .051, .1)
    v, target, distance = max(float(ego_speed), 0.), max(float(target_speed), 0.), 0.
    distances = [0.]
    for _ in times[1:]:
        acceleration = float(np.clip((target - v) / .5, -decel_mps2, 1.))
        distance += max(0., v * .1 + .5 * acceleration * .01)
        v = max(0., v + acceleration * .1)
        distances.append(distance)
    # Sample all timing envelopes once, rather than rebuilding path geometry
    # for every actor/hypothesis/time. This runs in the collector's tick loop.
    from agents.navigation.route_recovery import _geometry
    points, segments, lengths, arcs = _geometry(path)
    stations = np.asarray(distances)[None] * np.array([.8, 1., 1.2])[:, None]
    indices = np.clip(np.searchsorted(arcs, stations, side='right') - 1, 0, len(lengths) - 1)
    fractions = np.clip((stations - arcs[indices]) / lengths[indices], 0., 1.)
    ego_points = points[indices] + fractions[:, :, None] * segments[indices]
    tangents = segments[indices] / lengths[indices, None]
    if ego_heading is not None:
        tangents[:, 0] = [math.cos(ego_heading), math.sin(ego_heading)]
    normals = np.stack([tangents[:, :, 1], -tangents[:, :, 0]], axis=2)
    for actor in predictions:
        if actor.actor_type == 'static' or not len(actor.means) or actor.actor_id in settled_follower_ids:
            continue
        current, velocity = _current(actor), observed_velocity(actor)
        if (actor.actor_type == 'pedestrian' and getattr(actor, 'observation_count', 0) < 2 and
                _finite_path_distances(path, [current])[0] <
                ego_extent[1] + max(actor.extent_xy) + margin_m + 4.5 * .5):
            # An unknown first observation cannot certify a lateral escape.
            # Braking can still be safe if no pedestrian motion up to 4.5m/s
            # can reach the ego body before the next observations arrive.
            prefix = times <= .5
            delta = current[None, None] - ego_points[:, prefix]
            dx = np.maximum(np.abs(np.sum(delta * tangents[:, prefix], axis=-1)) - ego_extent[0], 0.)
            dy = np.maximum(np.abs(np.sum(delta * normals[:, prefix], axis=-1)) - ego_extent[1], 0.)
            prefix_safe = np.all(np.hypot(dx, dy) > max(actor.extent_xy) + margin_m + 4.5 * times[prefix])
            if not (allow_unobserved_braking and target == 0. and prefix_safe):
                blockers.append(dict(actor_id=actor.actor_id, kind='pedestrian_motion_unobserved',
                                     required_observations=2))
                continue
        hypotheses = [('observed', current[None] + times[:, None] * velocity)]
        supported = [('posterior', actor.means)] + [
            (mode.name, mode.means) for mode in actor.modes if mode.probability >= hard_mode_probability]
        for name, means in supported:
            trajectory = np.stack([np.interp(times, np.r_[0., actor.times],
                np.r_[current[dim], means[:, dim]]) for dim in range(2)], axis=1)
            terminal = ((means[-1] - means[-2]) / (actor.times[-1] - actor.times[-2])
                        if len(means) > 1 else velocity)
            trajectory += np.maximum(times - actor.times[-1], 0.)[:, None] * terminal
            hypotheses.append((name, trajectory))
        for name, trajectory in hypotheses:
            velocity_steps = np.gradient(trajectory, .1, axis=0)
            speeds = np.linalg.norm(velocity_steps, axis=1)
            fallback = velocity / max(np.linalg.norm(velocity), .01)
            if np.linalg.norm(velocity) <= .3:
                fallback = tangents[1, 0]
            heading = np.where((speeds > .3)[:, None],
                velocity_steps / np.maximum(speeds[:, None], .01), fallback)
            if getattr(actor, 'current_heading_xy', None) is not None:
                heading[0] = np.asarray(actor.current_heading_xy) / max(np.linalg.norm(actor.current_heading_xy), .01)
            side = np.stack([heading[:, 1], -heading[:, 0]], axis=-1)
            align = np.abs(np.sum(heading[None] * tangents, axis=-1))
            cross = np.abs(np.sum(side[None] * tangents, axis=-1))
            ex = align * actor.extent_xy[0] + cross * actor.extent_xy[1]
            ey = cross * actor.extent_xy[0] + align * actor.extent_xy[1]
            difference = trajectory[None] - ego_points
            dx = np.abs(np.sum(difference * tangents, axis=-1)) - ego_extent[0] - ex
            dy = np.abs(np.sum(difference * normals, axis=-1)) - ego_extent[1] - ey
            ax = np.abs(np.sum(difference * heading[None], axis=-1)) - actor.extent_xy[0] - align * ego_extent[0] - cross * ego_extent[1]
            ay = np.abs(np.sum(difference * side[None], axis=-1)) - actor.extent_xy[1] - cross * ego_extent[0] - align * ego_extent[1]
            separation = np.maximum.reduce([dx, dy, ax, ay])
            minimum = min(minimum, float(np.maximum(separation, 0.).min()))
            # Full rectangle SAT needs both bodies' axes. The ego axes alone
            # produce fictitious rear/side conflicts during diagonal departure.
            required = np.full_like(separation, margin_m) + tracking_margin_m * np.minimum(times[None] / .7, 1.)
            initial = separation[:, :1]
            # Known current proximity cannot be undone by freezing lateral
            # motion. A committed escape may open that gap monotonically;
            # it must reach the full future margin within two seconds.
            if np.all(initial > .05) and np.any(initial < required[:, :1]):
                t = np.minimum(times[None] / 2., 1.)
                required = np.minimum(required, initial + (required - initial) * t*t*(3. - 2.*t))
            conflicts = np.argwhere(separation < required - 1e-6)
            if len(conflicts):
                scale_index, index = conflicts[0]
                blockers.append(dict(actor_id=actor.actor_id, kind='swept_motion',
                    hypothesis=name, time_s=float(times[index]), progress_scale=(.8, 1., 1.2)[scale_index]))
                break
    return dict(safe=not blockers, blocking_actors=blockers,
                minimum_swept_clearance_m=float(minimum) if math.isfinite(minimum) else None,
                horizon_s=float(horizon_s), target_speed_mps=target)
