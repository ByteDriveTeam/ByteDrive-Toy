"""Bounded solid-object queries; native environment boxes are indexed once.

Static boxes have no invented motion covariance or maneuver intent. Props keep
their actor IDs; environment boxes have stable negative planning IDs. Stored
annotations continue using the original bbox/source/semantic fields.
"""
import hashlib
import math
import numpy as np
import carla

from agents.navigation.behavior_prediction import ActorPrediction


ENV_SOLIDS = ('Static', 'Poles', 'GuardRail', 'Fences', 'Walls')


def box_payload(bb, semantic='static', source='env', actor_id=None):
    return {'semantic': semantic, 'source': source, 'id': actor_id,
            'location': [bb.location.x, bb.location.y, bb.location.z],
            'extent': [bb.extent.x, bb.extent.y, bb.extent.z],
            'rotation': [bb.rotation.roll, bb.rotation.pitch, bb.rotation.yaw]}


def actor_box(actor):
    bb, transform = actor.bounding_box, actor.get_transform()
    center = carla.Location(bb.location.x, bb.location.y, bb.location.z)
    transform.transform(center)
    world_box = carla.BoundingBox(center, bb.extent)
    world_box.rotation = carla.Rotation(
        pitch=transform.rotation.pitch + bb.rotation.pitch,
        yaw=transform.rotation.yaw + bb.rotation.yaw,
        roll=transform.rotation.roll + bb.rotation.roll)
    return box_payload(world_box, source='actor', actor_id=int(actor.id))


def box_key(box):
    return tuple(round(float(value), 2) for field in ('location', 'extent', 'rotation')
                 for value in box[field])


class StaticObstacleIndex:
    def __init__(self, world, radius_m=60., minimum_height_m=.08):
        self.world, self.radius_m = world, float(radius_m)
        self.minimum_height_m = float(minimum_height_m)
        boxes = {}
        for name in ENV_SOLIDS:
            for bb in world.get_level_bbs(getattr(carla.CityObjectLabel, name)):
                box = box_payload(bb, 'pole' if name == 'Poles' else 'static')
                if self._usable(box):
                    boxes.setdefault(box_key(box), box)
        self.boxes = list(boxes.values())
        # A spatial grid keeps long driving sessions independent of map size.
        self.cells = {}
        self.large = []
        for index, box in enumerate(self.boxes):
            radius = math.hypot(*box['extent'][:2])
            if radius > 40.:
                self.large.append(index)
                continue
            x, y = box['location'][:2]
            for ix in range(math.floor((x-radius)/20.), math.floor((x+radius)/20.)+1):
                for iy in range(math.floor((y-radius)/20.), math.floor((y+radius)/20.)+1):
                    self.cells.setdefault((ix, iy), []).append(index)

    def _usable(self, box):
        values = [value for field in ('location', 'extent', 'rotation') for value in box[field]]
        return (all(math.isfinite(value) for value in values) and
                min(box['extent']) > .001 and
                box['extent'][2] * 2. >= self.minimum_height_m)

    def nearby(self, ego):
        origin = actor_box(ego)
        x, y, z = origin['location']
        indices = set(self.large)
        radius = self.radius_m
        for ix in range(math.floor((x-radius)/20.), math.floor((x+radius)/20.)+1):
            for iy in range(math.floor((y-radius)/20.), math.floor((y+radius)/20.)+1):
                indices.update(self.cells.get((ix, iy), []))
        props = [actor_box(actor) for actor in self.world.get_actors().filter('static.prop.*')
                 if hasattr(actor, 'bounding_box')]
        # Prefer the actor source when a spawned prop is also exposed by level boxes.
        boxes = {box_key(box): box for box in [self.boxes[i] for i in sorted(indices)] + props}
        result = []
        for key, box in boxes.items():
            if not self._usable(box):
                continue
            if abs(box['location'][2] - z) > box['extent'][2] + origin['extent'][2]:
                continue
            distance = math.hypot(box['location'][0]-x, box['location'][1]-y)
            if distance > radius + math.hypot(*box['extent'][:2]):
                continue
            identifier = box['id']
            if identifier is None:
                identifier = -1 - int(hashlib.sha256(repr(key).encode('ascii')).hexdigest()[:12], 16)
            result.append((int(identifier), box))
        return result

    def predictions(self, ego, horizon_s=3., step_s=.25):
        times = np.arange(step_s, horizon_s + 1e-6, step_s)
        return [ActorPrediction(identifier, 'static', times,
                    np.repeat(np.asarray(box['location'][:2])[None], len(times), axis=0),
                    np.zeros((len(times), 2, 2)), 0., tuple(box['extent'][:2]),
                    current_position=tuple(box['location'][:2]), current_velocity=(0., 0.),
                    static_box=box)
                for identifier, box in self.nearby(ego)]
