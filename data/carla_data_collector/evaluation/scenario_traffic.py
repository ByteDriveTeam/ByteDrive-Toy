"""Deterministic local traffic around a constructed lateral maneuver."""

import math
import carla


class ScenarioTraffic:
    def __init__(self, world, start, actors, case, mode, density, release_step):
        self.world, self.start, self.actors = world, start, actors
        self.case, self.mode, self.release_step = case, mode, release_step
        self.cars = []
        self.target_side = (1 if case == "avoid_left" else -1) if case.startswith("avoid") else 0
        self.tags = {}
        self.last_step = 0
        if mode == "occupied":
            self.add(25. if case.startswith("avoid") else 1., self.target_side, 0., "blocking")
        elif mode == "fast_rear":
            self.add(-45., self.target_side, 16., "fast_rear")
        elif mode == "parallel":
            self.add(1., self.target_side, 4., "parallel")
        elif mode == "oncoming":
            self.add(140., self.target_side, -12., "oncoming")
        # Dense packs in the immediate area, with a deliberate initial gap
        # around the merge. The explicit hazard above can close that gap.
        for index in range(density):
            group, slot = index % 3, index // 3
            side = (1, -1, 0)[group]
            if mode == "oncoming" and side == self.target_side:
                side = 0
            distance = (-34. - 12. * slot) if group != 2 else (-26. - 12. * slot)
            if mode == "fast_rear" and side == self.target_side:
                distance = -70. - 12. * slot
            # The forced opposite-side obstacle remains stationary: keep its
            # approaching traffic queued behind it instead of running into it.
            self.add(distance, side, 4. if side == self.target_side else 0., "background")

    def pose(self, distance, side):
        options = ([self.start] if abs(distance) < 1e-4 else
                   self.start.next(distance) if distance > 0. else self.start.previous(-distance))
        if not options:
            return None
        waypoint = options[0]
        pose = waypoint.transform
        forward, right = pose.get_forward_vector(), pose.get_right_vector()
        width = waypoint.lane_width
        return carla.Transform(carla.Location(
            x=pose.location.x - side * width * right.x,
            y=pose.location.y - side * width * right.y,
            z=pose.location.z + .25), pose.rotation)

    def add(self, distance, side, speed, tag):
        bp = self.world.get_blueprint_library().find("vehicle.audi.tt")
        pose = self.pose(distance, side)
        if pose is None:
            if tag == "background":
                return
            raise RuntimeError("traffic route unavailable")
        lane = self.world.get_map().get_waypoint(pose.location, project_to_road=False,
                                                 lane_type=carla.LaneType.Driving)
        if lane is None and tag == "background":
            return
        if speed < 0.:
            pose.rotation.yaw += 180.
        actor = self.world.try_spawn_actor(bp, pose)
        if actor is None:
            if tag == "background":
                return
            raise RuntimeError("traffic spawn failed: {} distance={} side={}".format(tag, distance, side))
        self.actors.append(actor)
        if speed == 0.:
            actor.set_simulate_physics(False)
        self.cars.append({"actor": actor, "distance": distance, "side": side,
                          "speed": speed, "tag": tag})
        self.tags.setdefault(tag, []).append(int(actor.id))

    def update(self, step, ego):
        dt = max(0, step - self.last_step) * .1
        self.last_step = step
        if self.mode == "late_intrusion" and step == 28:
            start = self.start.transform.location
            forward = self.start.transform.get_forward_vector()
            location = ego.get_location()
            distance = ((location.x - start.x) * forward.x +
                        (location.y - start.y) * forward.y)
            self.add(distance + 7., self.target_side, 0., "blocking")
        for item in sorted(self.cars, key=lambda item: item["distance"], reverse=True):
            actor = item["actor"]
            speed = item["speed"]
            if item["tag"] in ("blocking", "parallel") and self.release_step >= 0 and step >= self.release_step:
                if speed < 8.:
                    actor.set_simulate_physics(True)
                    item["speed"] = speed = 8.
            # Give scripted actors observable simulator velocity; positions
            # advance through physics instead of teleporting through ego.
            if speed != 0.:
                ahead = [other["distance"] for other in self.cars
                         if other is not item and other["side"] == item["side"]
                         and other["distance"] > item["distance"]]
                if speed > 0. and ahead and dt > 0.:
                    speed = min(speed, max(0., (min(ahead) - 8. - item["distance"]) / dt))
                origin = self.start.transform
                forward = origin.get_forward_vector()
                location = ego.get_location()
                relative_x, relative_y = location.x - origin.location.x, location.y - origin.location.y
                ego_distance = relative_x * forward.x + relative_y * forward.y
                ego_lateral = relative_x * forward.y - relative_y * forward.x
                ego_forward = ego.get_transform().get_forward_vector()
                aligned = abs(ego_forward.x * forward.x + ego_forward.y * forward.y)
                crossed = abs(ego_forward.x * forward.y - ego_forward.y * forward.x)
                ego_width = ego.bounding_box.extent.y * aligned + ego.bounding_box.extent.x * crossed
                lateral_overlap = (abs(ego_lateral - item["side"] * self.start.lane_width) <
                                   ego_width + actor.bounding_box.extent.y + .25)
                if speed > 0. and lateral_overlap and ego_distance > item["distance"]:
                    velocity = ego.get_velocity()
                    ego_speed = math.hypot(velocity.x, velocity.y)
                    gap = ego_distance - item["distance"]
                    speed = min(speed, max(0., ego_speed + (gap - 8.) / 1.2))
                    if dt > 0.:
                        speed = min(speed, max(0., (gap - 6.) / dt))
                item["distance"] += speed * dt
                pose = self.pose(item["distance"], item["side"])
                if pose is None:
                    continue
                pose.location.z -= .20
                if speed < 0.:
                    pose.rotation.yaw += 180.
                actor.set_transform(pose)
                heading = math.radians(pose.rotation.yaw)
                actor.set_target_velocity(carla.Vector3D(
                    x=abs(speed) * math.cos(heading), y=abs(speed) * math.sin(heading), z=0.))

