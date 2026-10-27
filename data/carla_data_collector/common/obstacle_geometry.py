"""NumPy geometry shared by static planning and collision supervision."""
import math
import numpy as np


SOLID_SEMANTICS = ('vehicle', 'pedestrian', 'static', 'pole')


def box_corners_xy(box, padding=0.):
    angle = math.radians(float(box['rotation'][2]))
    forward = np.asarray([math.cos(angle), math.sin(angle)])
    normal = np.asarray([-forward[1], forward[0]])
    extent = np.asarray(box['extent'][:2], dtype=float) + float(padding)
    return (np.asarray(box['location'][:2], dtype=float) +
            np.asarray([-1., -1., 1., 1.])[:, None] * extent[0] * forward +
            np.asarray([-1., 1., 1., -1.])[:, None] * extent[1] * normal)


def vertical_overlap(first, second):
    return abs(float(first['location'][2]) - float(second['location'][2])) <= (
        float(first['extent'][2]) + float(second['extent'][2]))


def polygon_overlap(first, second):
    for polygon in (first, second):
        edges = polygon[[1, 2]] - polygon[[0, 1]]
        for edge in edges:
            normal = np.asarray([-edge[1], edge[0]])
            a, b = first.dot(normal), second.dot(normal)
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def path_hits_box(path, box, ego_extent, margin=.35):
    """Check oriented body footprints over finite segments, including the return."""
    points = np.asarray(path, dtype=float)[:, :2]
    corners = box_corners_xy(box)
    radius = math.hypot(*ego_extent) + float(margin)
    lower, upper = corners.min(0) - radius, corners.max(0) + radius
    for start, end in zip(points[:-1], points[1:]):
        if np.any(np.maximum(start, end) < lower) or np.any(np.minimum(start, end) > upper):
            continue
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-5:
            continue
        heading = math.degrees(math.atan2(delta[1], delta[0]))
        for progress in np.linspace(0., 1., max(2, int(math.ceil(length / .5)) + 1)):
            center = start + progress * delta
            ego_box = {'location': center, 'extent': ego_extent, 'rotation': [0., 0., heading]}
            if polygon_overlap(box_corners_xy(ego_box, margin), corners):
                return True
    return False


def projected_extents(box, tangent):
    angle = math.radians(float(box['rotation'][2]))
    direction = np.asarray([math.cos(angle), math.sin(angle)])
    normal = np.asarray([-direction[1], direction[0]])
    along = abs(float(np.dot(direction, tangent)))
    across = abs(float(np.dot(normal, tangent)))
    x, y = box['extent'][:2]
    return along * x + across * y, across * x + along * y
