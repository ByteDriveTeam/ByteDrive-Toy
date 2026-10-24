"""Continuous targets on the assigned route, with bounded forward progress."""

import numpy as np


class RouteTracker:
    def __init__(self, points):
        points = np.asarray(points, dtype=np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
        self.points = points[keep]
        self.index = 0
        self.arc = 0.0
        self.lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arcs = np.r_[0.0, np.cumsum(self.lengths)]

    def target(self, position, lookahead_m):
        if len(self.points) < 2:
            return self.points[-1].tolist()
        # Search locally so crossing routes cannot jump to a later segment.
        start = max(self.index - 2, 0)
        end = min(self.index + 20, len(self.lengths))
        origins = self.points[start:end]
        segments = self.points[start + 1:end + 1] - origins
        lengths = self.lengths[start:end]
        fraction = np.clip(np.sum((np.asarray(position) - origins) * segments, axis=1)
                           / lengths ** 2, 0.0, 1.0)
        nearest = origins + fraction[:, None] * segments
        local = int(np.argmin(np.sum((nearest - position) ** 2, axis=1)))
        index = start + local
        self.arc = max(self.arc, float(self.arcs[index] + fraction[local] * lengths[local]))
        self.index = max(self.index, index)
        target_arc = min(self.arc + max(float(lookahead_m), 0.1), self.arcs[-1])
        index = min(int(np.searchsorted(self.arcs, target_arc, side="right")) - 1,
                    len(self.lengths) - 1)
        fraction = (target_arc - self.arcs[index]) / self.lengths[index]
        return (self.points[index] + fraction *
                (self.points[index + 1] - self.points[index])).tolist()

    def local_points(self, count=50):
        return self.points[max(0, self.index - 3):self.index + count].tolist()
