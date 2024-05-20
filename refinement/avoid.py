import numpy as np
from scipy.spatial import ConvexHull

import numpy as np
from scipy.spatial import ConvexHull

class Region():

    def __init__(self, x: float, y: float, height: float, width: float):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        
        self.extended_region = []
        
    def __str__(self):
        return f"Region: x={self.x}, y={self.y}, height={self.height}, width={self.width}"

    def sample_state(self):
        return np.array([np.random.uniform(low=self.x, high=self.x + self.width),
                         np.random.uniform(low=self.y, high=self.y + self.height)])
        
    def in_region(self, last_state, next_state):
        # Check if line segment intersects with the rectangle
        if self.intersect_rect(last_state, next_state):
            return True
        
        # Check if line segment intersects with any convex hull
        for hull in self.extended_region:
            if self.intersect_hull(last_state, next_state, hull):
                # print("Hull avoided")
                return True
        
        return False

    def intersect_rect(self, start, end):
        # Rectangle boundaries
        x_min, x_max = self.x, self.x + self.width
        y_min, y_max = self.y, self.y + self.height

        # Use Cohen-Sutherland or Liang-Barsky line clipping algorithm to detect intersection
        return self.line_clip(start, end, x_min, x_max, y_min, y_max)
        
    def intersect_hull(self, start, end, hull):
        # Check intersection with each edge of the convex hull
        for simplex in hull.simplices:
            if self.segment_intersect(start, end, hull.points[simplex[0]], hull.points[simplex[1]]):
                return True
        return False

    def line_clip(self, start, end, x_min, x_max, y_min, y_max):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        p = [-dx, dx, -dy, dy]
        q = [start[0] - x_min, x_max - start[0], start[1] - y_min, y_max - start[1]]
        u1, u2 = 0, 1

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False  # Parallel and outside
            else:
                t = q[i] / float(p[i])
                if p[i] < 0:
                    u1 = max(u1, t)
                else:
                    u2 = min(u2, t)
                if u1 > u2:
                    return False

        return True


    def segment_intersect(self, p1, p2, p3, p4):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # collinear
            elif val > 0:
                return 1  # clockwise
            else:
                return 2  # counterclockwise

        def on_segment(p, q, r):
            if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
                return True
            return False

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, p3): return True
        if o2 == 0 and on_segment(p1, p2, p4): return True
        if o3 == 0 and on_segment(p3, p4, p1): return True
        if o4 == 0 and on_segment(p3, p4, p2): return True

        return False


    def extend_region(self, list_of_points):
        hull = ConvexHull(list_of_points)
        self.extended_region.append(hull)


class Avoid():
    
    def __init__(self, list_of_regions):
        
        self.list_of_regions = list_of_regions
    
    def check_trigger(self, last_state:np.ndarray, next_state:np.ndarray):
        
        for region in self.list_of_regions:
            if region.in_region(last_state, next_state):
                return True
        return False
    
    def check_region(self,  last_state:np.ndarray, next_state:np.ndarray):
        
        for region in self.list_of_regions:
            if region.in_region(last_state, next_state):
                # print("avoided", region)
                return region
        return None
    
    def reward(self,  last_state:np.ndarray, next_state:np.ndarray):
        region = self.check_region(last_state, next_state)
        if region:
            return -10
        else:
            return 0