import numpy as np
from scipy.spatial import ConvexHull
class AbstractState():

    def __init__(self, x: float, y: float, height: float, width: float):
        self.x = x
        self.y = y
        self.height = height
        self.width = width

    def sample_state(self):
        return np.array([np.random.uniform(low=self.x, high=self.x + self.width),
                         np.random.uniform(low=self.y, high=self.y + self.height)])

class Goal(AbstractState):

    def __init__(self, x:float, y:float, height:float, width:float):
        super().__init__(x, y, height, width)
        # self.current_goal = self.sample_state()

    def reset(self):
        self.current_goal = self.sample_state()
        return self.current_goal
        
    def predicate(self, state:np.ndarray):

        return self.current_goal[0] - 1 <= state[0] <= self.current_goal[0] + 1 \
                and self.current_goal[1] - 1 <= state[1] <= self.current_goal[1] + 1
    
    def in_goal_region(self, state:np.ndarray):

        return self.x <= state[0] <= self.x + self.width \
                and self.y <= state[1] <= self.y + self.height
    
    def reward(self, state:np.ndarray):
        if self.predicate(state[:2]):
            return 10
        else:
            
            return state[3] - np.sqrt(np.sum((state[:2] - self.current_goal)**2))

class ModifiedGoal(Goal):
    def __init__(self,  x:float, y:float, height:float, width:float, hull, reachable:bool=False):
        super().__init__(x, y, height, width)
        self.hull = hull
        self.reachable = reachable
        
        print(self.hull)
        
    def sample_state(self):
        state = super().sample_state()

        while not self.in_goal_region(state):
            state = super().sample_state()
        return state

    def in_goal_region(self, point):
        new_points = np.vstack([self.hull.points, np.array(point).reshape(1,-1)])
        new_hull = ConvexHull(new_points)
        return list(new_hull.vertices) == list(self.hull.vertices)
