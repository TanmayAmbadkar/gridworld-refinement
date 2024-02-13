import numpy as np
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
        self.current_goal = None

    def reset(self):
        self.current_goal = self.sample_state()
        return self.current_goal
        
    def predicate(self, state:np.ndarray):

        return self.x <= state[0] <= self.x + self.width and self.y <= state[1] <= self.y + self.height
    
    def reward(self, state:np.ndarray):
        if np.linalg.norm(state - self.current_goal)**2 < 0.2:
            return 5
        else:
            return -np.abs(np.sum(np.array([(self.x + self.width)/2, (self.y + self.height)/2]) - state))/100

class ModifiedGoal(Goal):
    def __init__(self,  x:float, y:float, height:float, width:float, classifier, reachable:bool=False):
        super().__init__(x, y, height, width)
        self.classifier = classifier
        self.reachable = reachable

    
    def predicate(self, state:np.ndarray):
        if super().predicate(state):
            prediction = self.classifier.predict(state.reshape(1,-1))
            return prediction == self.reachable
        else:
            return False

    def reward(self, state:np.ndarray):
        if self.predicate(state):
            return super.reward(state)
        
    def sample_state(self):
        state = super().sample_state()

        while not self.predicate(state):
            state = super().sample_state()
        
        return state
