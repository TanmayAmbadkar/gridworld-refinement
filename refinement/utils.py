import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN



from collections import deque

class CacheStates:

    def __init__(self, maxlen:int = 10000):

        self.states = deque(maxlen=maxlen)
        self.reachable = deque(maxlen=maxlen)
    
    def __len__(self):
        return len(self.states)
    
    def insert(self, state:np.ndarray, reach:bool):

        self.states.append(state)
        self.reachable.append(reach)
    
    def return_dataset(self):
        return np.array(self.states), np.array(self.reachable)
    
    def clear(self):
        
        self.states.clear()
        self.reachable.clear()


# def train_model(cached_states:CacheStates):

#     model = LogisticRegression()
#     model.fit(*cached_states.return_dataset())
#     pickle.dump(model, open("svc.pkl", "wb"))
#     return model

# def train_model(cached_states:CacheStates):

#     dataset = cached_states.return_dataset()
#     dataset = dataset[0][dataset[1] == 1]
#     return ConvexHull(dataset)

def train_model(cached_states:CacheStates, eps=0.5, min_samples=5):
    """
    Create a robust convex hull that is less sensitive to outliers.

    Args:
        points (array-like): Array of points, where each point is an array-like [x, y].
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        ConvexHull: Convex hull object created from scipy.spatial.ConvexHull.
    """
    # Perform DBSCAN clustering
    dataset = cached_states.return_dataset()
    points = dataset[0][dataset[1] == 1]
    # clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    # # Get indices of core samples
    # core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True
    
    # # Extract core samples
    # core_points = points[core_samples_mask]
    
    # Compute convex hull of core samples
    
    if len(points) == 0:
        return None
    hull = ConvexHull(points)
    
    return hull
