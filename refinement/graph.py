from __future__ import annotations

import numpy as np
import gymnasium as gym
from ppo.policy_sb3 import train_policy, test_policy
from refinement.utils import CacheStates, train_model
from refinement.goal import Goal, ModifiedGoal
from refinement.avoid import Avoid
class Node():

    def __init__(self, goal:np.ndarray, splittable:bool = True, final:bool=False, name:str = ""):
        self.goal = goal
        self.splittable = splittable
        self.children = {}
        self.parents = {}
        self.final = final
        self.name = name
        self.idx = 0 

    def sample_state(self):
        return self.goal.sample_state()
    
    def __iter__ (self):
        return self

    def add_child(self, child:Node, avoid=None):
        self.children[id(child)] = {
            "child": child, 
            "reach_probability": 0, 
            "policy": None,
            "avoid": avoid
        }
        
    def add_parent(self, parent:Node,):
        self.parents[id(parent)] = parent

    def __next__(self):
        
        keys = list(self.children.keys())
        if self.idx == len(keys):
            raise StopIteration
        else:
            self.idx+=1
            return self.children[keys[self.idx-1]]

    def remove_child(self, child:Node):
        self.children.pop(id(child))
    
    def print_graph(self):
        pass

def split_goal(goal:Goal, cached_states:CacheStates):

    hull = train_model(cached_states)
    
    if hull is None:
        return None

    goal_r = ModifiedGoal(
        x = goal.x, 
        y = goal.y,
        height = goal.height,
        width = goal.width,
        hull = hull,
        reachable = True
    )
    
    return goal_r

def add_avoid_region(avoid, trajectories: list, k: int = 3):
    
    map_region_points = {}
    violations = 0
    for region in avoid.list_of_regions:
        map_region_points[region] = []
        
    for trajectory in trajectories:
        region = avoid.check_region(trajectory[0][-2], trajectory[0][-1])
        if region is not None:
            map_region_points[region].extend(trajectory[0][-k:])
            violations+=1
    
    for region, points in map_region_points.items():
        if len(points) >= 3:
            region.extend_region(points)
       
    return violations/len(trajectories)
    
    
def depth_first_traversal(head: Node, env: gym.Env, minimum_reach: float = 0.9, n_episodes: int = 3000, n_episodes_test: int = 3000, path: str = ""):

    edges = []
    file = open(path + "/result.txt", "w")
    explore(head, env, minimum_reach, edges, n_episodes, n_episodes_test, file)


def explore(parent: Node, env: gym.Env, minimum_reach: float = 0.9, edges: list = [], n_episodes: int = 3000, n_episodes_test: int = 3000, file = None):

    if parent.final:
        return False

    for child in parent:
        if parent.name+"_"+child['child'].name not in edges:
            
            print(f"Evaluating edge ({parent.name}, {child['child'].name})")
            policy = train_policy(env, parent, child['child'], child['avoid'], n_episodes, minimum_reach)
            reach, cached_states, trajectories = test_policy(policy, env, parent, child['child'], child['avoid'], n_episodes_test)

            print(f"Edge ({parent.name}, {child['child'].name}) reach probability: {reach}")
            
            
            print(f"{parent.name}, {child['child'].name}: {reach}", file = file)
            if reach < minimum_reach and child['child'].splittable:

                print(f"Edge ({parent.name}, {child['child'].name}) not realised: {reach}")
                
                if child['avoid'] is not None:
                    print("Violations: ", add_avoid_region(child['avoid'], trajectories, k=1))
                
                goal_r = split_goal(goal = child['child'].goal, cached_states = cached_states)

                # if goal_r is not None:
                #     goal_r_node = Node(
                #         goal = goal_r, 
                #         splittable=False,
                #         final = child['child'].final,
                #         name = child['child'].name + "_r"
                #     )
                    
                #     parent.add_child(goal_r_node, avoid = child['avoid'])
                
                if child['avoid'] is not None:
                    goal_node_avoid = Node(
                        goal = child['child'].goal, 
                        splittable=False,
                        final = child['child'].final,
                        name = child['child'].name + "_avoid"
                    )
                    
                    parent.add_child(goal_node_avoid, avoid = child['avoid'])


                
                for other_parent in child['child'].parents.values():
                    if id(parent) != id(other_parent):
                        parent.add_child(other_parent)
                        
                
                # grandparent.add_child(goal_r_node)
            
            parent.children[id(child['child'])]['reach_probability'] = reach
            parent.children[id(child['child'])]['policy'] = policy
            edges.append(parent.name+"_"+child['child'].name)

            del cached_states
            status = explore(child['child'], env, minimum_reach, edges, n_episodes, file)
            
            if status:
                return False
        
        if child['child'].final and reach>=minimum_reach:
            return True
            




