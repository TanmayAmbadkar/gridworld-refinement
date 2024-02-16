from __future__ import annotations

import numpy as np
import gymnasium as gym
from ppo.policy import train_policy
from refinement.utils import CacheStates, train_model
from refinement.goal import Goal, ModifiedGoal
class Node():

    def __init__(self, goal:np.ndarray, splittable:bool = True, final:bool=False, name:str = ""):
        self.goal = goal
        self.splittable = splittable
        self.children = {}
        self.final = final
        self.name = name
        self.idx = 0 

    def sample_state(self):
        return self.goal.sample_state()
    
    def __iter__ (self):
        return self

    def add_child(self, child:Node):
        self.children[id(child)] = {
            "child": child, 
            "reach_probability": 0, 
            "policy": None
        }

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

    model = train_model(cached_states)

    goal_r = ModifiedGoal(
        x = goal.x, 
        y = goal.y,
        height = goal.height,
        width = goal.width,
        classifier = model,
        reachable = True
    )


    return goal_r


def depth_first_traversal(head: Node, env: gym.Env, minimum_reach: float = 0.9, n_episodes: int = 3000):

    edges = []
    explore(head, env, minimum_reach, edges, n_episodes)

def explore(parent: Node, env: gym.Env, minimum_reach: float = 0.9, edges: list = [], n_episodes: int = 3000):

    if parent.final:
        return False

    for child in parent:
        if parent.name+"_"+child['child'].name not in edges:
            
            print(f"Evaluating edge ({parent.name}, {child['child'].name})")
            reach, policy, cached_states = train_policy(env, parent, child['child'], n_episodes, minimum_reach)

            print(f"Edge ({parent.name}, {child['child'].name}) reach probability: {reach}")
            if reach < minimum_reach and child['child'].splittable:

                print(f"Edge ({parent.name}, {child['child'].name}) not realised: {reach}")
                goal_r = split_goal(goal = child['child'].goal, cached_states = cached_states)

                goal_r_node = Node(
                    goal = goal_r, 
                    splittable=False,
                    final = child['child'].final,
                    name = child['child'].name + "_r"
                )

                parent.add_child(goal_r_node)
                # grandparent.add_child(goal_r_node)
            
            parent.children[id(child['child'])]['reach_probability'] = reach
            parent.children[id(child['child'])]['policy'] = policy
            edges.append(parent.name+"_"+child['child'].name)

            del cached_states
            status = explore(child['child'], env, minimum_reach, edges, n_episodes)
            
            if status:
                return False
        
        if child['child'].final and reach>=minimum_reach:
            return True
            




