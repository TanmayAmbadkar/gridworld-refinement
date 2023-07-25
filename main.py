
# import random
# random.seed(0)
# import numpy as np
# np.random.seed(0)
# import torch
# torch.random.manual_seed(0)

from refinement.graph import Node, depth_first_traversal
from refinement.goal import Goal
from env.dirl_grid import RoomsEnv
from env.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM

start_region = Goal(1, 1, 6, 6)
mid_region = Goal(17, 1, 12, 6)
goal_region = Goal(17, 17, 6, 6)

start_node = Node(start_region, False, False, "start")
mid_node = Node(mid_region, True, False, "mid")
goal_node = Node(goal_region, False, True, "goal")

start_node.add_child(mid_node)
mid_node.add_child(goal_node)



env = RoomsEnv(GRID_PARAMS_LIST[0], START_ROOM[0], FINAL_ROOM[0])
depth_first_traversal(start_node, env, 0.9)