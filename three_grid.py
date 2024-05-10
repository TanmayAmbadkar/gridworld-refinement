from refinement.graph import Node, depth_first_traversal
from refinement.goal import Goal
from env.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM
from refinement.avoid import Avoid, Region

from env.gridworldenv import ContinuousGridworld

start_region = Goal(1, 1, 6, 6)
mid_region_1 = Goal(1, 17, 6, 12)
mid_region_2 = Goal(9, 1, 6, 12)
goal_region = Goal(17, 17, 6, 6)

start_node = Node(start_region, False, False, "start")
mid_node_1 = Node(mid_region_1, True, False, "mid_1")
mid_node_2 = Node(mid_region_2, True, False, "mid_2")
goal_node = Node(goal_region, True, True, "goal")


start_node.add_child(mid_node_1)
start_node.add_child(mid_node_2)
mid_node_1.add_child(goal_node)
mid_node_2.add_child(goal_node)

goal_node.add_parent(mid_node_1)
goal_node.add_parent(mid_node_2)
mid_node_1.add_parent(start_node)
mid_node_2.add_parent(start_node)

# 
# env = RoomsEnv(GRID_PARAMS_LIST[1], START_ROOM[1], FINAL_ROOM[1], max_timesteps=MAX_TIMESTEPS[1])
custom_doors={
    ((0, 0), (0, 1)): 7,
    ((0, 1), (0, 2)): 7,
    ((0, 2), (1, 2)): 7,
    ((1, 1), (1, 2)): 7,
    ((1, 0), (1, 1)): 7,
    ((1, 0), (2, 0)): 7,
    ((2, 0), (2, 1)): 7,
    ((2, 1), (2, 2)): 7,
}
print(custom_doors)
env = ContinuousGridworld(
    custom_doors=custom_doors,
    render = False
)
    

    
def run_3grid(minimum_reach: float = 0.9, n_episodes: int = 4000,  n_episodes_test: int = 4000, path: str = ""):
    depth_first_traversal(start_node, env, minimum_reach, n_episodes, n_episodes_test, path)

