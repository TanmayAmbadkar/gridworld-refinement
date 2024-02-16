from refinement.graph import Node, depth_first_traversal
from refinement.goal import Goal
from env.dirl_grid import RoomsEnv
from env.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM

start_region = Goal(1, 1, 6, 6)
# mid_region = Goal(17, 1, 12, 6)
goal_region = Goal(9, 17, 6, 12)

start_node = Node(start_region, False, False, "start")
# mid_node = Node(mid_region, True, False, "mid")
goal_node = Node(goal_region, True, True, "goal")

start_node.add_child(goal_node)
# mid_node.add_child(goal_node)



env = RoomsEnv(GRID_PARAMS_LIST[0], START_ROOM[0], FINAL_ROOM[0], max_timesteps=MAX_TIMESTEPS[1])

def run_3grid(minimum_reach: float = 0.9, n_episodes: int = 5000):
    depth_first_traversal(start_node, env, minimum_reach, n_episodes)