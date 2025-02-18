import os
import gymnasium as gym

import ale_py 

from Breakout.breakout_class import BreakoutState
from Breakout.mcts_class import MCTS
from Breakout.mcts_node_class import MCTSNode


gym.register_envs(ale_py)


environment_name = 'ALE/Breakout-v5'

env = gym.make(environment_name, render_mode='human', obs_type="ram")


initial_state = BreakoutState(env)
root = MCTSNode(initial_state)
mcts = MCTS(root)
best_next_state = mcts.search(iterations=1000)