import numpy as np
from collections import defaultdict
import math

from NODE import Node
from MCTS import MCTS

import ale_py
import os
import gymnasium as gym


def play_game_with_mcts(env, mcts, episodes=1):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = mcts.get_best_action(state[0]) 
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


gym.register_envs(ale_py)

# Initialiser l'environnement
# environment_name = 'ALE/VideoChess-v5'
environment_name = 'ALE/TicTacToe3D-v5'

env = gym.make(environment_name, render_mode='human', obs_type = 'ram')
env.metadata['render_fps'] = 60 


# Créer l'instance MCTS
mcts = MCTS(env)

# Jouer quelques parties
play_game_with_mcts(env, mcts, episodes=30)