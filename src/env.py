import os
from random import randrange
import gymnasium as gym

import ale_py 

from stable_baselines3 import A2C

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder



import ale_py


gym.register_envs(ale_py)
ale = ale_py.ALEInterface()
ale.setInt('random_seed', 123)

# Enable screen display and sound output
ale.setBool('display_screen', True)
ale.setBool('sound', False)


environment_name = 'ALE/VideoChess-v5'
# environment_name = 'ALE/TicTacToe3D-v5'


env = gym.make(environment_name, render_mode='human', obs_type = 'ram')
env.metadata['render_fps'] = 60 


ale.loadROM(ale_py.roms.get_rom_path(environment_name))


episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render() 
        # Get the list of legal actions
        legal_actions = ale.getLegalActionSet()
        num_actions = len(legal_actions)

    
        action = legal_actions[randrange(num_actions)]
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')
env.close()