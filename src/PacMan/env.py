import os
import gymnasium as gym

import ale_py 

from stable_baselines3 import A2C

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

# !python -m atari_py.import_roms ../Atari-2600-VCS-ROM-Collection/ROMS


import ale_py

gym.register_envs(ale_py)


environment_name = 'MsPacman-v4'

env = gym.make(environment_name, render_mode='human')



episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render() 
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')
env.close()
