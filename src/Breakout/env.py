import os
import gymnasium as gym

import ale_py 

gym.register_envs(ale_py)


environment_name = 'ALE/Breakout-v5'

env = gym.make(environment_name, render_mode='human', obs_type="ram")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)





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
