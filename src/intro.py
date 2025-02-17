import os
import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


print("/n_______________________________________________________")

episode_over = False

while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # print("Observation:", observation)
    # print("Reward:", reward)
    # print("Info:", info)
    # print("Terminated:", terminated)
    # print("Truncated:", truncated)
    # print("")


    episode_over = terminated or truncated


env.close()