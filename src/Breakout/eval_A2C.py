import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import ale_py 

gym.register_envs(ale_py)

environment_name = 'ALE/Breakout-v5'

env = gym.make(environment_name, render_mode='human', obs_type="ram")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

env = make_atari_env(environment_name, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

a2c_path = os.path.join('Breakout','Training', 'Models', 'A2C_model')

# Model randomly initialized
model_rnd = A2C('CnnPolicy', env, verbose=1)


# Model trained for 400k timesteps
model_trn = A2C.load(a2c_path, env)
print("Model loaded successfully")


# Evaluate the agent
print('_____Randomly initialized model_____')
mean_reward, std_reward = evaluate_policy(model_rnd, model_rnd.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
print('____________________________________')

print('_____Trained model_____')
mean_reward, std_reward = evaluate_policy(model_trn, model_trn.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
print('____________________________________')

logs_path = os.path.join('Breakout','Training','Logs', 'A2C_1')
os.execvp('tensorboard', ['tensorboard', '--logdir', logs_path])

env.close()