import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

import os

import ale_py

gym.register_envs(ale_py)

# Créez l'environnement avec les mêmes wrappers que lors de l'entraînement
env = make_atari_env('MsPacman-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)  # Empile 4 frames
env = VecTransposeImage(env)  # Transpose les images pour qu'elles soient au format (canaux, hauteur, largeur)

# Chargez le modèle
model = DQN.load('Training/Saved Models/dqn_pacman', env=env)


evaluate_policy(model, env, n_eval_episodes=10, render=True)

logs_path = os.path.join('Training','Logs', 'dqn_pacman_tensorboard/DQN_1')
os.execvp('tensorboard', ['tensorboard', '--logdir', logs_path])

env.close()

