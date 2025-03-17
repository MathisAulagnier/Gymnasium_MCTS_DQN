import os
import gymnasium as gym

import ale_py 

import os

import torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

import ale_py

gym.register_envs(ale_py)

environment_name = 'MsPacman-v4'

env = gym.make(environment_name, render_mode='rgb_array')
env.metadata['render_fps'] = 30  # Définir le FPS de rendu

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

env.action_space

env.observation_space

environment_name = 'MsPacman-v4'

env = make_atari_env(environment_name, n_envs=1, seed=0)

env = VecFrameStack(env, n_stack=4)

log_path = os.path.join('Training', 'Logs','dqn_pacman_tensorboard')

model = DQN(
    'CnnPolicy',  # Utilisation d'un CNN pour traiter les images
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,  # Taille de la mémoire de replay
    learning_starts=10000,  # Nombre de steps avant de commencer l'apprentissage
    batch_size=32,
    tau=1.0,  # Paramètre pour la mise à jour du réseau cible
    gamma=0.99,  # Facteur de discount
    train_freq=4,  # Fréquence de mise à jour du réseau
    target_update_interval=1000,  # Fréquence de mise à jour du réseau cible
    exploration_fraction=0.1,  # Fraction de l'exploration
    exploration_initial_eps=1.0,  # Exploration initiale
    exploration_final_eps=0.01,  # Exploration finale
    tensorboard_log=log_path  # Dossier pour les logs TensorBoard
)

model.learn(total_timesteps=1000000, log_interval=100)

dqn_path = os.path.join('Training', 'Saved Models', 'dqn_pacman')

model.save(dqn_path)
env.close()
