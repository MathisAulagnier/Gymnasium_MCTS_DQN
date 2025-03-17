import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import ale_py

gym.register_envs(ale_py)

# Créez l'environnement avec les mêmes wrappers que lors de l'entraînement
env = make_atari_env('MsPacman-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)  # Empile 4 frames
env = VecTransposeImage(env)  # Transpose les images pour qu'elles soient au format (canaux, hauteur, largeur)

# Chargez le modèle
model = DQN.load('Training/Saved Models/dqn_pacman', env=env)

# Évaluez le modèle
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    