import gymnasium as gym

import os
import ale_py 


from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback


gym.register_envs(ale_py)

environment_name = 'ALE/Breakout-v5'

env = gym.make(environment_name, render_mode="rgb_array")  # Pour l'affichage
env = AtariWrapper(env)  # Prétraitement (redimensionnement, grayscale...)


print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Environnement vectorisé
env = DummyVecEnv([env])

log_path = os.path.join('Training', 'Logs')

model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# Callback pour sauvegarder le modèle périodiquement
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./models/", name_prefix="a2c_breakout")

model.learn(total_timesteps=400000)

a2c_path = os.path.join('Breakout','Training', 'Models', 'A2C_model')

model.save(a2c_path)

del model