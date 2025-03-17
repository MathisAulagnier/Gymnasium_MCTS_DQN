import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

import ale_py

gym.register_envs(ale_py)

# 1) Créez un dossier pour stocker vos vidéos
video_folder = "recorded_videos"
os.makedirs(video_folder, exist_ok=True)

# 2) Créez l'environnement avec les mêmes wrappers que lors de l'entraînement
env = make_atari_env('MsPacman-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# 3) Enveloppez l'environnement pour enregistrer la vidéo
#    - record_video_trigger=lambda x: x == 0 : enregistre la toute première (et unique) session
#    - video_length=5000 : durée max en nombre de steps de la vidéo (à adapter)
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x == 0,
    video_length=5000,
    name_prefix="pacman-agent"
)

# 4) Chargez le modèle
model = DQN.load('Training/Saved Models/dqn_pacman', env=env)

# 5) Exécutez votre agent
obs = env.reset()
# Pour éviter de boucler infiniment, fixez un max de steps ou d'épisodes
max_steps = 5000
for step in range(max_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # Si vous voulez couper dès qu'un épisode est terminé :
    if dones[0]:
        obs = env.reset()  # Relance un nouvel épisode si vous voulez enchaîner

env.close()

