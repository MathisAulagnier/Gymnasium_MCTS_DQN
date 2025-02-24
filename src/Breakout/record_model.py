import os
import gymnasium as gym
import ale_py
import time
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

from stable_baselines3.common.env_util import make_atari_env


gym.register_envs(ale_py)

# Définition des chemins
a2c_path = os.path.join('Breakout', 'Training', 'Models', 'A2C_model.zip')
video_path = os.path.join('Breakout', 'Training', 'Videos')
os.makedirs(video_path, exist_ok=True)

# Création de l'environnement
environment_name = 'ALE/Breakout-v5'
env = make_atari_env(environment_name, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# 3) Enveloppez l'environnement pour enregistrer la vidéo
#    - record_video_trigger=lambda x: x == 0 : enregistre la toute première (et unique) session
#    - video_length=5000 : durée max en nombre de steps de la vidéo (à adapter)
env = VecVideoRecorder(
    env,
    video_folder=video_path,
    record_video_trigger=lambda x: x == 0,
    video_length=1000,
    name_prefix="pacman-agent"
)

# Chargement du modèle
model = A2C.load(a2c_path, env=env)

life_counter = 0
# Exécution du modèle pour générer une vidéo
obs = env.reset()
max_steps = 1000
for step in range(max_steps):
    action, _states = model.predict(obs)
    time.sleep(0.5)
    obs, rewards, dones, info = env.step(action)
    if dones[0]:
        life_counter += 1
        if life_counter == 5:
            break

# Fermeture de l'environnement
env.close()
print(f"Vidéo enregistrée dans {video_path}")

print('___________Trained model____________')
print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')
print('____________________________________')