import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import ale_py 


from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 1

environment_name = 'ALE/Breakout-v5'

a2c_path = os.path.join('Breakout', 'Training', 'Models', 'A2C_model')
video_path = os.path.join('Breakout', 'Training', 'Videos')

# Environnement non vectorisé pour l'enregistrement vidéo du modèle aléatoire
env_video_random = gym.make(environment_name, render_mode='rgb_array', obs_type="ram")
env_video_random = RecordVideo(env_video_random, video_folder=video_path, name_prefix="random_model",
                               episode_trigger=lambda x: True)
env_video_random = RecordEpisodeStatistics(env_video_random, buffer_length=num_eval_episodes)

# Enregistrement vidéo avec un modèle non entraîné (aléatoire)
for episode_num in range(num_eval_episodes):
    obs, info = env_video_random.reset()

    episode_over = False
    while not episode_over:
        action = env_video_random.action_space.sample()
        obs, reward, terminated, truncated, info = env_video_random.step(action)

        episode_over = terminated or truncated

print('_____Randomly initialized model_____')
print(f'Episode time taken: {env_video_random.time_queue}')
print(f'Episode total rewards: {env_video_random.return_queue}')
print(f'Episode lengths: {env_video_random.length_queue}')
print('____________________________________')

env_video_random.close()

# Chargement du modèle entraîné
env_trained = gym.make(environment_name, render_mode='rgb_array', obs_type="ram")
env_video_trained = RecordVideo(env_trained, video_folder=video_path, name_prefix="trained_model",
                                episode_trigger=lambda x: True)

model_trn = A2C.load(a2c_path, env_trained)

# Enregistrement vidéo avec le modèle entraîné
for episode_num in range(num_eval_episodes + 3):
    obs, info = env_video_trained.reset()

    episode_over = False
    while not episode_over:
        action, _states = model_trn.predict(obs)
        obs, reward, terminated, truncated, info = env_video_trained.step(action)

        episode_over = terminated or truncated

print('___________Trained model____________')
print(f'Episode time taken: {env_video_trained.time_queue}')
print(f'Episode total rewards: {env_video_trained.return_queue}')
print(f'Episode lengths: {env_video_trained.length_queue}')
print('____________________________________')

env_video_trained.close()