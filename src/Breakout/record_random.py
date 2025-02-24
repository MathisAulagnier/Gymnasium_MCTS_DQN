import gymnasium as gym
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
