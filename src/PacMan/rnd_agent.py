import os
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

# Remove manual environment registration
import ale_py
gym.register_envs(ale_py)  # Removed

# 1) Create a folder to store your videos
video_folder = "recorded_videos"
os.makedirs(video_folder, exist_ok=True)

# 2) Use correct environment ID
env_id = 'MsPacman-v4'  # Updated environment ID
env = make_atari_env(env_id, n_envs=1, seed=0)

# Apply wrappers
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# 3) Wrap the environment to record the video
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x == 0,  # Records the first episode
    video_length=500,                     # Adjust as needed
    name_prefix="pacman-agent"
)

# 4) Run your agent
obs = env.reset()
max_steps = 500
for step in range(max_steps):
    # Sample a single action and wrap it in a list
    action = [env.action_space.sample()]
    obs, rewards, dones, info = env.step(action)
    
    if dones[0]:
        obs = env.reset()

env.close()
