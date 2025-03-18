import gymnasium as gym
import ale_py
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    WarpFrame,
    ClipRewardEnv,
)
from stable_baselines3.common.monitor import Monitor

# Configuration de l'environnement
environment_name = 'ALE/VideoCheckers-v5'

# Création de l'environnement sans make_atari_env
def make_env(env_id, seed=0):
    env = gym.make(env_id, render_mode='rgb_array')
    # Suppression de NoopResetEnv car il n'est pas compatible avec cet environnement
    env = MaxAndSkipEnv(env, skip=4)  # Skip des frames
    env = EpisodicLifeEnv(env)  # Terminer l'épisode après une vie perdue
    env = WarpFrame(env)  # Redimensionner les frames
    env = ClipRewardEnv(env)  # Limiter les récompenses
    env = Monitor(env)  # Surveillance de l'environnement
    env.reset(seed=seed)  # Initialisation de la graine aléatoire
    return env

# Création de l'environnement
env = DummyVecEnv([lambda: make_env(environment_name, seed=0)])
# Empilement des frames pour prendre en compte l'historique des observations
env = VecFrameStack(env, n_stack=4)

# Chemin pour les logs TensorBoard
log_path = os.path.join('Training', 'Logs', 'dqn_checkers_tensorboard')
os.makedirs(log_path, exist_ok=True)

# Vérification de la disponibilité du GPU
print(f"GPU disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

# Initialisation du modèle DQN
model = DQN(
    'CnnPolicy',  # Utilisation d'un CNN pour traiter les images
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=40000,  # Taille de la mémoire de replay
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

# Entraînement du modèle
model.learn(total_timesteps=1000000, log_interval=100)

# =============Save & Reload Model=============
dqn_path = os.path.join('Training', 'Saved Models', 'dqn_checkers')
os.makedirs(dqn_path, exist_ok=True)

# Sauvegarde du modèle
model.save(dqn_path)

# Évaluation du modèle
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Fermeture de l'environnement
env.close()