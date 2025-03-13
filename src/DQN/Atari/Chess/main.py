import gymnasium as gym
import ale_py
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration de l'environnement
environment_name = 'ALE/VideoChess-v5'
# Création de l'environnement Atari prétraité
env = make_atari_env(environment_name, n_envs=1, seed=0)
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
    buffer_size=40000,  # Réduire la taille du tampon de replay
    learning_starts=10000,  # Nombre de steps avant de commencer l'apprentissage
    batch_size=32,
    tau=1.0,  # Paramètre pour la mise à jour du réseau cible
    gamma=0.99,  # Facteur de discount
    train_freq=4,  # Fréquence de mise à jour du réseau
    target_update_interval=1000,  # Fréquence de mise à jour du réseau cible
    exploration_fraction=0.1,  # Fraction de l'exploration
    exploration_initial_eps=1.0,  # Exploration initiale
    exploration_final_eps=0.01,  # Exploration finale
    tensorboard_log=None  # Dossier pour les logs TensorBoard
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