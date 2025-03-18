import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from _Env import ChessEnv
import cloudpickle

'''
def get_action_from_index(board, action_index):
    """Transforme un index en coup valide"""
    legal_moves = list(board.legal_moves)
    if action_index < len(legal_moves):
        return legal_moves[action_index]
    return legal_moves[0]  # Sécurité : si index hors limite, jouer un coup valide
'''

# Création de l'environnement
env = ChessEnv()
env.reset()

serialized_env = cloudpickle.dumps(env)
deserialized_env = cloudpickle.loads(serialized_env)

# Utilisez un chemin absolu
log_path = os.path.abspath(os.path.join('Training', 'Logs', 'dqn_chess_tensorboard'))
os.makedirs(log_path, exist_ok=True)

# Vérifiez que le répertoire a bien été créé
if not os.path.exists(log_path):
    raise FileNotFoundError(f"Le répertoire {log_path} n'a pas pu être créé.")

# Vérifier si un GPU est disponible
print(f"GPU disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

# Initialisation du modèle DQN
model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=1e-5,  # Réduire le taux d'apprentissage pour stabiliser l'entraînement
    buffer_size=200000,  # Augmenter la taille du replay buffer
    learning_starts=20000,  # Commencer l'apprentissage après plus d'exploration
    batch_size=64,  # Augmenter la taille du batch
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.5,  # Augmenter la durée de l'exploration
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    tensorboard_log=log_path
)

# Entraînement du modèle
model.learn(total_timesteps=1000000, log_interval=100)

'''
# Utilisez un chemin absolu
dqn_path = os.path.abspath(os.path.join('Training', 'Saved Models', 'dqn_chess'))
os.makedirs(dqn_path, exist_ok=True)

# Vérifiez que le répertoire a bien été créé
if not os.path.exists(dqn_path):
    raise FileNotFoundError(f"Le répertoire {dqn_path} n'a pas pu être créé.")

model.save(dqn_path)

# Évaluation du modèle
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
'''
'''
# Affichage d'une partie
state = env.reset()
print(env.render(mode='unicode'))
done = False

while not done:
    # Obtenir l'index du meilleur coup prédit
    action_index = model.predict(state, deterministic=True)[0]

    # Transformer l'index en un coup valide
    action = get_action_from_index(env.unwrapped.board, action_index)

    # Exécuter l'action dans l'environnement
    state, reward, done, truncated, info = env.step(action)

    print(env.render(mode='unicode'))
    print("\n\n\n")

'''

# Fermeture de l'environnement
env.close()