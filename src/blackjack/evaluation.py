from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from blackjack.agent import BlackjackAgent

# hyperparameters
learning_rate = 0.001
n_episodes = 1000000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

agent.load()


# Créer une matrice de la politique optimale
policy = np.zeros((22, 11))  # 22 car max sum est 21, +1 pour l'indexation facile

# Parcourir tous les états possibles
for player_sum in range(12, 22):  # On commence à 12 car en dessous, on hit toujours
    for dealer_card in range(1, 11):  # 1 = As, 10 = 10, J, Q, K
        obs = (player_sum, dealer_card, False)  # On suppose pas d'As utilisable, i.e. ici l'As vaut 1
        policy[player_sum, dealer_card] = agent.get_action(obs)

# Affichage avec seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(policy[12:22, 1:11], cmap="coolwarm", annot=True, cbar=False,
            xticklabels=range(1, 11), yticklabels=range(12, 22))
plt.xlabel("Carte Visible du Croupier")
plt.ylabel("Somme des Cartes du Joueur")
plt.title("Politique Apprise (0 = Stick, 1 = Hit)")
plt.show()