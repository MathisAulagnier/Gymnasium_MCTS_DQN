from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

from collections import defaultdict
import gymnasium as gym
from tqdm import tqdm

from Blackjack.agent import BlackjackAgent

# hyperparameters
learning_rate = 0.001
n_episodes = 3000000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


save_interval = 100000  # Save policy every 100,000 iterations
save_dir = "blackjack/output/policy_figures"

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

def save_policy_figure(agent, iteration):
    """Saves the learned policy as a heatmap figure."""
    policy = np.zeros((22, 11))
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            obs = (player_sum, dealer_card, False)
            policy[player_sum, dealer_card] = agent.get_action(obs)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(policy[12:22, 1:11], cmap="coolwarm", annot=True, cbar=False,
                xticklabels=range(1, 11), yticklabels=range(12, 22))
    plt.xlabel("Carte Visible du Croupier")
    plt.ylabel("Somme des Cartes du Joueur")
    plt.title(f"Politique Apprise - {iteration} itérations (0 = Stick, 1 = Hit)")
    
    save_path = os.path.join(save_dir, f"policy_{iteration}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Policy saved at {save_path}")



for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
    if episode % save_interval == 0:
        save_policy_figure(agent, episode)

# save the trained agent
agent.save()
print("Training completed! Agent saved!")

