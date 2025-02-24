import gymnasium as gym
import ale_py
from _MCTS import MCTS

environment_name = "ALE/VideoCheckers-v5"

env = gym.make(environment_name, render_mode="human")
mcts = MCTS(env)  # Passer l'environnement à MCTS

episodes = 1
for episode in range(1, episodes+1):
    state, info = env.reset()
    terminated = truncated = False
    score = 0
    iteration = 0  # Correction de "iter"

    while not (terminated or truncated):
        iteration += 1
        env.render()

        # Sélection de l’action avec MCTS
        action = mcts.best_move(state)

        # Exécution de l'action
        state, reward, terminated, truncated, info = env.step(action)
        score += reward

    print(f'Episode: {episode}, Score: {score}, Truncated: {truncated}, Info: {info}, Terminated: {terminated}, Iterations: {iteration}')

env.close()
