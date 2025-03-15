import ale_py
import random
from _MCTS import MCTS
from _Node import Node
import gymnasium as gym

# Initialisation de l'environnement
env = gym.make("ALE/TicTacToe3D-v5", render_mode="human")
observation, info = env.reset()  # Récupérer l'état initial du jeu

# Création de la racine de l'arbre MCTS
root = Node(observation)
mcts = MCTS(env, iterations=3) #mettre iterations = 1000 normalement mais le pc lache quand je met plus que 3-4

# Boucle du jeu
done = False
truncated = False
step = 0

while not done and not truncated:
    print(f"\n Tour {step}")

    # L'IA utilise MCTS pour choisir la meilleure action
    best_action = mcts.search(root) #long mais ca marche, mettre "best_action = random..." permet d'aller plus vite pour  tests

    # Appliquer l'action choisie dans l'environnement
    observation, reward, done, truncated, info = env.step(best_action)

    # Afficher les informations du tour
    print(f"Action jouée : {best_action}, Récompense : {reward}") # reward est toujours 0...
    step += 1

# Fin de la partie
print("Partie terminée !")

# Fermer l'environnement
env.close()

#ca joue et finit les parties sans pb, donc le main est clean normalement, les pbs viennent de _Node.py et _MCTS.py