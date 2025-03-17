import numpy as np
import random
import gymnasium as gym
from _Node import Node

def coords_to_index(x, y, z, size=4):
    return x * (size ** 2) + y * size + z

class MCTS:
    def __init__(self, env, iterations=1000):
        self.base_env = env
        self.iterations = iterations

    def search(self, root):    
        # Cloner l’environnement pour éviter de modifier l’état réel
        sim_env = gym.make("ALE/TicTacToe3D-v5")  # Nouvelle instance propre
        obs, _ = sim_env.reset()  # Obtenir un état initial

        if isinstance(obs, np.ndarray):
            root_state = obs  # Si Gymnasium retourne un état sous forme de tableau
        else:
            raise TypeError(f"L'état du jeu doit être un tableau NumPy, mais c'est {type(obs)}")

        root = Node(root_state)  # Utiliser l'état sous forme de tableau
        
        for _ in range(self.iterations):
            node = self.select(root, sim_env)
            child = self.expand(node, sim_env) or node
            reward = self.simulate(node, sim_env)
            self.backpropagate(node, reward)
        
        return root.best_child(exploration_weight=0).action  # Sélectionne le meilleur coup

    def select(self, node, env):
        while node.children:
            node = node.best_child()
        return node
    
    def expand(self, node, env):
        if not node.is_fully_expanded():
            print("1")
            legal_moves = node.get_possible_actions()
            random.shuffle(legal_moves)  # Mélanger pour varier les choix

            for move in legal_moves:
                move_index = coords_to_index(*move)  # Convertir (x, y, z) en un index valide
                move_index = move_index % env.action_space.n # pour rester dans les incides jouables, sinon le code casse
                if not any(child.action == move_index for child in node.children):
                    new_state, _, terminated, truncated, _ = env.step(move_index)  # Utiliser l'index
                    if not terminated and not truncated:
                        child = Node(new_state, node, move_index)  # Stocker l'index
                        node.children.append(child)
                        return child
        print("2")
        return None  # Aucun coup possible

    def simulate(self, node, env):
        # Simule une partie en jouant des coups aléatoires jusqu'à un état terminal
        state = node.state
        total_reward = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated and reward == 0:
                #Ajout de récompense manuelle pour la victoire
                total_reward += 1

        return total_reward  # Retourne la récompense finale # toujours 0 actuellement...

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent