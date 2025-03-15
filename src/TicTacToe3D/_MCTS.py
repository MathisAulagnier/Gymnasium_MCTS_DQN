import numpy as np
import random
import gymnasium as gym
from _Node import Node

class MCTS:
    def __init__(self, env, iterations=1000):
        self.base_env = env
        self.iterations = iterations

    def search(self, root):
        # Sauvegarde de l'état actuel
        state_snapshot = self.base_env.unwrapped.clone_state(include_rng=True)
        # Cloner l’environnement pour éviter de modifier l’état réel
        sim_env = gym.make("ALE/TicTacToe3D-v5")  # Nouvelle instance propre
        sim_env.reset()  # Obtenir un état initial
        sim_env.unwrapped.restore_state(state_snapshot)
        
        root = Node(sim_env)
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
            print("on est ici")
            legal_moves = node.get_possible_actions()
            #legal_moves = get_legal_moves(self.env, node.state)
            #legal_moves = list(range(env.action_space.n))
            random.shuffle(legal_moves)  # Mélanger pour varier les choix
            
            for move in legal_moves:
                if not any(child.action == move for child in node.children):
                    new_state, _, terminated, truncated, _ = env.step(move)
                    if not terminated and not truncated:
                        child = Node(new_state, node, move)
                        node.children.append(child)
                        return child
        print("on est la")
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

        return total_reward  # Retourne la récompense finale # toujours 0 actuellement...

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent