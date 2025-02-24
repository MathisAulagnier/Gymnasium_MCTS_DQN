import gymnasium as gym
import ale_py
import random
import math
from _Node import Node

def get_legal_moves(env, state):
    """Retourne uniquement les coups valides pour le joueur actuel en respectant les règles des dames."""
    legal_moves = []
    capture_moves = []  # Liste des coups de capture (prises obligatoires)

    for action in range(env.action_space.n):  # Parcours toutes les actions possibles
        new_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            continue  # Ignorer les actions qui mènent à une fin de partie

        # Vérifier si l'action implique une capture
        if is_capture_move(state, new_state):
            capture_moves.append(action)
        else:
            legal_moves.append(action)

    # Si des prises sont possibles, elles sont obligatoires
    print(legal_moves)
    return capture_moves if capture_moves else legal_moves


def is_capture_move(prev_state, new_state):
    """Vérifie si le mouvement entre prev_state et new_state est une capture."""
    return count_pieces(prev_state) > count_pieces(new_state)


def count_pieces(state):
    """Compte le nombre de pions sur l'échiquier."""
    return sum(1 for cell in state.flatten() if cell != 0)  # Exclut les cases vides


class MCTS:
    def __init__(self, env, simulations=1000):
        self.env = env  # Stocker l'environnement
        self.simulations = simulations

    def selection(self, node):
        """ Sélectionne le meilleur nœud en suivant la stratégie UCB1. """
        while node.children:
            node = node.best_child()
        return node

    def expansion(self, node):
        """Étend l'arbre en ajoutant un nœud enfant correspondant à un coup valide."""
        if not node.is_fully_expanded(self.env):
            legal_moves = get_legal_moves(self.env, node.state)  # Récupère uniquement les bons coups

            for move in legal_moves:
                if not any(child.action == move for child in node.children):
                    new_state, _, terminated, truncated, _ = self.env.step(move)
                    if not terminated and not truncated:
                        child = Node(new_state, node, move)
                        node.children.append(child)
                        return child
        return None  # Aucun coup possible

    def simulation(self, node):
        """ Joue une partie aléatoire depuis le nœud sélectionné et retourne le score. """
        state = node.state
        total_reward = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = self.env.action_space.sample()
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward

        return total_reward

    def backpropagation(self, node, result):
        """ Met à jour les statistiques de l'arbre en remontant les résultats. """
        while node is not None:
            node.visits += 1
            node.wins += result  # Utiliser le score au lieu de victoire/défaite
            node = node.parent

    def best_move(self, state):
        """ Trouve la meilleure action en effectuant plusieurs simulations. """
        root = Node(state)

        for _ in range(self.simulations):
            node = self.selection(root)
            child = self.expansion(node) or node
            result = self.simulation(child)
            self.backpropagation(child, result)

        return root.best_child(exploration_weight=0).action  # Retourne la meilleure action
