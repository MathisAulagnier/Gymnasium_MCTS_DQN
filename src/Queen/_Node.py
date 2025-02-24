import gymnasium as gym
import ale_py
import math

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # État du jeu
        self.parent = parent  # Nœud parent
        self.action = action  # Action ayant mené à cet état
        self.children = []  # Liste des enfants
        self.visits = 0  # Nombre de fois visité
        self.wins = 0  # Score accumulé

    def is_fully_expanded(self, env):
        """ Vérifie si tous les enfants ont été explorés """
        return len(self.children) == env.action_space.n

    def best_child(self, exploration_weight=math.sqrt(2)):
        """Utilise UCB1 pour choisir le meilleur enfant."""
        return max(
            self.children,
            key=lambda child: (child.wins / (child.visits + 1e-6)) +
            exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )
