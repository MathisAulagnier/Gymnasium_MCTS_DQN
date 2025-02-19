import random

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board  # État de l'échiquier
        self.parent = parent  # Nœud parent
        self.move = move  # Coup qui a mené à cet état
        self.children = []  # Liste des enfants
        self.visits = 0  # Nombre de fois visité
        self.wins = 0  # Nombre de victoires

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, exploration_weight=1.0):
        """Utilise UCB1 pour choisir le meilleur enfant."""
        return max(self.children, key=lambda child: child.wins / child.visits + exploration_weight * (2 * (self.visits)**0.5 / (1 + child.visits)))