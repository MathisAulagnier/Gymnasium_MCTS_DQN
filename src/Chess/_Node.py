import chess
import random
import math

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()    # Copie de l'état de l'échiquier
        self.parent = parent         # Nœud parent
        self.move = move             # Coup qui a mené à cet état
        self.children = []           # Liste des enfants
        self.visits = 0              # Nombre d'occurrences
        self.wins = 0                # Score total (récompenses cumulées)

    def is_fully_expanded(self):
        # Le nœud est entièrement développé si le nombre d'enfants correspond
        # au nombre de coups légaux possibles.
        legal_moves = list(self.board.legal_moves)
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_weight=math.sqrt(2)):
        """Sélectionne le meilleur enfant en utilisant la formule UCB1."""
        best = None
        best_val = -float("inf")
        for child in self.children:
            # On ajoute un très petit nombre pour éviter la division par zéro.
            exploitation = child.wins / (child.visits + 1e-6)
            exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            ucb1_value = exploitation + exploration
            if ucb1_value > best_val:
                best_val = ucb1_value
                best = child
        return best
