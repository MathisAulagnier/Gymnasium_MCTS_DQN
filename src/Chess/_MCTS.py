import chess
import random
import math

from Chess._Node import Node

class MCTS:
    def __init__(self, simulations=1000):
        self.simulations = simulations

    def selection(self, node):
        """ Sélectionne le meilleur nœud en suivant la stratégie UCB1. """
        while node.children:
            node = node.best_child()
        return node

    def expansion(self, node):
        """ Étend l'arbre en ajoutant un nœud enfant correspondant à un coup possible. """
        if not node.is_fully_expanded():
            legal_moves = list(node.board.legal_moves)
            random.shuffle(legal_moves)  # Mélanger pour varier les choix
            for move in legal_moves:
                if not any(child.move == move for child in node.children):
                    new_board = node.board.copy()
                    new_board.push(move)
                    child = Node(new_board, node, move)
                    node.children.append(child)
                    return child
        return None  # Aucun coup possible

    def simulation(self, node):
        """ Joue une partie aléatoire depuis le nœud sélectionné et retourne le résultat. """
        board_copy = node.board.copy()
        
        while not board_copy.is_game_over():
            legal_moves = list(board_copy.legal_moves)
            random_move = random.choice(legal_moves)  # Choix aléatoire
            board_copy.push(random_move)

        result = board_copy.outcome().winner
        if result is None:  # Partie nulle
            return 0
        return 1 if result == chess.WHITE else -1  # Victoire ou défaite

    def backpropagation(self, node, result):
        """ Met à jour les statistiques de l'arbre en remontant les résultats. """
        while node is not None:
            node.visits += 1
            node.wins += result  # Victoire = 1, Défaite = -1, Nul = 0
            node = node.parent  # Remonte dans l'arbre

    def best_move(self, board):
        """ Trouve le meilleur coup en effectuant plusieurs simulations. """
        root = Node(board)

        for _ in range(self.simulations):
            node = self.selection(root)
            child = self.expansion(node) or node  # Expansion si possible, sinon on garde le nœud sélectionné
            result = self.simulation(child)
            self.backpropagation(child, result)

        return root.best_child(exploration_weight=0).move  # Meilleur coup basé sur le score
    
    