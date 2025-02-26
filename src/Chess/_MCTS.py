import chess
import random
import math

from Chess._Node import Node

class MCTS:
    def __init__(self, simulations=1000):
        self.simulations = simulations

    def selection(self, node):
        """
        Descend dans l'arbre tant que le nœud de départ n'est pas terminal
        et est entièrement développé.
        """
        while (not node.board.is_game_over()) and node.is_fully_expanded():
            node = node.best_child()
        return node

    def expansion(self, node):
        """
        Si le nœud n'est pas terminal et non entièrement développé, ajoute
        un enfant correspondant à un coup non encore exploré.
        """
        if node.board.is_game_over():
            return None
        legal_moves = list(node.board.legal_moves)
        random.shuffle(legal_moves)  # Pour varier les choix
        for move in legal_moves:
            if not any(child.move == move for child in node.children):
                child = Node(node.board, parent=node, move=move)
                child.board.push(move)
                node.children.append(child)
                return child
        return None

    def simulation(self, node):
        """
        Joue de manière aléatoire depuis l'état du nœud jusqu'à la fin de la partie.
        Retourne la récompense (du point de vue des Blancs) :
            1  => victoire des Blancs,
           -1  => victoire des Noirs,
            0  => match nul.
        """
        board_copy = node.board.copy()
        while not board_copy.is_game_over():
            legal_moves = list(board_copy.legal_moves)
            board_copy.push(random.choice(legal_moves))
        outcome = board_copy.outcome()
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner == chess.WHITE else -1

    def backpropagation(self, node, reward):
        """
        Remonte dans l'arbre en mettant à jour le nombre de visites et le score.
        À chaque niveau, on inverse la récompense pour tenir compte de l'alternance.
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            reward = -reward  # Inverse pour le parent
            node = node.parent



    def best_move(self, root):
        """
        Effectue un nombre fixé de simulations MCTS à partir de la racine et
        retourne le coup le plus visité ainsi que le nœud associé.
        """
        for i in range(self.simulations):
            if i % 100 == 0:
                print(f"Simulation {i}/{self.simulations}")
            leaf = self.selection(root)
            child = self.expansion(leaf)
            node_to_simulate = child if child is not None else leaf
            reward = self.simulation(node_to_simulate)
            self.backpropagation(node_to_simulate, reward)

        best_child = max(root.children, key=lambda c: c.visits, default=None)
        if best_child is not None:
            return best_child.move, best_child
        else:
            # Si aucune expansion n'a été réalisée, on retourne un coup au hasard.
            return random.choice(list(root.board.legal_moves)), None
