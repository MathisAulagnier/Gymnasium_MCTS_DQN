import random
from src.Chess.node import Node

class MCTS:
    def __init__(self, simulations=1000):
        self.simulations = simulations

    def selection(self, node):
        return node.best_child()

    def expansion(self, node):
        if not node.is_fully_expanded():
            legal_moves = list(node.board.legal_moves)
            random.shuffle(legal_moves)  # Mélange les coups pour éviter toujours le même choix
            for move in legal_moves:
                if not any(child.move == move for child in node.children):
                    new_board = node.board.copy()
                    new_board.push(move)
                    child = Node(new_board, node, move)  # Suppression de chess.node
                    node.children.append(child)
                    return child
        return None  # Aucun coup possible

    def simulation(self, node):
        """
        Simule une partie aléatoire depuis le nœud sélectionné et retourne le résultat.

        Args:
            node (Node): Le nœud de départ.

        Returns:
            int: 1 si victoire, 0 si nul, -1 si défaite.
        """
        pass

    def backpropagation(self, node, result):
        """
        Met à jour les statistiques de l'arbre en remontant les résultats.

        Args:
            node (Node): Le nœud de départ.
            result (int): Le résultat de la simulation (1, 0, -1).
        """
        pass

    def best_move(self, board):
        """
        Trouve le meilleur coup à jouer depuis un état donné en effectuant plusieurs simulations.

        Args:
            board (chess.Board): L'échiquier actuel.

        Returns:
            chess.Move: Le meilleur coup trouvé.
        """
        pass