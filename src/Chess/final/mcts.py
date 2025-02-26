import chess
import random
import math

from utils import material_score  
from node import Node   

class MCTS:
    def __init__(self, board, color_player=chess.WHITE, iterations=1000, use_heuristic=False, heuristic_weight=0.5):
        """
        Initialise la recherche MCTS.
        
        :param board: état initial du plateau
        :param color_player: la couleur du joueur pour lequel on cherche le meilleur coup
        :param iterations: nombre d'itérations de l'algorithme MCTS
        :param use_heuristic: True pour utiliser une évaluation heuristique en update
        :param heuristic_weight: coefficient de pondération de l'évaluation matérielle
        """
        self.root = Node(board)
        self.color_player = color_player
        self.iterations = iterations
        self.use_heuristic = use_heuristic
        self.heuristic_weight = heuristic_weight

    def best_move(self):
        """
        Effectue plusieurs itérations de MCTS, puis retourne le meilleur coup
        (celui qui a été le plus visité depuis la racine).
        """
        for _ in range(self.iterations):
            # 1. Sélection : descendre dans l'arbre pour trouver un noeud à développer
            leaf = self.selection()
            # 2. Simulation : à partir de ce nœud, simuler une partie jusqu'à la fin
            simulation_result = self.simulation(leaf)
            # 3. Rétropropagation : mettre à jour tous les nœuds de la feuille jusqu’à la racine
            self.backpropagation(leaf, simulation_result)
        
        # Le meilleur coup est celui dont le nœud enfant a été le plus visité.
        best_child = max(self.root.children, key=lambda child: child.visits)
        return best_child.move

    def selection(self):
        """
        Sélectionne un nœud à développer en parcourant l'arbre depuis la racine avec UCB1.
        """
        current_node = self.root
        
        # Tant que le nœud courant n'est pas terminal :
        while not current_node.is_terminal_node():
            # S'il existe encore des coups non explorés dans ce nœud, on l'étend
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                # Sinon, on choisit le meilleur enfant via UCB1 pour continuer la descente
                current_node = current_node.best_child()
        return current_node

    def simulation(self, node):
        """
        À partir du nœud fourni, exécute une simulation aléatoire (rollout)
        jusqu'à ce que l'on atteigne une position terminale.
        
        :return: un résultat numérique (1 pour victoire, 0 pour défaite, 0.5 pour match nul)
        """
        simulate_board = node.board.copy()

        while not simulate_board.is_game_over():
            moves = list(simulate_board.legal_moves)
            # Choix aléatoire d'un coup dans la liste des coups légaux
            move = random.choice(moves)
            simulate_board.push(move)

        # Détermination du résultat de la simulation
        if simulate_board.is_checkmate():
            # En cas d'échec et mat, le joueur qui doit jouer perd.
            # On compare la couleur active avec notre joueur d'intérêt.
            result = 0 if simulate_board.turn == self.color_player else 1
        else:
            # Si match nul ou autre finalité, on peut retourner 0.5
            result = 0.3

        return result

    def backpropagation(self, node, result):
        """
        Met à jour la branche de l'arbre en remontant de la feuille jusqu’à la racine.
        
        :param node: le nœud à partir duquel revenir vers la racine
        :param result: le résultat de la simulation (1, 0 ou 0.5)
        """
        while node is not None:
            # La méthode update intègre le résultat et éventuellement un bonus heuristique
            node.update(result, use_heuristic=self.use_heuristic, heuristic_weight=self.heuristic_weight, color_player=self.color_player)
            node = node.parent