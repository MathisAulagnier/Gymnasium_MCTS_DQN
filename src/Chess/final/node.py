import math
import random
import chess
from utils import material_score

class Node:
    def __init__(self, board, move=None, parent=None):
        """
        Initialisation d'un nœud de l'arbre MCTS.
        
        :param board: instance du plateau de jeu (par exemple, un objet board de pychess)
        :param move: le coup qui a permis d'atteindre cet état (None pour la racine)
        :param parent: nœud parent (None pour la racine)
        """
        self.board = board              # L'état du plateau à ce nœud
        self.move = move                # Le coup qui a mené à cet état
        self.parent = parent            # Le nœud parent
        self.children = []              # Liste des enfants
        self.wins = 0                   # Gain cumulé des simulations
        self.visits = 0                 # Nombre de fois que ce nœud a été visité
        
        # Liste des coups légaux non encore explorés board.legal_moves
        self.untried_moves = list(board.legal_moves) if board is not None else []
    
    def ucb1(self, exploration_constant=1.41):
        """
        Calcule la valeur UCB1 pour ce nœud.
        
        :param exploration_constant: paramètre d'exploration, souvent noté "C"
        :return: valeur UCB1
        """
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, exploration_constant=1.41):
        """
        Sélectionne et renvoie l'enfant avec le meilleur score UCB1.
        
        :param exploration_constant: paramètre d'exploration
        :return: le nœud enfant avec la meilleure valeur UCB1
        """
        return max(self.children, key=lambda child: child.ucb1(exploration_constant))
    
    def expand(self):
        """
        Développe un enfant à partir d'un coup non encore exploré.
        
        :return: le nœud enfant nouvellement créé
        """
        # Sélectionner un coup non encore exploré
        move = self.untried_moves.pop()
        # Copier le plateau pour appliquer le coup
        next_board = self.board.copy()  # On suppose que board.copy() existe pour dupliquer l'état
        next_board.push(move)            # Appliquer le coup au plateau
        # Créer le nouveau nœud enfant
        child_node = Node(next_board, move, self)
        self.children.append(child_node)
        return child_node
    
    def update(self, result, use_heuristic=False, heuristic_weight=1e-4, color_player=chess.WHITE):
        """
        Met à jour le nœud avec le résultat d'une simulation.
        
        :param result: le résultat de la simulation (par exemple, 1 pour une victoire, 0 pour une défaite, éventuellement 0.5 pour un match nul)
        :param use_heuristic: True si l'on souhaite intégrer un bonus basé sur l'évaluation matérielle
        :param heuristic_weight: coefficient de pondération de l'évaluation matérielle
        :param color_player: la couleur du joueur pour lequel on souhaite évaluer l'état
        """
        if use_heuristic:
            # Calcul du score matériel en passant la couleur du joueur évalué
            material = material_score(self.board, color_player)
            # On ajoute au résultat le bonus (ou malus) de l'évaluation matérielle multiplié par un coefficient
            self.wins += material # result + heuristic_weight * material
        else:
            self.wins += result
        self.visits += 1

    
    def is_fully_expanded(self):
        """
        Vérifie si le nœud a été entièrement développé (tous les coups possibles ont été explorés).
        
        :return: True si tous les coups ont été explorés, False sinon.
        """
        return len(self.untried_moves) == 0
    
    def is_terminal_node(self):
        """
        Vérifie si le nœud représente une position terminale (fin de la partie).
        
        :return: True si la partie est terminée à partir de ce plateau, False sinon.
        """
        return self.board.is_game_over()  # On suppose que board peut fournir cette information.
