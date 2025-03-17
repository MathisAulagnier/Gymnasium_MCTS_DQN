import chess
import numpy as np
import random
import math

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        
    def uct_select_child(self, c_param=1.41): # c_param = sqrt(2)
        """Sélectionne un enfant selon la formule UCT"""
        # UCT = win_rate + c_param * sqrt(log(parent_visits) / child_visits)
        return max(self.children, key=lambda child: 
                  (child.wins / child.visits if child.visits > 0 else float('inf')) + 
                  c_param * math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf'))
    
    def expand(self):
        """Crée un nouvel enfant en jouant un coup non essayé"""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node
        
    def is_fully_expanded(self):
        """Vérifie si tous les coups possibles ont été essayés"""
        return len(self.untried_moves) == 0
        
    def is_terminal_node(self):
        """Vérifie si le nœud est terminal (partie terminée)"""
        return self.board.is_game_over()
