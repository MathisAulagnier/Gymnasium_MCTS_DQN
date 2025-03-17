import chess
import chess.engine
import numpy as np
import random
import math
import os

from node import MCTSNode

# Path Stockfish
stockfish_path = os.path.join("/Applications/Stockfish.app/Contents/MacOS", "stockfish")

class MCTS:
    def __init__(self, board, stockfish_path="stockfish"):
        """
        Initialise le MCTS avec un échiquier et le chemin vers Stockfish
        """
        self.root = MCTSNode(board)
        self.stockfish_path = stockfish_path
        self.engine = None
        self.external_engine = None  # Pour utiliser un moteur déjà ouvert
    
    def set_engine(self, engine):
        """Utilise un moteur externe déjà ouvert"""
        self.external_engine = engine
    
    def _get_engine(self):
        """Obtient une instance du moteur, priorité à l'externe s'il existe"""
        if self.external_engine:
            return self.external_engine
        
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        return self.engine
    
    def _close_engine(self):
        """Ferme l'engine Stockfish si nous l'avons créé nous-mêmes"""
        if self.engine and not self.external_engine:
            self.engine.quit()
            self.engine = None
    
    def simulation(self, node):
        """
        Phase de simulation: joue une partie jusqu'à la fin
        en utilisant Stockfish comme adversaire pour les noirs
        """
        engine = self._get_engine()
        board = node.board.copy()
        
        try:
            # Jouer la partie jusqu'à la fin avec Stockfish comme adversaire
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    # Les blancs jouent un coup aléatoire mais légèrement biaisé par le matériel
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        # Simuler chaque coup et évaluer le matériel
                        move_values = []
                        for move in legal_moves:
                            board_copy = board.copy()
                            board_copy.push(move)
                            # Si le coup capture une pièce, il est plus probable d'être choisi
                            material_after = self.material_diff(board_copy)
                            material_before = self.material_diff(board)
                            material_gain = material_after - material_before
                            # Ajouter un peu de bruit pour l'exploration
                            move_values.append(material_gain + random.uniform(0, 0.1))
                        
                        # Sélectionner un coup avec plus de chance pour les coups ayant un gain matériel
                        total = sum(max(val, 0.01) for val in move_values)  # Éviter les divisions par zéro
                        move_probs = [max(val, 0.01)/total for val in move_values]
                        selected_move = random.choices(legal_moves, weights=move_probs, k=1)[0]
                        board.push(selected_move)
                else:
                    # Les noirs jouent avec Stockfish avec un temps limité
                    result = engine.play(board, chess.engine.Limit(time=0.01))
                    board.push(result.move)
                
            # Évaluation du résultat du point de vue des blancs
            if board.is_checkmate():
                return 1 if not board.turn == chess.WHITE else 0
            else:
                # Si partie nulle, évaluer en fonction du matériel
                material_advantage = self.material_diff(board)
                if material_advantage > 0:
                    return 0.6  # Légèrement positif pour les blancs
                elif material_advantage < 0:
                    return 0.4  # Légèrement négatif pour les blancs
                else:
                    return 0.5  # Neutre
        
        except Exception as e:
            print(f"Erreur pendant la simulation: {e}")
            return 0.5
