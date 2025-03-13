import gymnasium as gym
from gymnasium import spaces
import chess
import numpy as np


class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.int32)
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))  # Nombre de coups légaux actuels

    def reset(self, **kwargs):
        self.board.reset()
        return self._get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if action >= len(legal_moves):
            action = 0  # Sécurité : si l'action est invalide, choisir le premier coup légal
        move = legal_moves[action]

        # Vérifier si le coup capture une pièce
        captured_piece = None
        if self.board.is_capture(move):
            captured_square = move.to_square
            captured_piece = self.board.piece_at(captured_square)

        #print(self.board)
        # Exécuter le coup
        self.board.push(move)

        # Calculer la récompense
        reward = self._calculate_reward(captured_piece)
        done = self.board.is_game_over()

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        # Convertir le plateau en une représentation numérique
        return np.zeros((8, 8), dtype=np.int32)  # Exemple simplifié

    def _calculate_reward(self, captured_piece=None):
        """
        Calcule la récompense en fonction de l'état du jeu et des pièces capturées.
        """
        reward = 0.0

        # Récompense pour les pièces capturées
        if captured_piece:
            color = self.board.turn  # La couleur du joueur qui vient de jouer
            piece_symbol = captured_piece.symbol()
            #print(piece_symbol)

            if color:  # Si l'agent joue les noirs
                if piece_symbol == "P":
                    reward += 0.1  # Récompense pour la capture d'un pion blanc
                elif piece_symbol == "N" or piece_symbol == "B":
                    reward += 0.5  # Récompense pour la capture d'un cavalier ou fou blanc
                elif piece_symbol == "R":
                    reward += 1.0  # Récompense pour la capture d'une tour blanche
                elif piece_symbol == "Q":
                    reward += 5.0  # Récompense pour la capture d'une reine blanche
            else:  # Si l'agent joue les blancs
                if piece_symbol == "p":
                    reward += 0.1  # Récompense pour la capture d'un pion noir
                elif piece_symbol == "n" or piece_symbol == "b":
                    reward += 0.5  # Récompense pour la capture d'un cavalier ou fou noir
                elif piece_symbol == "r":
                    reward += 1.0  # Récompense pour la capture d'une tour noire
                elif piece_symbol == "q":
                    reward += 5.0  # Récompense pour la capture d'une reine noire

        # Récompense pour les résultats de la partie
        if self.board.is_game_over():
            if self.board.is_checkmate():
                if self.board.turn:  # Si c'est au tour des noirs (les blancs ont gagné)
                    reward += 10.0  # Récompense pour un échec et mat avec les blancs
                else:
                    reward -= 10.0  # Sanction pour un échec et mat avec les noirs
            else:  # Match nul
                reward -= 0.5  # Sanction pour un match nul

        return reward

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass