import gymnasium as gym
from gymnasium import spaces
import chess
import numpy as np
import itertools




class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.int32)
        # Nombre de coups possible légaux OU non
        #self.action_space = spaces.Discrete(4032) # True_val
        self.action_space = spaces.Discrete(80) # Train_val
        #self.unplayed = 0

    def reset(self, **kwargs):
        self.board.reset()
        return self._get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        #move_want = self.action_plateau()[action]
        #move = chess.Move.from_uci(move_want)
        if action >= len(legal_moves):
            reward = -1  # Pénalité pour un coup illégal
            done = False
            return self._get_observation(), reward, done, False, {}
        move = legal_moves[action]

        '''
        if move not in legal_moves:
            self.unplayed += 1
            if self.unplayed < 30:
                # Coup illégal : pénaliser l'agent ou mapper à un coup légal
                reward = (-0.001)*self.board.halfmove_clock/2 # Pénalité pour un coup illégal
                done = False
            else:
                reward = -35  # Pénalité pour un coup illégal
                done = False
            return self._get_observation(), reward, done, False, {}
        self.unplayed = 0
        '''

        # Vérifier si le coup capture une pièce
        captured_piece = None
        if self.board.is_capture(move):
            captured_square = move.to_square
            captured_piece = self.board.piece_at(captured_square)

        # Exécuter le coup
        self.board.push(move)

        # Calculer la récompense
        reward = self._calculate_reward(captured_piece)
        done, reason, final_reward = self.is_game_over_manual()
        if done:
            print(f"La partie est terminée : {reason}")
        reward += final_reward + 0.15

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        piece_to_value = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        observation = np.zeros((8, 8), dtype=np.int32)
        for i in range(8):
            for j in range(8):
                piece = self.board.piece_at(chess.square(j, 7 - i))
                if piece:
                    observation[i, j] = piece_to_value.get(piece.symbol(), 0)
        return observation

    def _calculate_reward(self, captured_piece=None):
        """
        Calcule la récompense en fonction de l'état du jeu et des pièces capturées.
        """
        reward = 0.0

        # Récompense pour les pièces capturées
        if captured_piece:
            color = self.board.turn  # La couleur du joueur qui vient de jouer
            piece_symbol = captured_piece.symbol()

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

        return reward

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass

    def is_checkmate(self):
        return self.board.is_check() and not any(self.board.legal_moves)

    def is_stalemate(self):
        return not self.board.is_check() and not any(self.board.legal_moves)

    def is_insufficient_material(self):
        # Liste des pièces restantes
        pieces = list(self.board.piece_map().values())
        # Cas où il n'y a que les deux rois
        if len(pieces) == 2:
            return True
        # Cas où il y a un roi et un fou ou un cavalier contre un roi
        if len(pieces) == 3 and any(piece.piece_type in [chess.BISHOP, chess.KNIGHT] for piece in pieces):
            return True
        return False

    def is_fifty_moves(self):
        return self.board.halfmove_clock >= 30  # 100 demi-coups = 50 coups complets

    def is_threefold_repetition(self):
        return self.board.can_claim_threefold_repetition()

    def is_game_over_manual(self):
        # Échec et mat
        if self.is_checkmate():
            return True, "checkmate", +20.0
        # Pat
        if self.is_stalemate():
            return True, "stalemate", -1.0
        # Matériel insuffisant
        if self.is_insufficient_material():
            return True, "insufficient_material", +10.0
        # Rule of fifty moves
        if self.is_fifty_moves():
            return True, "fifty_moves", -3.0
        # Triple répétition
        if self.is_threefold_repetition():
            return True, "threefold_repetition", -5.0
        # La partie n'est pas terminée
        return False, None, 0.0

    def action_plateau(self, tab = None):
        if tab is None:
            # Générer toutes les cases de l'échiquier
            colonnes = "abcdefgh"
            lignes = "12345678"
            cases = [c + l for c in colonnes for l in lignes]

            # Générer toutes les paires possibles sans doublons
            paires = [f"{a}{b}" for a, b in itertools.permutations(cases, 2)]
            return paires
        else :
            return tab