import chess
import math
import random

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()    # Copie de l'état de l'échiquier
        self.parent = parent         # Nœud parent
        self.move = move             # Coup qui a mené à cet état
        self.children = []           # Liste des enfants
        self.visits = 0              # Nombre d'occurrences
        self.wins = 0                # Score total (récompenses cumulées)

    def is_fully_expanded(self):
        """Vérifie si tous les coups légaux ont été explorés."""
        legal_moves = list(self.board.legal_moves)
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_weight=math.sqrt(2)):
        """Sélectionne le meilleur enfant en utilisant la formule UCB1."""
        best = None
        best_val = -float("inf")
        for child in self.children:
            # Éviter la division par zéro
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
            ucb1_value = exploitation + exploration
            if ucb1_value > best_val:
                best_val = ucb1_value
                best = child
        return best


class MCTSPlayer:
    def __init__(self, iterations=1000):
        self.iterations = iterations
    
    def get_move(self, board):
        """Trouve le meilleur coup en utilisant MCTS."""
        # Vérifier s'il y a des coups légaux
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        root = Node(board)
        
        # Effectuer les itérations MCTS
        for _ in range(self.iterations):
            # Phase 1: Sélection
            node = self.select(root)
            
            # Phase 2: Expansion
            if not node.board.is_game_over():
                node = self.expand(node)
            
            # Phase 3: Simulation
            reward = self.simulate(node.board)
            
            # Phase 4: Propagation inverse
            self.backpropagate(node, reward)
        
        # Sélectionner le coup avec le plus grand nombre de visites
        if not root.children:
            raise RuntimeError("Aucun enfant trouvé malgré des coups légaux.")
        
        best_child = max(root.children, key=lambda x: x.visits)
        
        # Vérifier si le coup est légal avant de le retourner
        if best_child.move in legal_moves:
            return best_child.move
        else:
            # Si le coup n'est pas légal, lever une erreur
            raise RuntimeError(f"Le coup {best_child.move} n'est pas légal.")
    
    def select(self, node):
        """Phase de sélection: traverse l'arbre pour trouver un nœud à développer."""
        while not node.board.is_game_over() and node.is_fully_expanded():
            if not node.children:
                return node
            node = node.best_child()
        return node
    
    def expand(self, node):
        """Phase d'expansion: ajoute un nouvel enfant au nœud."""
        legal_moves = list(node.board.legal_moves)
        
        if not legal_moves or node.board.is_game_over():
            return node
            
        # Trouver les coups inexplorés
        explored_moves = [child.move for child in node.children]
        unexplored_moves = [move for move in legal_moves if move not in explored_moves]
        
        if unexplored_moves:
            move = random.choice(unexplored_moves)
            new_board = node.board.copy()
            new_board.push(move)
            child = Node(new_board, parent=node, move=move)
            node.children.append(child)
            return child
        
        return node
    
    def simulate(self, board):
        """Phase de simulation: joue une partie aléatoire jusqu'à la fin."""
        board_copy = board.copy()
        
        while not board_copy.is_game_over():
            legal_moves = list(board_copy.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board_copy.push(move)
        
        # Calculer la récompense
        if board_copy.is_checkmate():
            reward = 1 if board_copy.turn != board.turn else 0
        elif board_copy.is_stalemate() or board_copy.is_insufficient_material() or board_copy.is_fifty_moves() or board_copy.is_repetition():
            reward = 0.5
        else:
            reward = 0
        
        return reward
    
    def evaluate_material(self, board):
        """Évalue l'avantage matériel sur l'échiquier."""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def backpropagate(self, node, reward):
        """Phase de propagation inverse: met à jour les statistiques des nœuds."""
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent


# Code de test
if __name__ == "__main__":
    # Créer un échiquier dans la position initiale
    board = chess.Board()

    # Créer le joueur MCTS avec 1000 itérations
    mcts_player = MCTSPlayer(iterations=1000)

    # Jouer contre l'utilisateur
    while not board.is_game_over():
        print("\nÉchiquier actuel:")
        print(board)
        print(f"Au trait: {'Blancs' if board.turn == chess.WHITE else 'Noirs'}")
        
        if board.turn == chess.WHITE:
            # MCTS joue les Blancs
            move = mcts_player.get_move(board)
            if move:
                print(f"MCTS joue: {move}")
                try:
                    board.push(move)
                except AssertionError as e:
                    print(f"Erreur: {e}")
                    raise RuntimeError("Le coup MCTS est invalide.")
            else:
                print("Pas de coups légaux disponibles pour MCTS.")
        else:
            # L'utilisateur joue les Noirs
            valid_move = False
            while not valid_move:
                try:
                    move_str = input("Entrez votre coup en notation UCI (ex: e7e5): ")
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        valid_move = True
                        board.push(move)
                    else:
                        print("Coup invalide, essayez encore.")
                        print("Coups légaux:", [move.uci() for move in board.legal_moves])
                except Exception as e:
                    print(f"Erreur: {e}")
                    print("Format invalide, utilisez la notation UCI (ex: e7e5)")

    # Afficher le résultat de la partie
    print("\nPartie terminée!")
    print(board)
    if board.is_checkmate():
        winner = "Noirs" if board.turn == chess.WHITE else "Blancs"
        print(f"Échec et mat! {winner} gagnent.")
    elif board.is_stalemate():
        print("Pat! La partie est nulle.")
    elif board.is_insufficient_material():
        print("Matériel insuffisant pour mater! La partie est nulle.")
    elif board.is_fifty_moves():
        print("Règle des 50 coups! La partie est nulle.")
    elif board.is_repetition():
        print("Triple répétition! La partie est nulle.")
