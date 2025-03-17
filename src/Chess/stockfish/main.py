import chess
import numpy as np
import random
import math

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Coup qui a mené à cet état
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        
    def uct_select_child(self, c_param=1.41):
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

class MCTS:
    def __init__(self, board):
        """
        Initialise le MCTS avec un échiquier
        """
        self.root = MCTSNode(board)
        
    def material_diff(self, board):
        """
        Calcule la différence de matériel du point de vue des blancs
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3, 
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Le roi n'est pas compté dans le matériel
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material
    
    def selection(self):
        """Phase de sélection: parcourir l'arbre jusqu'à un nœud non complètement développé"""
        node = self.root
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.uct_select_child()
        return node
    
    def expansion(self, node):
        """Phase d'expansion: ajoute un nouveau nœud à l'arbre"""
        if not node.is_terminal_node() and not node.is_fully_expanded():
            return node.expand()
        return node
    
    def simulation(self, node):
        """
        Phase de simulation: joue une partie jusqu'à la fin avec des coups aléatoires
        mais légèrement biaisés par la capture de pièces
        """
        board = node.board.copy()
        
        try:
            # Jouer la partie jusqu'à la fin avec des coups aléatoires
            while not board.is_game_over():
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
                        # Pour les blancs, un gain matériel est positif
                        # Pour les noirs, un gain matériel est négatif par rapport à la fonction material_diff
                        if board.turn == chess.WHITE:
                            material_gain = material_after - material_before
                        else:
                            material_gain = material_before - material_after
                        
                        # Ajouter un peu de bruit pour l'exploration
                        move_values.append(material_gain + random.uniform(0, 0.1))
                    
                    # Sélectionner un coup avec plus de chance pour les coups ayant un gain matériel
                    total = sum(max(val, 0.01) for val in move_values)  # Éviter les divisions par zéro
                    move_probs = [max(val, 0.01)/total for val in move_values]
                    selected_move = random.choices(legal_moves, weights=move_probs, k=1)[0]
                    board.push(selected_move)
                
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
    
    def backpropagation(self, node, result):
        """
        Phase de rétropropagation: met à jour les statistiques des nœuds
        """
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent
    
    def get_best_move(self, simulations=10):
        """
        Exécute le MCTS et retourne le meilleur coup
        """
        for _ in range(simulations):
            # Phase 1: Sélection
            node = self.selection()
            
            # Phase 2: Expansion
            if not node.is_terminal_node():
                node = self.expansion(node)
            
            # Phase 3: Simulation
            result = self.simulation(node)
            
            # Phase 4: Rétropropagation
            self.backpropagation(node, result)
        
        # Retourner le coup qui a été le plus visité
        if not self.root.children:
            # S'il n'y a pas d'enfants, choisir un coup aléatoire
            return random.choice(list(self.root.board.legal_moves))
        
        # Imprime les statistiques des meilleurs coups pour le débogage
        for child in sorted(self.root.children, key=lambda c: c.visits, reverse=True)[:5]:
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            print(f"Coup: {child.move}, Visites: {child.visits}, Taux de victoire: {win_rate:.4f}")
        
        best_child = max(self.root.children, key=lambda child: child.visits)
        return best_child.move
    
    def update_root(self, move):
        """
        Met à jour la racine de l'arbre après que l'adversaire a joué
        """
        # Chercher si le nœud existe déjà dans les enfants
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None  # Détacher de l'ancien parent
                return
        
        # Si le coup n'a pas été exploré, créer un nouveau nœud racine
        new_board = self.root.board.copy()
        new_board.push(move)
        self.root = MCTSNode(new_board)


def get_smart_random_move(board):
    """
    Choisit un coup légèrement plus intelligent qu'aléatoire:
    - Priorise les captures
    - Évite de perdre du matériel si possible
    - Priorise les échecs
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    # Évaluer chaque coup
    move_scores = []
    for move in legal_moves:
        score = 0
        # Capture ?
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                piece_values = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3, 
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9,
                    chess.KING: 100  # Valeur arbitrairement élevée
                }
                score += piece_values.get(captured_piece.piece_type, 1)
        
        # Échec ?
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            score += 0.5
        
        # Évite de mettre sa pièce en prise
        moving_piece = board.piece_at(move.from_square)
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_attacked_by(not board.turn, move.to_square):
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3, 
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 100
            }
            score -= piece_values.get(moving_piece.piece_type, 0) * 0.8
        
        # Un peu d'aléatoire
        score += random.uniform(0, 0.2)
        
        move_scores.append((move, score))
    
    # Trier par score
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 70% du temps prendre le meilleur coup, 30% prendre un coup aléatoire
    if random.random() < 0.7:
        return move_scores[0][0]
    else:
        return random.choice(legal_moves)


def main():
    board = chess.Board()
    
    # Créer le MCTS
    mcts = MCTS(board)
    
    # Ouverture prédéfinie pour les blancs (par exemple, l'ouverture italienne)
    opening_moves = ["e2e4", "g1f3", "f1c4"]  # e4, Nf3, Bc4
    
    print("Début de la partie")
    print(board)
    
    # Jouer l'ouverture prédéfinie
    opening_index = 0
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if opening_index < len(opening_moves):
                # Jouer un coup d'ouverture
                move = chess.Move.from_uci(opening_moves[opening_index])
                opening_index += 1
                print(f"Coup d'ouverture: {move}")
            else:
                # Utiliser MCTS pour trouver le meilleur coup
                print("MCTS réfléchit...")
                move = mcts.get_best_move(simulations=1000)  # Ajustez le nombre de simulations
                print(f"MCTS a choisi: {move}")
            
            board.push(move)
        else:
            # Adversaire "semi-intelligent" pour les noirs
            move = get_smart_random_move(board)
            print(f"Adversaire a joué: {move}")
            board.push(move)
            
            # Mettre à jour la racine de l'arbre MCTS
            mcts.update_root(move)
        
        print(board)
        print("-------------------")
    
    print("Partie terminée")
    print(f"Résultat: {board.result()}")

if __name__ == "__main__":
    main()
