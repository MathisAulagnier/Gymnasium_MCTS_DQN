import chess
import random
import math

# =============================================================================
# Classes MCTS et Node
# =============================================================================

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()    # Copie de l'état de l'échiquier
        self.parent = parent         # Nœud parent
        self.move = move             # Coup qui a mené à cet état
        self.children = []           # Liste des enfants
        self.visits = 0              # Nombre de visites
        self.wins = 0                # Récompense cumulée

    def is_fully_expanded(self):
        # Un nœud est entièrement développé lorsque tous les coups légaux ont été explorés.
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, exploration_weight=math.sqrt(2)):
        best = None
        best_val = -float("inf")
        for child in self.children:
            exploitation = child.wins / (child.visits + 1e-6)
            exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            ucb1_value = exploitation + exploration
            if ucb1_value > best_val:
                best_val = ucb1_value
                best = child
        return best


class MCTS:
    def __init__(self, simulations=1000):
        self.simulations = simulations

    def selection(self, node):
        """
        Descend dans l'arbre à partir de 'node' tant que ce nœud n'est pas terminal
        et est entièrement développé.
        """
        while not node.board.is_game_over() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def expansion(self, node):
        """
        Ajoute un enfant représentant un coup non encore exploré tant que le nœud n'est pas terminal.
        """
        if node.board.is_game_over():
            return None
        legal_moves = list(node.board.legal_moves)
        random.shuffle(legal_moves)  # Pour introduire de la diversité
        for move in legal_moves:
            if not any(child.move == move for child in node.children):
                child = Node(node.board, parent=node, move=move)
                child.board.push(move)
                node.children.append(child)
                return child
        return None

    def simulation(self, node):
        """
        Effectue une simulation aléatoire à partir de l'état du nœud jusqu'à la fin de la partie.
        Renvoie la récompense du point de vue des Blancs:
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
        Remonte dans l'arbre en mettant à jour le nombre de visites et les gains.
        La récompense est inversée à chaque niveau pour tenir compte du changement de tour.
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            reward = -reward  # inversion pour le parent
            node = node.parent

    def best_move(self, root):
        """
        Effectue un certain nombre de simulations à partir de la racine et
        renvoie le coup le plus visité (ainsi que le nœud associé).
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
            return random.choice(list(root.board.legal_moves)), None


# =============================================================================
# Partie joueur vs IA
# =============================================================================

def main():
    board = chess.Board()
    mcts = MCTS(simulations=1000)

    # Choix de la couleur par l'utilisateur
    user_color = None
    while user_color not in ["W", "B"]:
        user_color = input("Voulez-vous jouer avec les Blancs (W) ou les Noirs (B) ? (W/B) : ").strip().upper()
    player_color = chess.WHITE if user_color == "W" else chess.BLACK

    # La racine de l'arbre MCTS est basée sur l'état du plateau courant.
    root = Node(board)

    print("\nEntrez votre coup en notation UCI (ex: e2e4).")
    print(board, "\n")

    while not board.is_game_over():
        if board.turn == player_color:
            # Tour du joueur
            move = None
            while move not in board.legal_moves:
                user_input = input("Votre coup (UCI ex: e2e4) : ")
                try:
                    move = chess.Move.from_uci(user_input)
                    if move not in board.legal_moves:
                        print("Coup illégal, réessayez.")
                except ValueError:
                    print("Format invalide, réessayez.")
            board.push(move)

            # Mise à jour de l'arbre MCTS
            found = False
            for child in root.children:
                if child.move == move:
                    root = child
                    found = True
                    break
            if not found:
                root = Node(board.copy())
        else:
            # Tour de l'IA
            print("Tour de l'IA...")
            ai_move, ai_child = mcts.best_move(root)
            print(f"L'IA joue : {ai_move}")
            board.push(ai_move)
            if ai_child is not None:
                root = ai_child
            else:
                root = Node(board.copy())

        print("\n" + str(board) + "\n")

    outcome = board.outcome()
    print("🎉 Partie terminée !")
    if outcome.winner is None:
        print("Match nul")
    elif outcome.winner == player_color:
        print("Félicitations, vous avez gagné !")
    else:
        print("L'IA a gagné !")


if __name__ == "__main__":
    main()