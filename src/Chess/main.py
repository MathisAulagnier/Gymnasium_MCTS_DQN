import chess
from Chess._MCTS import MCTS
from Chess._Node import Node 
import random

def print_tree(node, indent=0, max_depth=2):
        """Affiche une partie de l'arbre jusqu'à max_depth niveaux."""
        if indent > max_depth:
            return
        prefix = "  " * indent
        if node.move:
            move_str = node.move.uci()
        else:
            move_str = "Root"
        print(f"{prefix}{move_str} | Visits: {node.visits}, Wins: {node.wins:.2f}")
        for child in node.children:
            print_tree(child, indent + 1, max_depth)


def main():
    mcts = MCTS(simulations=1000)
    
    # Création d'un plateau global.
    board = chess.Board()
    
    print("État initial de l'échiquier :")
    print(board)
    
    # Construction de la racine pour MCTS à partir de ce plateau.
    root = Node(board)
    
    print("\nLancement du MCTS...")
    best_move_found, best_child = mcts.best_move(root)
    print("\n✔ 1000 simulations effectuées.")
    print(f"Meilleur coup trouvé par le MCTS : {best_move_found}")

    print("\n🌳 Arbre partiel de recherche MCTS :")
    # print_tree(root, max_depth=2)

    
    # Mise à jour DU PLATEAU global.
    board.push(best_move_found)
    
    # Mise à jour de l'arbre en conservant le sous-arbre correspondant au coup joué.
    if best_child is not None:
        root = best_child
    else:
        root = Node(board.copy())
    
    print("\nNouvel état de l'échiquier après le coup du MCTS :")
    print(board)
    print(f"\n🔍 Nombre total de visites de la racine : {root.visits} (devrait être proche de 1000)")
    
    print("\n🔄 Simulation d'une partie contre un joueur aléatoire...\n")
    # La partie se poursuit : MCTS joue pour les Blancs et l'adversaire aléatoire pour les Noirs.
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move, _ = mcts.best_move(root)
            # print_tree(root, max_depth=1)
        else:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        
        # Mise à jour de l'arbre : si le sous-arbre correspondant existe, on le récupère.
        found = False
        for child in root.children:
            if child.move == move:
                root = child
                found = True
                break
        if not found:
            root = Node(board.copy())
        
        print(board, "\n")
    
    print("\n🎉 Partie terminée ! Résultat :", board.outcome())


if __name__ == "__main__":
    main()
