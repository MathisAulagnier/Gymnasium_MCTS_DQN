import chess
import random
import math

from mcts import MCTS
from node import Node
from utils import *

def update_mcts_root(mcts_instance, move, board):
    """
    Met à jour la racine de l'arbre MCTS en gardant le sous‐arbre
    correspondant au coup joué (pour la persistance entre les coups).
    Si le coup n'existe pas dans l'arbre, on recrée une nouvelle instance de MCTS.
    """
    found = False
    for child in mcts_instance.root.children:
        if child.move == move:
            child.parent = None  # détachement du sous-arbre
            mcts_instance.root = child
            found = True
            break
    if not found:
        # Si le sous-arbre n'a pas été trouvé (cas improbable)
        mcts_instance.root = Node(board)
    return mcts_instance  # On retourne l'instance mise à jour.

def main():
    # Choix de la couleur pour le joueur humain
    human_color_input = ""
    while human_color_input.lower() not in ["w", "b"]:
        human_color_input = input("Choisissez votre couleur ([W]hite ou [B]lack) : ")
    human_color = chess.WHITE if human_color_input.lower() == "w" else chess.BLACK
    ai_color = not human_color

    # Création du plateau initial.
    board = chess.Board()
    
    # Paramètres de MCTS pour l'IA
    iterations = 1000          # Nombre d'itérations pour MCTS à chaque coup de l'IA
    use_heuristic = True
    heuristic_weight = 0.01

    print("Début de la partie !")
    print(board)

    # Création persistante de l'instance MCTS à partir de la position initiale.
    mcts = MCTS(board, color_player=ai_color, iterations=iterations, 
                use_heuristic=use_heuristic, heuristic_weight=heuristic_weight)

    while not board.is_game_over():
        if board.turn == human_color:
            print("\nVotre tour.")
            move = user_move(board)
            board.push(move)
            # Mettre à jour l'arbre MCTS en fonction du coup joué par l'humain.
            mcts = update_mcts_root(mcts, move, board)
            # Affiche l'évaluation de la position après le coup du joueur
            print("Évaluation de la position après votre coup :", material_score(board, human_color))
        else:
            print("\nTour de l'IA (MCTS). Réflexion en cours ...")
            # Ici, on délègue à l'arbre déjà existant afin de ne pas repartir de zéro.
            best_move = mcts.best_move()

            # Affichage d'informations sur l'arbre de recherche
            print("\n-- Informations sur l'arbre MCTS --")
            print(f"Nombre total de visites à la racine : {mcts.root.visits}")
            if mcts.root.children:
                for child in mcts.root.children:
                    move_str = child.move.uci() if child.move is not None else "N/A"
                    print(f"Coup {move_str} : {child.visits} visites, wins = {child.wins}")
            print("\nArbre de recherche (affichage limité à 2 niveaux) :")
            print_tree(mcts.root, max_depth=2)
            print("------------------------------------\n")
            
            print("L'IA joue :", best_move)
            board.push(best_move)
            # On met à jour l'arbre MCTS à partir du coup joué par l'IA.
            mcts = update_mcts_root(mcts, best_move, board)
        
        # Affiche le plateau après chaque coup.
        print("\n" + board.unicode())

    # Fin de partie.
    print("\nLa partie est terminée.")
    if board.is_checkmate():
        winner = "Humain" if board.turn != human_color else "IA"
        print(f"Échec et mat ! Le gagnant est : {winner}")
    elif board.is_stalemate():
        print("Partie nulle par pat.")
    elif board.is_insufficient_material():
        print("Partie nulle par matériel insuffisant.")
    else:
        print("Partie terminée par une autre règle.")

if __name__ == "__main__":
    main()