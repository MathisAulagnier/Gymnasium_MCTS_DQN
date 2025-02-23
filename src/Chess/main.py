import chess
from Chess._Node import Node
from Chess._MCTS import MCTS
import random



# board = chess.Board() # Créer un échiquier
# mcts = MCTS(simulations=1000) # Créer un objet MCTS
# print(board.legal_moves)  # Afficher les coups légaux
# best_move = mcts.best_move(board) # Trouver le meilleur coup grâce à MCTS
# board.push(best_move) # Jouer le coup
# user_move = chess.Move.from_uci("a7a5") # Créer un coup utilisateur
# print(user_move in board.legal_moves) # Vérifier si le coup est légal

# Créer une partie d'échecs contre l'utilisateur
def main(player_color=random.choice([chess.WHITE, chess.BLACK])):
    board = chess.Board()
    mcts = MCTS(simulations=1000)

    print("Bienvenue ! Vous jouez les Blancs. Entrez votre coup en notation UCI (ex: e2e4).")
    print(board)

    while not board.is_game_over():
        # Tour du joueur
        if board.turn == player_color:
            move = None
            while move not in board.legal_moves:
                user_input = input("Votre coup (UCI ex: e2e4) : ")
                try:
                    move = chess.Move.from_uci(user_input)
                    if move not in board.legal_moves:
                        print("Coup illégal, réessayez.")
                        print("Coups légaux (UCI) : ", [board.parse_san(str(move)) for move in board.legal_moves])
                except ValueError:
                    print("Format invalide, réessayez.")
            board.push(move)
        
        # Tour de l'IA
        else:
            print("\nL'ordinateur réfléchit...")
            move = mcts.best_move(board)
            board.push(move)
            print(f"L'ordinateur joue : {move}")

        # Affichage de l'échiquier après chaque coup
        print(board)

    # Fin du jeu
    outcome = board.outcome()
    if outcome.winner is None:
        print("Partie nulle !")
    elif outcome.winner == chess.WHITE:
        print("Vous avez gagné ! 🎉")
    else:
        print("L'ordinateur a gagné... 😞")

if __name__ == "__main__":
    main()