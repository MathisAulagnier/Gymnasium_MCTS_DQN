import chess
from _class import *

board = chess.Board()
mcts = MCTS(simulations=1000)

best_move = mcts.best_move(board)
board.push(best_move)

print(board)  # Afficher l'échiquier après le coup

best_move = mcts.best_move(board)
board.push(best_move)

print(board)  # Afficher l'échiquier après le coup

best_move = mcts.best_move(board)
board.push(best_move)

print(board)  # Afficher l'échiquier après le coup
