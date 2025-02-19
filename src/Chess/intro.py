import chess

board = chess.Board()

print("____Legal Moves _____")
print(board.legal_moves)

print(chess.Move.from_uci("a8a1") in board.legal_moves)

board.push_san("e4")

print(board)
