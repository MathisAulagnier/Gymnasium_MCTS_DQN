import chess

PV = {
    'pawn': 100,
    'knight': 320,
    'bishop': 330,
    'rook': 500,
    'queen': 950
}

DRAW_VALUE = 0

def material_score(board, color_player):
    if board.is_insufficient_material():
        return DRAW_VALUE

    # Vérification d'une situation d'échec et mat
    # if board.is_checkmate():
    #     # Si c'est le tour du joueur dont nous évaluons la position et qu'il est en mat, c'est une situation perdante.
    #     if board.turn == color_player:
    #         return -10000
    #     else:
    #         return 10000

    # Comptage des pièces pour les deux couleurs
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    value = (
        PV['pawn'] * (wp - bp) +
        PV['knight'] * (wn - bn) +
        PV['bishop'] * (wb - bb) +
        PV['rook'] * (wr - br) +
        PV['queen'] * (wq - bq)
    )

    # On renverse la valeur si le joueur évalué n'est pas celui qui doit jouer actuellement.
    # Cela permet de conserver une interprétation positive lorsque l'état est favorable au joueur "color_player".
    return value if color_player == board.turn else -value


def print_tree(node, depth=0, max_depth=2):
    """
    Affiche de manière récursive l'arbre de recherche jusqu'à max_depth.
    Pour chaque nœud, on affiche le coup joué (s'il existe), le nombre de visites et le total des victoires (wins).
    """
    indent = "  " * depth
    if node.move is not None:
        move_str = node.move.uci()
    else:
        move_str = "Root"
    print(f"{indent}{move_str} - Visites: {node.visits}, Wins: {node.wins}")
    
    if depth < max_depth:
        for child in node.children:
            print_tree(child, depth + 1, max_depth)

def user_move(board):
    """
    Demande à l'utilisateur de saisir un coup jusqu'à ce qu'il saisisse un coup légal.
    """
    while True:
        try:
            move_input = input("Entrez votre coup au format UCI (ex: e2e4) : ")
            move = chess.Move.from_uci(move_input.strip())
            if move in board.legal_moves:
                return move
            else:
                print("Coup illégal. Réessayez.")
        except Exception as e:
            print("Entrée invalide. Réessayez.")

