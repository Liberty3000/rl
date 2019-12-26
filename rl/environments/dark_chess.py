import copy, gym, random
import chess, chess.svg
import numpy as np

col = 'a b c d e f g h'.split()
row = '1 2 3 4 5 6 7 8'.split()
coords = [c+r for r in row for c in col]
squares = len(coords)
h,w = len(row),len(col)

axes = ['p','n','b','r','q','k','P','N','B','R','Q','K']
piece_values = {None:0,'p':1,'n':3,'b':3,'r':5,'q':9,'k':23,
                       'P':1,'N':3,'B':3,'R':5,'Q':9,'K':23}

stochastic_policy = lambda board:random.choice([str(x) for x in board.pseudo_legal_moves])

# returns true if a piece at `src` can capture an opponent's piece at `dst`
def attackable(board, src, dst):
    if board.piece_at(src).symbol().islower() and board.piece_at(dst).symbol().isupper() or \
       board.piece_at(src).symbol().isupper() and board.piece_at(dst).symbol().islower(): return True
    else: return False

# returns a set of pieces that are observable to each player with their respective revealer
def detections(board, color=chess.WHITE):
    targets = [] # pieces that are attackable by white
    for sqr in range(squares):
        if board.piece_at(sqr) and board.is_attacked_by(color, sqr):
            for atk in board.attackers(color, sqr):
                if attackable(board, atk, sqr):
                    targets += [coords[atk] + coords[sqr]]
    return targets

# returns a partially observable view of the environment from the perspective of the attacker
def fog_of_war(board, detections, attacker=chess.WHITE):
    # squares that we can observe that are empty are 0
    arr = np.zeros((len(axes),squares))

    # squares that we can observe that are empty are 1 along the final dimension
    for sqr in range(squares):
        if board.piece_at(sqr) is None and board.is_attacked_by(attacker, sqr):
            arr[-1,sqr] = 1
        if board.piece_at(sqr) and board.is_attacked_by(attacker, sqr):
            axis = board.piece_at(sqr)
            arr[axes.index(axis.symbol()),sqr] = 1

    # squares that are observable that are not empty are 1 on the channel
    # corresponding to that piece type
    for from_to in detections:
        from_,to_ = from_to[:2], from_to[2:]
        # our pieces
        axis = board.piece_at(coords.index(from_))
        if axis: arr[axes.index(axis.symbol()),coords.index(from_)] = 1
        # their pieces
        axis = board.piece_at(coords.index(to_))
        if axis: arr[axes.index(axis.symbol()),coords.index(to_)] = 1

    return arr

def visualize(detections):
    squares,arrows = set(),set()
    for move in detections:
        squares.add(coords.index(move[2:]))
        arrows.add((coords.index(move[:2]), coords.index(move[2:])))
    return squares, arrows

class Player:
    def __init__(self, color, model=lambda a,b:random.choice(b)):
        super().__init__()
        self.color = color
        self.model = model
        self.preprocess = lambda x:fog_of_war(x, detections(x, self.color))

    def interact(self, board, logits=False):
        state = self.preprocess(board)
        legal_moves = [str(move) for move in board.pseudo_legal_moves]
        move = self.model(state, legal_moves)
        return move

class DarkChess(gym.Env):
    def __init__(self, opponent=Player(chess.BLACK), win_reward=100, verbose=False):
        self.board = chess.Board()
        self.opponent = opponent
        self.win_reward = win_reward
        self.verbose = verbose
        self.metadata = {}

    def shape_reward(self, move):
        if move == '0000': return 0
        target = self.board.piece_at(coords.index(str(move)[2:4]))
        return 0 if not target else piece_values[target.symbol().lower()]

    def legal_moves(self, board=None):
        board = self.board if not board else board
        return [str(x) for x in board.pseudo_legal_moves]

    def parameterize_state(self, board):
        legal_moves = set([str(x) for x in board.pseudo_legal_moves])
        legal_moves = np.array([move2onehot[move] for move in legal_moves])
        return np.array(legal_moves).astype(np.float32)

    def reset(self):
        self.board = chess.Board()
        return self.board

    def step(self, move):
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.board, 0, 1, self.metadata

        reward = self.shape_reward(move)
        self.board.push(chess.Move.from_uci(move))
        self.metadata['white_board'] = copy.deepcopy(self.board)
        self.metadata['white_detections'] = detections(self.board, chess.WHITE)

        if self.board.king(chess.BLACK) is None or self.board.has_insufficient_material(0):
            return self.board, self.win_reward, 1, self.metadata
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.board, reward, 1, self.metadata

        black_move = stochastic_policy(self.board)
        self.board.push(chess.Move.from_uci(black_move))
        self.metadata['black_board'] = copy.deepcopy(self.board)
        self.metadata['black_detections'] = detections(self.board, chess.BLACK)

        if self.board.king(chess.WHITE) is None or self.board.has_insufficient_material(1):
            return self.board, -self.win_reward, 1, self.metadata
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.board, reward, 1, self.metadata

        return self.board, reward, 0, self.metadata

    def render(self, size=500):
        wsquares,warrows = visualize(detections(self.board, chess.WHITE))
        bsquares,barrows = visualize(detections(self.board, chess.BLACK))
        squares = list(wsquares) + list(bsquares)
        arrows  = list(warrows) + list(barrows)
        svg = chess.svg.board(size=size, board=self.board, squares=squares, arrows=arrows)
        return svg
