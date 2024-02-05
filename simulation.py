import numpy as np


class TicTacToe:
    """An Ultra Simple Tic-Tac-Toe Game"""

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.O_win = np.array([1, 1, 1])
        self.X_win = np.array([-1, -1, -1])

    def move(self, i, j, player):
        if self.board[i, j] == 0:
            self.board[i, j] = player
        else:
            print("That is not a valid move")

    def check_terminal_player(self, player):
        player_array = np.array([0, 0, 0]) + player
        if (self.board[:, 0] == player_array).all():
            return True
        elif (self.board[:, 1] == player_array).all():
            return True
        elif (self.board[:, 2] == player_array).all():
            return True
        elif (self.board[0, :] == player_array).all():
            return True
        elif (self.board[1, :] == player_array).all():
            return True
        elif (self.board[2, :] == player_array).all():
            return True
        elif ((self.board[0, 0] == player) and
              (self.board[1, 1] == player) and
              (self.board[2, 2] == player)
        ):
            return True
        elif ((self.board[0, 2] == player) and
              (self.board[1, 1] == player) and
              (self.board[2, 0] == player)
        ):
            return True
        else:
            return False

    def check_terminal_all(self, recorder=None):
        if self.check_terminal_player(1):
            print("O's win!")
            if recorder is not None:
                recorder.winner = 1
                recorder.terminal.append(1)
            return True
        elif self.check_terminal_player(-1):
            if recorder is not None:
                recorder.winner = -1
                recorder.terminal.append(1)
            print("X's win!")
            return True
        elif np.sum(self.get_legal_actions()) == 0:
            if recorder is not None:
                recorder.winner = 0
                recorder.terminal.append(1)
            print("Stalemate!")
            return True
        else:
            if recorder is not None:
                recorder.terminal.append(0)
            return False

    def get_flat_board(self):
        return self.board.flatten()

    def get_legal_actions(self, recorder=None):
        legal_actions = np.where(self.get_flat_board() == 0, 1, 0)
        if recorder is not None:
            recorder.action_space.append(legal_actions)
        return legal_actions
