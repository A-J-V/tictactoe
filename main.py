# Ultra Simple TicTacToe for RL Sanity Check
import numpy as np
import time

import ai
import ai_utils
from simulation import TicTacToe


if __name__ == '__main__':
    game = TicTacToe()
    recorder = ai_utils.GameRecorder()
    ai_x = ai.RandomAI(player=-1)
    ai_o = ai.MLPAI(player=1)
    while not game.check_terminal_all():
        # X moves
        recorder.player.append(-1)
        recorder.state.append(game.get_flat_board())
        action_space = game.get_legal_actions(recorder=recorder)
        x_action = ai_x.select_move(game.get_flat_board(), action_space, recorder=recorder)
        game.move(x_action[0], x_action[1], -1)
        end = game.check_terminal_all(recorder=recorder)
        recorder.tick()
        if end:
            break

        # O moves
        recorder.player.append(1)
        recorder.state.append(game.get_flat_board())
        action_space = game.get_legal_actions(recorder=recorder)
        o_action = ai_o.select_move(game.get_flat_board(), action_space, recorder=recorder)
        game.move(o_action[0], o_action[1], 1)
        end = game.check_terminal_all(recorder=recorder)
        recorder.tick()
        if end:
            break
    game_recording = recorder.record()

