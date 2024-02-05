"""Utilities that make data manipulation and AI training cleaner"""
import torch as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class GameRecorder:
    """Helper class to record a game's data for later AI training."""
    def __init__(self):
        self.player = []
        self.state = []
        self.action_space = []
        self.action = []
        self.action_probs = []
        self.v_est = []
        self.terminal = []
        self.num_turns = 0
        self.winner = None

    def tick(self):
        assert (len(self.state) ==
                len(self.action_space) ==
                len(self.action) ==
                len(self.action_probs) ==
                len(self.player) ==
                len(self.v_est)), "Game Recording is incomplete"
        self.num_turns += 1

    def get_td_error(self, vt, vtp, player, winner):
        """Given the value estimates for t and t+1, player, and winner, calculate TD error for the player."""
        rewards = np.zeros_like(vt.values)
        if winner == player:
            rewards[-1] = 1
        elif winner == 0:
            rewards[-1] = 0
        else:
            rewards[-1] = -1

        td_error = vtp - vt + rewards
        return td_error

    def calculate_gae(self, td_error, gamma=0.99, lambda_=0.90):
        # Calculate GAE
        gae = []
        gae_t = 0
        for t in reversed(range(len(td_error))):
            delta = td_error.iloc[t]
            gae_t = delta + gamma * lambda_ * gae_t
            gae.insert(0, gae_t)
        return gae

    def record(self):
        state_df = pd.DataFrame(self.state)
        action_space_df = pd.DataFrame(self.action_space)
        action_probs_df = pd.DataFrame(self.action_probs)
        game_record = pd.concat([state_df, action_space_df, action_probs_df], axis=1)
        game_record['player'] = self.player
        game_record['action'] = self.action
        game_record['v_est'] = self.v_est
        game_record['v_est_next'] = game_record['v_est'].shift(-1, fill_value=0)
        game_record['terminal'] = self.terminal
        game_record['winner'] = self.winner
        game_record['o_td_error'] = self.get_td_error(game_record['v_est'],
                                                      game_record['v_est_next'],
                                                      player=1,
                                                      winner=self.winner,
                                                      )
        game_record['o_gae'] = self.calculate_gae(game_record['o_td_error'])
        return game_record
