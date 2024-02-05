import numpy as np
import torch as T
from torch import nn


class RandomAI:
    def __init__(self, player):
        self.player = player

    def select_move(self, _, action_space, recorder=None):
        action_probs = np.where(action_space == 1, 1 / np.sum(action_space), 0)
        action_selection = np.argmax(np.random.multinomial(1, action_probs))

        if recorder is not None:
            recorder.action.append(action_selection)
            recorder.action_probs.append(action_probs)
            recorder.v_est.append(0.0)

        action_selection = np.unravel_index(action_selection, (3, 3))
        return action_selection


class MLPAI(nn.Module):
    def __init__(self, player):
        super().__init__()
        self.player = player
        self.network = nn.Sequential(
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 9)
        )
        self.policy = nn.Sequential(
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 9)
        )
        self.value = nn.Sequential(
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 1)
        )

    def forward(self, x):
        x = self.network(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def predict_probs(self, x, action_space):
        policy_pred, value_pred = self.forward(x)
        probs = T.softmax(policy_pred - policy_pred.max(), dim=1)
        probs = T.where(action_space == 1, probs, T.zeros_like(probs))
        probs /= T.sum(probs, dim=1, keepdim=True)
        return probs, value_pred

    def select_move(self, x, action_space, recorder=None):
        x = T.from_numpy(x).float().unsqueeze(0)
        action_space = T.from_numpy(action_space).float().unsqueeze(0)
        action_probs, value_pred = self.predict_probs(x, action_space)
        action_selection = T.multinomial(action_probs, 1)

        if recorder is not None:
            recorder.action.append(action_selection.item())
            recorder.action_probs.append(action_probs.squeeze(0).detach().cpu().numpy())
            recorder.v_est.append(value_pred.detach().cpu().numpy().item())

        action_selection = np.unravel_index(action_selection.item(), (3, 3))
        return action_selection
