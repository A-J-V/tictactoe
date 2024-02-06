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
            recorder.actions.append(action_selection)
            recorder.action_probs.append(action_probs[action_selection])
            recorder.v_est.append(0.0)

        action_selection = np.unravel_index(action_selection, (3, 3))
        return action_selection


class PpoMlpAgent(nn.Module):
    def __init__(self, player):
        super().__init__()
        self.player = player
        self.network = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
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
        if T.isnan(probs).any():
            raise Exception("NAN DETECTED!")
        return probs, value_pred

    def select_move(self, x, action_space, recorder=None, device='cuda'):
        with T.inference_mode():
            x = T.from_numpy(x).float().unsqueeze(0).to(device)
            action_space = T.from_numpy(action_space).float().unsqueeze(0).to(device)
            action_probs, value_pred = self.predict_probs(x, action_space)
        try:
            action_selection = T.multinomial(action_probs, 1)
        except:
            print(x)
            print(action_probs)
            print(action_space)
            raise Exception("RuntimeError: probability tensor contains either `inf`, `nan` or element < 0")

        if recorder is not None:
            recorder.actions.append(action_selection.item())
            action_probs = action_probs.squeeze(0).detach().cpu().numpy()
            recorder.action_probs.append(action_probs[action_selection])
            recorder.v_est.append(value_pred.detach().cpu().numpy().item())

        action_selection = np.unravel_index(action_selection.item(), (3, 3))
        return action_selection


class PpoAttentionAgent(nn.Module):
    def __init__(self, player):
        super().__init__()
        self.player = player
        board_size = 3
        position_tensor = T.zeros((2, board_size, board_size))

        for i in range(position_tensor.shape[-1]):
            for j in range(position_tensor.shape[-2]):
                position_tensor[0, i, j] = i
                position_tensor[1, i, j] = j
        position_tensor = position_tensor / 10
        self.position_tensor = position_tensor.unsqueeze(0)

        self.conv = nn.Conv2d(1, 30, kernel_size=(3, 3), stride=1, padding='same')
        self.attn = nn.MultiheadAttention(32, 1, batch_first=True)
        self.norm1 = nn.LayerNorm(32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        self.norm2 = nn.LayerNorm(32)

        self.policy = nn.Sequential(
            nn.Linear(32 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )
        self.value = nn.Sequential(
            nn.Linear(32 * 9, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, 1, 3, 3))
        x = self.conv(x)
        batch_position = self.position_tensor.to(x.device.type).expand(x.shape[0], 2, 3, 3)
        x = T.cat((batch_position, x), dim=1)

        x = x.view(x.shape[0], x.shape[1], x.shape[2] ** 2)
        x = x.permute(0, 2, 1)

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(attn_out + x)
        mlp_out = self.mlp(x)
        x = self.norm2(mlp_out + x)

        x = x.view(x.size(0), -1)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def predict_probs(self, x, action_space):
        policy_pred, value_pred = self.forward(x)
        probs = T.softmax(policy_pred - policy_pred.max(), dim=1)
        probs = T.where(action_space == 1, probs, T.zeros_like(probs))
        probs /= T.sum(probs, dim=1, keepdim=True)
        if T.isnan(probs).any():
            print("POLICY PRED")
            print(policy_pred)
            print("POLICY PROBS")
            print(probs)
            raise Exception("NAN DETECTED!")
        return probs, value_pred

    def select_move(self, x, action_space, recorder=None, device='cuda'):
        with T.inference_mode():
            x = T.from_numpy(x).float().unsqueeze(0).to(device)
            action_space = T.from_numpy(action_space).float().unsqueeze(0).to(device)
            action_probs, value_pred = self.predict_probs(x, action_space)
        try:
            action_selection = T.multinomial(action_probs, 1)
        except:
            print(x)
            print(action_probs)
            print(action_space)
            raise Exception("RuntimeError: probability tensor contains either `inf`, `nan` or element < 0")

        if recorder is not None:
            recorder.actions.append(action_selection.item())
            action_probs = action_probs.squeeze(0).detach().cpu().numpy()
            recorder.action_probs.append(action_probs[action_selection])
            recorder.v_est.append(value_pred.detach().cpu().numpy().item())

        action_selection = np.unravel_index(action_selection.item(), (3, 3))
        return action_selection


class PpoCnnAgent(nn.Module):
    def __init__(self, player):
        super().__init__()
        self.player = player
        self.network = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same')

        self.policy = nn.Sequential(
            nn.Linear(288, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )
        self.value = nn.Sequential(
            nn.Linear(288, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, 1, 3, 3))
        x = self.network(x)
        x = x.view(batch_size, -1)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def predict_probs(self, x, action_space):
        policy_pred, value_pred = self.forward(x)
        probs = T.softmax(policy_pred - policy_pred.max(), dim=1)
        probs = T.where(action_space == 1, probs, T.zeros_like(probs))
        probs /= T.sum(probs, dim=1, keepdim=True)
        if T.isnan(probs).any():
            print("POLICY PRED")
            print(policy_pred)
            print("POLICY PROBS")
            print(probs)
            raise Exception("NAN DETECTED!")
        return probs, value_pred

    def select_move(self, x, action_space, recorder=None, device='cuda'):
        with T.inference_mode():
            x = T.from_numpy(x).float().unsqueeze(0).to(device)
            action_space = T.from_numpy(action_space).float().unsqueeze(0).to(device)
            action_probs, value_pred = self.predict_probs(x, action_space)
        try:
            action_selection = T.multinomial(action_probs, 1)
        except:
            print(x)
            print(action_probs)
            print(action_space)
            raise Exception("RuntimeError: probability tensor contains either `inf`, `nan` or element < 0")

        if recorder is not None:
            recorder.actions.append(action_selection.item())
            action_probs = action_probs.squeeze(0).detach().cpu().numpy()
            recorder.action_probs.append(action_probs[action_selection])
            recorder.v_est.append(value_pred.detach().cpu().numpy().item())

        action_selection = np.unravel_index(action_selection.item(), (3, 3))
        return action_selection
