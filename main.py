"""Ultra Simple TicTacToe for RL Sanity Check"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import ai
import ai_utils
from simulation import TicTacToe

avg_wins = []
if __name__ == '__main__':
    ai_o = ai.PpoAttentionAgent(player=1)
    device = 'cuda'
    ai_o = ai_o.to(device)
    optimizer = torch.optim.Adam(ai_o.parameters(), lr=0.0001)
    for iteration in tqdm(range(50)):
        winners = []
        for i in range(5000):
            game = TicTacToe()
            recorder = ai_utils.GameRecorder()
            ai_x = ai.RandomAI(player=-1)
            while game.check_terminal_all() == 100:

                # X moves
                recorder.player.append(-1)
                recorder.state.append(game.get_flat_board())
                action_space = game.get_legal_actions(recorder=recorder)
                x_action = ai_x.select_move(game.get_flat_board(), action_space, recorder=recorder)
                game.move(x_action[0], x_action[1], -1)
                end = game.check_terminal_all(recorder=recorder)
                recorder.tick()
                if end != 100:
                    winners.append(end)
                    break

                # O moves
                recorder.player.append(1)
                recorder.state.append(game.get_flat_board())
                action_space = game.get_legal_actions(recorder=recorder)
                o_action = ai_o.select_move(game.get_flat_board(), action_space, recorder=recorder)
                game.move(o_action[0], o_action[1], 1)
                end = game.check_terminal_all(recorder=recorder)
                recorder.tick()
                if end != 100:
                    winners.append(end)
                    break
            game_recording = recorder.record()
            game_recording.to_csv(f"./game_records/game_{i}.csv", index=False)

        dataset = ai_utils.TicTacToeDataset(data_path="./game_records", player=1)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=512,
                                shuffle=True)

        loss_fn = ai_utils.PPOLoss(c2=0.0001)
        for epoch in range(3):
            ai_utils.train_agent(ai_o, loss_fn, device, dataloader, optimizer)

        files = os.listdir("./game_records")
        for file in files:
            file_path = os.path.join("./game_records", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        avg = sum(winners) / len(winners)
        print(f"Average Reward: {avg}")
        avg_wins.append(avg)
        torch.save(ai_o.state_dict(), f"./model_checkpoints/attn_checkpoint_{iteration}.pth")
torch.save(ai_o, f"attn_ppo_o.pth")
fig = plt.Figure()
plt.plot(avg_wins)
plt.savefig("./attn_rewards_plot.png")
