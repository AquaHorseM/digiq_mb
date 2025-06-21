import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs

import hydra
import itertools
import datetime
import threading
from pathlib import Path

from tqdm import tqdm
import wandb

from digiq.models.transition_model import TransformerTerminal
from digiq.models.encoder import ActionEncoder, GoalEncoder
from digiq.data.utils import ReplayBuffer
from digiq.data.utils import ReplayBufferDataset

class TerminalModelTrainer:
    def __init__(self, accelerator:Accelerator=None, load_path:str=None, save_path:str=None, epoch:int=None, val_interval:int=None, print_interval:int=None,
                 state_dim:int=None, goal_dim:int=None,
                 action_encoder_backbone:str=None, action_encoder_cache_dir:str=None, goal_encoder_backbone:str=None, goal_encoder_cache_dir:str=None,
                 seed:int=None):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.accelerator = accelerator
        self.device = self.accelerator.device

        # self.state_encoder = None
        self.action_encoder = ActionEncoder(backbone=action_encoder_backbone, cache_dir=action_encoder_cache_dir, device=self.device)
        self.goal_encoder = GoalEncoder(backbone=goal_encoder_backbone, cache_dir=goal_encoder_cache_dir, device=self.device)

        #self.trainsition_model = MLPTransition(state_dim=state_dim, action_dim=action_dim, device=self.device)
        self.terminal_model = TransformerTerminal(state_dim=state_dim, goal_dim=goal_dim, device=self.device)
        
        self.optimizer = optim.Adam(self.terminal_model.parameters(), lr=1e-4)
        self.terminal_model, self.optimizer = self.accelerator.prepare(self.terminal_model, self.optimizer)

        self.load_path = load_path
        self.save_path = save_path
        self.epoch = epoch
        self.val_interval = val_interval
        self.print_interval = print_interval
        self.load(load_path, self.device)

    def save(self, path):
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.terminal_model.state_dict(), f"{path}/digiq_TerminalModel_{time}.pth")
        print(f'saved best model in {path}/digiq_TerminalModel_{time}.pth')

    def load(self, path: str, device: str):
        if path:
            self.terminal_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            self.terminal_model.init_weight()
            self.terminal_model.to(device)

    def parse_obs(self, observation):
        if type(observation) == str:
            observation = [observation]
        
        # obs example: Previous Actions: Goal: Go to newegg.com</s>
        previous_actions = []
        goals = []
        for obs in observation:
            previous_action_match = re.search(r'Previous Actions: (.*?)Goal:', obs)
            goal_match = re.search(r'Goal: (.*?)</s>', obs)
            
            # Map None to an empty string if no match is found
            previous_actions.append(previous_action_match.group(1) if previous_action_match else "")
            goals.append(goal_match.group(1) if goal_match else "")

        return previous_actions, goals

    def loss(self, batch, criterion):
        observation, action, reward, next_observation, done, mc_return, state, next_state = batch["observation"], batch["action"], batch["reward"], batch["next_observation"], batch["done"], batch["mc_return"], batch["s_rep"], batch["next_s_rep"]
        past_action, goal = self.parse_obs(observation)
        with torch.no_grad():
            goal = self.goal_encoder(goal)
        terminal_pre = self.terminal_model.forward(state, goal)
        loss = criterion(terminal_pre, reward.float())

        # compute accuracy and recall
        preds = (terminal_pre >= 0.5).long()

        #print(reward)
        true_positive = ((preds == 1) & (reward > 0)).sum().item()
        pred_positive = (preds == 1).sum().item()
        actual_positive = (reward > 0).sum().item()

        return {"loss": loss, "tp": true_positive, "pp": pred_positive, "ap": actual_positive}


    def offpolicy_train_loop(self, data_path_general, data_path_web_shop, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2, bagging=False):
        # step1: load and construct dataset
        assert data_path_general is not None, "data path is required"
        assert data_path_web_shop is not None, "data path is required"

        all_data_general = torch.load(data_path_general, weights_only=False)
        all_data_web_shop = torch.load(data_path_web_shop, weights_only=False)
        all_data = all_data_general + all_data_web_shop
        random.shuffle(all_data)

        rewards = [data['reward'] for data in all_data]
        pos_weight = torch.tensor([(len(rewards) - sum(rewards)) / sum(rewards)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss()

        train_data = all_data[:int(len(all_data)*train_ratio)]
        val_data = all_data[int(len(all_data)*train_ratio):]
        val_rewards = [data['reward'] for data in val_data]
        
        train_buffer = ReplayBuffer(False, batch_size, capacity=capacity)
        val_buffer = ReplayBuffer(False, batch_size, capacity=capacity)

        for d in train_data:
            train_buffer.insert(**d)
        for d in val_data:
            val_buffer.insert(**d)
        
        train_dataset = ReplayBufferDataset(train_buffer)
        val_dataset = ReplayBufferDataset(val_buffer)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)
        # step2: train and val
        best_f1 = 0
        for epoch in range(self.epoch):
            train_loss = []
            for batch in train_dataloader:
                with self.accelerator.accumulate(self.terminal_model):
                    train_info = self.loss(batch, criterion)
                    train_loss.append(train_info["loss"])
                    self.optimizer.zero_grad()
                    self.accelerator.backward(train_info["loss"])
                    self.optimizer.step()
                    # wandb.log(train_info)
            train_loss = sum(train_loss) / len(train_loss)

            if epoch % self.print_interval == 0:
                print(f"epoch {epoch} "f"train: {train_loss:.4f}")

            if epoch % self.val_interval == 0:
                val_loss = []
                total_tp = 0
                total_pp = 0
                total_ap = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_info = self.loss(batch, criterion)
                        val_loss.append(val_info["loss"])
                        total_tp += val_info["tp"]
                        total_pp += val_info["pp"]
                        total_ap += val_info["ap"]
                val_loss = sum(val_loss) / len(val_loss)
                precision = total_tp / total_pp if total_pp > 0 else None
                recall = total_tp / total_ap if total_ap > 0 else None

                print(f'epoch {epoch} val loss: {val_loss} precision: {precision} recall: {recall}')

                if precision is not None and recall is not None:
                    f1 = 2 * precision * recall / (precision + recall)
                    if self.accelerator.is_main_process and f1 > best_f1:
                        best_f1 = f1
                        self.save(self.save_path)


def TerminalModel_offpolicy_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60 * 60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir=config.train.save_path)

    # wandb.login(key=config.tools.wandb_key)
    # wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    trainer = TerminalModelTrainer(
        accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval, print_interval=config.train.print_interval,
        state_dim=config.TerminalModel.state_dim, goal_dim=config.TerminalModel.goal_dim,
        action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir, goal_encoder_backbone=config.Goal_encoder.goal_encoder_backbone, goal_encoder_cache_dir=config.Goal_encoder.goal_encoder_cache_dir,
        seed=config.train.seed
    )

    trainer.offpolicy_train_loop(data_path_general=config.data.data_path_general, data_path_web_shop=config.data.data_path_web_shop, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio)

@hydra.main(config_name="train_terminal", config_path="../../scripts/config/main", version_base="1.3")
def main(config):
    TerminalModel_offpolicy_train(config)

if __name__ == "__main__":
    main()