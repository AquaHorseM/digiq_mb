import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import itertools
import datetime

from tqdm import tqdm
import wandb

from models.transition_model import Transition_Model
from data.utils import ReplayBuffer
from data.utils import ReplayBufferDataset

class TransitionModel_Trainer:
    def __init__(self, state_dim:int, action_dim:int, embed_dim:int, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", device:str='cuda', load_path:str=None, save_path:str=None):
        self.state_encoder = None # TODO
        self.action_encoder = None # TODO

        self.trainsition_model = Transition_Model(state_dim, action_dim, embed_dim, num_attn_layers, num_heads, activation, device)
        self.optimizer = optim.Adam(self.trainsition_model.parameters)

        self.load_path = load_path
        self.save_path = save_path
        self.device = device

        self.load(load_path, device)

    def save(self, path):
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.trainsition_model.state_dict(), f"{path}/digiq_TransitionModel_{time}.pth")

    def load(self, path:str, device:str):
        if path:
            self.trainsition_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            self.trainsition_model.init_weight()
            self.trainsition_model.to(device)

    def loss(self, batch):
        states_ori, actions_ori, next_states_ori = batch
        with torch.no_grad:
            states = self.state_encoder(states_ori)
            actions = self.action_encoder(actions_ori)
            next_states = self.state_encoder(next_states_ori)

        next_states_pre = self.trainsition_model.forward(states, actions)
        loss = F.mse_loss(next_states_pre, next_states)

        return {"loss": loss}

    def offpolicy_TransitionModel_train_loop(self, data_path, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2):
        # step1: load and construct dataset
        assert(data_path is not None), "data path is required"

        all_trajs = torch.load(data_path, weights_only=False)
        all_data = list(itertools.chain.from_iterable(all_trajs))
        train_data = all_data[:int(len(all_data)*train_ratio)]
        val_data = all_data[int(len(all_data)*train_ratio):]
        
        train_buffer = ReplayBuffer(batch_size, capacity=capacity)
        val_buffer = ReplayBuffer(batch_size, capacity=capacity)

        for d in train_data:
            train_buffer.insert(**d)
        for d in val_data:
            val_buffer.insert(**d)
        
        train_dataset = ReplayBufferDataset(train_buffer)
        val_dataset =ReplayBufferDataset(val_buffer)

        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=batch_size)
        val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=batch_size)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

        # step2: train and val
        for batch in tqdm(train_dataloader):
            info = self.loss(batch)
            wandb.log(info)
            
            self.optimizer.zero_grad()
            info["loss"].backward()
            self.optimizer.step()

        for batch in tqdm(val_dataloader):
            info = self.loss(batch)
            wandb.log(info)

        self.save(self.save_path)

    def online_TransitionModel_train_loop(self):
        pass