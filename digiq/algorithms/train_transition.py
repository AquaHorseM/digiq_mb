import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs

import hydra
import itertools
import datetime
import threading

from tqdm import tqdm
import wandb

from digiq.models.transition_model import Transition_Model
from digiq.models.encoder import ActionEncoder
from digiq.data.utils import TransitionReplayBuffer
from digiq.data.utils import TransitionReplayBufferDataset

class TransitionModel_Trainer:
    def __init__(self, accelerator:Accelerator=None, load_path:str=None, save_path:str=None, epoch:int=None, val_interval:int=None,
                 state_dim:int=None, action_dim:int=None, embed_dim:int=None, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", 
                 action_encoder_backbone:str=None, action_encoder_cache_dir:str=None):
        self.accelerator = accelerator
        self.device = self.accelerator.device

        # self.state_encoder = None
        self.action_encoder = ActionEncoder(backbone=action_encoder_backbone, cache_dir=action_encoder_cache_dir, device=self.device)

        self.trainsition_model = Transition_Model(state_dim, action_dim, embed_dim, num_attn_layers, num_heads, activation, self.device)
        self.optimizer = optim.Adam(self.trainsition_model.parameters())
        self.trainsition_model, self.optimizer = self.accelerator.prepare(self.trainsition_model, self.optimizer)

        self.load_path = load_path
        self.save_path = save_path
        self.epoch = epoch
        self.val_interval = val_interval
        self.load(load_path, self.device)

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
        states, actions, next_states = batch['state'], batch['action'], batch['next_state']
        with torch.no_grad():
            actions = self.action_encoder(actions)

        next_states_pre = self.trainsition_model.forward(states, actions)
        loss = F.mse_loss(next_states_pre, next_states)

        return {"loss": loss}

    def offpolicy_train_loop(self, data_path, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2):
        # step1: load and construct dataset
        assert(data_path is not None), "data path is required"

        all_data = torch.load(data_path, weights_only=False)
        train_data = all_data[:int(len(all_data)*train_ratio)]
        val_data = all_data[int(len(all_data)*train_ratio):]
        
        train_buffer = TransitionReplayBuffer(batch_size, capacity=capacity)
        val_buffer = TransitionReplayBuffer(batch_size, capacity=capacity)

        for d in train_data:
            train_buffer.insert(**d)
        for d in val_data:
            val_buffer.insert(**d)
        
        train_dataset = TransitionReplayBufferDataset(train_buffer)
        val_dataset =TransitionReplayBufferDataset(val_buffer)

        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=batch_size)
        val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=batch_size)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)

        # step2: train and val
        best_loss = float("inf")
        for epoch in range(self.epoch):
            for batch in train_dataloader:
                train_info = self.loss(batch)
                wandb.log(train_info)
                
            with self.accelerator.accumulate(self.trainsition_model):
                self.optimizer.zero_grad()
                self.accelerator.backward(train_info["loss"])
                self.optimizer.step()

            if epoch % self.val_interval == 0:
                for batch in val_dataloader:
                    val_info = self.loss(batch)
                    wandb.log(val_info)

                print(f'epoch {epoch} train loss: {train_info["loss"]} val loss: {val_info["loss"]}')
                if self.accelerator.is_main_process and val_info["loss"] < best_loss:
                    best_loss = val_info["loss"]
                    self.save(self.save_path)
                    print(f'saved best model with loss: {val_info["loss"]}')


    def online_train_loop(self):
        pass

@hydra.main(config_name="train_transition", config_path="../../scripts/config/main", version_base="1.3")
def TransitionModel_offpolicy_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir = config.train.save_path)

    wandb.login(key=config.tools.wandb_key)
    wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    trainer = TransitionModel_Trainer(
        accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval,
        state_dim=config.TransitionModel.state_dim, action_dim=config.TransitionModel.action_dim, embed_dim=config.TransitionModel.embed_dim, num_attn_layers=config.TransitionModel.num_attn_layers, num_heads=config.TransitionModel.num_heads, activation=config.TransitionModel.activation,
        action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir
    )

    trainer.offpolicy_train_loop(data_path=config.data.data_path, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio)

if __name__ == "__main__":
    TransitionModel_offpolicy_train()