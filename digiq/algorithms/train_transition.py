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

from tqdm import tqdm
import wandb

from models.transition_model import Transition_Model
from models.encoder import ActionEncoder
from data.utils import ReplayBuffer
from data.utils import ReplayBufferDataset

class TransitionModel_Trainer:
    def __init__(self, accelerator:Accelerator=None, load_path:str=None, save_path:str=None,
                 state_dim:int=None, action_dim:int=None, embed_dim:int=None, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", 
                 action_encoder_backbone:str=None, action_encoder_cache_dir:str=None):
        self.accelerator = accelerator
        self.device = self.accelerator.device

        # self.state_encoder = None
        self.action_encoder = ActionEncoder(backbone=action_encoder_backbone, cache_dir=action_encoder_cache_dir, device=self.device)

        self.trainsition_model = Transition_Model(state_dim, action_dim, embed_dim, num_attn_layers, num_heads, activation, self.device)
        self.optimizer = optim.Adam(self.trainsition_model.parameters)
        self.trainsition_model, self.optimizer = self.accelerator.prepare(self.trainsition_model, self.optimizer)

        self.load_path = load_path
        self.save_path = save_path

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
        observation, image_features, actions_ori, action_list, reward, next_observation, next_image_features, done, mc_return, q_rep_out, q_rep_out_list, states, next_states = batch
        with torch.no_grad:
            actions = self.action_encoder(actions_ori)

        next_states_pre = self.trainsition_model.forward(states, actions)
        loss = F.mse_loss(next_states_pre, next_states)

        return {"loss": loss}

    def offpolicy_train_loop(self, data_path, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2):
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
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)

        # step2: train and val
        for batch in tqdm(train_dataloader):
            info = self.loss(batch)
            wandb.log(info)
            
            with self.accelerator.accumulate(self.trainsition_model):
                self.optimizer.zero_grad()
                self.accelerator.backward(info["loss"])
                self.optimizer.step()

        for batch in tqdm(val_dataloader):
            info = self.loss(batch)
            wandb.log(info)

        if self.accelerator.is_main_process:
            self.save(self.save_path)

    def online_train_loop(self):
        pass

@hydra.main(config_name="train_transition", config_path="../../scripts/config/main", version_base="1.3")
def TransitionModel_offpolicy_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir = config.save_path)

    wandb.login(key=config.tools.wandb_key)
    wandb.init(project=config.project_name, entity=config.entity_name, name=config.run_name, config=dict(config))

    trainer = TransitionModel_Trainer(
        accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path,
        state_dim=config.TransitionModel.state_dim, action_dim=config.TransitionModel.action_dim, embed_dim=config.TransitionModel.embed_dim, num_attn_layers=config.TransitionModel.num_attn_layers, num_heads=config.TransitionModel.num_heads, activation=config.TransitionModel.activation,
        action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir
    )

    trainer.offpolicy_train_loop(data_path=config.data.data_path, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio)

if __name__ == "__main__":
    TransitionModel_offpolicy_train()