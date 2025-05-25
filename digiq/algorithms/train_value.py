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

import random
import re
import hydra
import itertools
import datetime
import threading
from tqdm import tqdm
import wandb

from digiq.models.value_model import Value_Model
from digiq.data.utils import ReplayBuffer
from digiq.data.utils import ReplayBufferDataset

class ValueModel_Trainer:
    def __init__(self, accelerator:Accelerator=None, load_path:str=None, save_path:str=None, epoch:int=None, val_interval:int=None, learn_metric:str=None, advantage_estimation:str=None,
                 state_dim:int=None, goal_dim:int=None, action_dim:int=None, embed_dim:int=None, num_attn_layers:int=None, num_heads:int=None,
                 goal_encoder_backbone:str=None, goal_encoder_cache_dir:str=None, action_encoder_backbone:str=None, action_encoder_cache_dir:str=None):
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.learn_metric = learn_metric
        self.advantage_estimation = advantage_estimation
        
        self.value_model = Value_Model(state_dim=state_dim, goal_dim=goal_dim, action_dim=action_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers, num_heads=num_heads,
                                      action_encoder_backbone=action_encoder_backbone, action_encoder_cache_dir=action_encoder_cache_dir,
                                      goal_encoder_backbone=goal_encoder_backbone, goal_encoder_cache_dir=goal_encoder_cache_dir,
                                      device=self.device)
        self.optimizer = optim.Adam(self.value_model.parameters())
        self.value_model, self.optimizer = self.accelerator.prepare(self.value_model, self.optimizer)
        
        if self.learn_metric == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.learn_metric == "regression":
            self.criterion = torch.nn.MSELoss()

        self.load_path = load_path
        self.save_path = save_path
        self.epoch = epoch
        self.val_interval = val_interval
        self.load(load_path, self.device)

    def save(self, path:str):
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.value_model.state_dict(), f"{path}/digiq_ValueModel_{time}.pth")

    def load(self, path:str, device:str):
        if path:
            self.value_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            self.value_model.init_weight()
            self.value_model.to(device)
    
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
    
    def loss(self, batch, validation=False):
        print("batch", batch)
        observation, _, reward, next_observation, done, mc_return, state, next_state = batch

        reward = torch.Tensor(reward).to(self.device).flatten()
        done = torch.Tensor(done).to(self.device).flatten()
        mc_return = torch.Tensor(mc_return).to(self.device).flatten()
        past_action, goal = self.parse_obs(observation)
        next_past_action, next_goal = self.parse_obs(next_observation)

        # both mc and bellman should obtain Q and V from the dataset (not from the model)
        v1, v2 = self.value_model.forward(state, goal, past_action)

        if self.advantage_estimation == "bellman":
            with torch.no_grad():
                v1_target, v2_target = self.value_model.forward(state=next_state, goal=next_goal, past_action=next_past_action)
            v1, v2, v1_target, v2_target = v1.flatten(), v2.flatten(), v1_target.flatten(), v2_target.flatten()
            v1_target = reward + (1 - done)*v1_target*self.gamma
            v2_target = reward + (1 - done)*v2_target*self.gamma
            v1_mc_return_mse = self.criterion(v1, mc_return)
            v2_mc_return_mse = self.criterion(v2, mc_return)
        elif self.advantage_estimation == "mc":
            if self.learn_metric == "classification":
                base_target = (mc_return.detach() > 0).long()
                v1_target = base_target.clone()
                v2_target = base_target.clone()

            elif self.learn_metric == "regression":
                base_target = mc_return.detach()
                v1_target = base_target.clone()
                v2_target = base_target.clone()

        v1_loss = self.criterion(v1, v1_target)
        v2_loss = self.criterion(v2, v2_target)

        if self.learn_metric == "classification":
            # classification uses CrossEntropyLoss, so we need to apply softmax for aggregation
            v1 = self.softmax(v1)[:, 1]
            v2 = self.softmax(v2)[:, 1]
        
        v_max = torch.maximum(v1, v2).flatten()

        if not validation:
            self.accelerator.backward(v1_loss+v2_loss)
        v1_loss, v2_loss = v1_loss.detach().cpu().item(), v2_loss.detach().cpu().item()
        v1, v2 = v1.detach().cpu(), v2.detach().cpu()
        v_max = v_max.detach().cpu()

        # calculate the probability for logging purpose
        info = {
            "v1.loss": v1_loss,
            "v2.loss": v2_loss,
            "v1.mean": torch.mean(v1).item(),
            "v1.min": torch.min(v1).item(),
            "v1.max": torch.max(v1).item(),
            "v1.std": torch.std(v1).item(),
            "v2.mean": torch.mean(v2).item(),
            "v2.max": torch.max(v2).item(),
            "v2.min": torch.min(v2).item(),
            "v2.std": torch.std(v2).item(),
            "v_max.std": torch.std(v_max).item(),
        }

        if self.advantage_estimation == "bellman":
            info.update({
                "v1_mc_return_mse": torch.mean(v1_mc_return_mse),
                "v2_mc_return_mse": torch.mean(v2_mc_return_mse),
            })

        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info
    
    def train_loop(self, data_path, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2):
        # step1: load and construct dataset
        assert(data_path is not None), "data path is required"

        all_data = torch.load(data_path, weights_only=False)
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
        best_loss = float("inf")
        for epoch in range(self.epoch):
            for batch in train_dataloader:
                train_info = self.loss(batch)
                wandb.log(train_info)

            if epoch % self.val_interval == 0:
                for batch in val_dataloader:
                    val_info = self.loss(batch, validation=True)
                    wandb.log(val_info)

                print(f'epoch {epoch} train loss: {train_info["v1.loss"] + train_info["v2.loss"]} val loss: {val_info["validation.v1.loss"] + val_info["validation.v2.loss"]}')
                if self.accelerator.is_main_process and val_info["validation.v1.loss"] + val_info["validation.v2.loss"] < best_loss:
                    best_loss = val_info["validation.v1.loss"] + val_info["validation.v2.loss"]
                    self.save(self.save_path)
                    print(f'saved best model with loss: {best_loss}')

@hydra.main(config_name="train_value", config_path="../../scripts/config/main", version_base="1.3")
def ValueModel_offpolicy_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir = config.train.save_path)

    wandb.login(key=config.tools.wandb_key)
    wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    trainer = ValueModel_Trainer(
        accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval,
        state_dim=config.TransitionModel.state_dim, goal_dim=config.TransitionModel.action_dim, embed_dim=config.TransitionModel.embed_dim, num_attn_layers=config.TransitionModel.num_attn_layers, num_heads=config.TransitionModel.num_heads,
        action_encoder_backbone=config.Action_encoder.action_encoder_backbone,
        action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir,
        goal_encoder_backbone=config.Goal_encoder.goal_encoder_backbone,
        goal_encoder_cache_dir=config.Goal_encoder.goal_encoder_cache_dir,
        action_dim=config.TransitionModel.action_dim
    )

    trainer.train_loop(data_path=config.data.data_path, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio)

if __name__ == "__main__":
    ValueModel_offpolicy_train()