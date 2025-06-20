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

from digiq.models.transition_model import MLPTransition, TransformerTransition
from digiq.models.encoder import ActionEncoder, GoalEncoder
from digiq.data.utils import ReplayBuffer
from digiq.data.utils import ReplayBufferDataset

class TransitionModelTrainer:
    def __init__(self, accelerator:Accelerator=None, load_path:str=None, save_path:str=None, epoch:int=None, val_interval:int=None,
                 state_dim:int=None, action_dim:int=None, goal_dim:int=None, embed_dim:int=None, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", 
                 action_encoder_backbone:str=None, action_encoder_cache_dir:str=None, goal_encoder_backbone:str=None, goal_encoder_cache_dir:str=None,
                 model_id:int=None, seed:int=None):
        self.model_id = model_id
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
        self.trainsition_model = TransformerTransition(state_dim=state_dim, action_dim=action_dim, device=self.device)
        
        self.optimizer = optim.Adam(self.trainsition_model.parameters(), lr=1e-4)
        self.trainsition_model, self.optimizer = self.accelerator.prepare(self.trainsition_model, self.optimizer)

        self.load_path = load_path
        self.save_path = save_path
        self.epoch = epoch
        self.val_interval = val_interval
        self.load(load_path, self.device)

    def save(self, path):
        if self.model_id is not None:
            name = f"digiq_TransitionModel_M{self.model_id}.pth"
            torch.save(self.trainsition_model.state_dict(), os.path.join(path, name))
        else:
            time=datetime.datetime.now().strftime("%m%d%H%M")
            torch.save(self.trainsition_model.state_dict(), f"{path}/digiq_TransitionModel_{time}.pth")
            print(f'saved best model in {path}/digiq_TransitionModel_{time}.pth')

    def load(self, path: str, device: str):
        if path:
            self.trainsition_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            self.trainsition_model.init_weight()
            self.trainsition_model.to(device)

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

    def loss(self, batch):
        observation, action, reward, next_observation, done, mc_return, state, next_state = batch["observation"], batch["action"], batch["reward"], batch["next_observation"], batch["done"], batch["mc_return"], batch["s_rep"], batch["next_s_rep"]
        with torch.no_grad():
            action = self.action_encoder(action)
        next_states_pre = self.trainsition_model.forward(state, action)
        loss = F.mse_loss(next_states_pre, next_state)
        return {"loss": loss}

    def offpolicy_train_loop(self, data_path_general, data_path_web_shop, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2, bagging=False):
        # step1: load and construct dataset
        assert data_path_general is not None, "data path is required"
        assert data_path_web_shop is not None, "data path is required"

        all_data_general = torch.load(data_path_general, weights_only=False)
        all_data_web_shop = torch.load(data_path_web_shop, weights_only=False)
        all_data = all_data_general + all_data_web_shop
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
                with self.accelerator.accumulate(self.trainsition_model):
                    train_info = self.loss(batch)
                    self.optimizer.zero_grad()
                    self.accelerator.backward(train_info["loss"])
                    self.optimizer.step()
                    wandb.log(train_info)

            if epoch % self.val_interval == 0:
                for batch in val_dataloader:
                    val_info = self.loss(batch)
                    wandb.log(val_info)

                print(f'epoch {epoch} train loss: {train_info["loss"]} val loss: {val_info["loss"]}')
                if self.accelerator.is_main_process and val_info["loss"] < best_loss:
                    best_loss = val_info["loss"]
                    self.save(self.save_path)

    def breman_train_loop(self, data_path_general, data_path_web_shop, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2, bagging=False):
        # step1: load and construct dataset
        assert data_path_general is not None, "data path is required"
        assert data_path_web_shop is not None, "data path is required"
        
        all_data_general = torch.load(data_path_general, weights_only=False)
        all_data_web_shop = torch.load(data_path_web_shop, weights_only=False)
        all_data = all_data_general
        if bagging:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(all_data), size=len(all_data), replace=True)
            all_data = [all_data[i] for i in idx]

        split = int(len(all_data) * train_ratio)
        train_data, val_data = all_data[:split], all_data[split:]

        train_buffer = ReplayBuffer(batch_size, capacity=capacity)
        val_buffer = ReplayBuffer(batch_size, capacity=capacity)

        for d in train_data:
            train_buffer.insert(**d)
        for d in val_data:
            val_buffer.insert(**d)

        train_dataset = ReplayBufferDataset(train_buffer)
        val_dataset = ReplayBufferDataset(val_buffer)

        g = torch.Generator()
        g.manual_seed(self.seed)
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=batch_size, generator=g)
        val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=batch_size, generator=g)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)

        # step2: train and val
        best_loss = float("inf")
        for epoch in range(self.epoch):
            for batch in train_dataloader:
                with self.accelerator.accumulate(self.trainsition_model):
                    train_info = self.loss(batch)
                    self.optimizer.zero_grad()
                    self.accelerator.backward(train_info["loss"])
                    self.optimizer.step()
                    wandb.log({f"train/loss_model_{self.model_id}": train_info["loss"]})

            if epoch % self.val_interval == 0:
                for batch in val_dataloader:
                    val_info = self.loss(batch)
                    wandb.log({f"val/loss_model_{self.model_id}": val_info["loss"]})

                if self.accelerator.is_main_process:
                    print(f"[M{self.model_id}] epoch {epoch} "f"train: {train_info['loss']:.4f}  val: {val_info['loss']:.4f}")
                    if val_info["loss"] < best_loss:
                        best_loss = val_info["loss"]
                        self.save(self.save_path)
                        print(f"[M{self.model_id}] saved best model (loss={best_loss:.4f})")

def TransitionModel_offpolicy_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60 * 60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir=config.train.save_path)

    wandb.login(key=config.tools.wandb_key)
    wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    trainer = TransitionModelTrainer(
        accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval,
        state_dim=config.TransitionModel.state_dim, goal_dim=config.TransitionModel.goal_dim, action_dim=config.TransitionModel.action_dim, embed_dim=config.TransitionModel.embed_dim, num_attn_layers=config.TransitionModel.num_attn_layers, num_heads=config.TransitionModel.num_heads, activation=config.TransitionModel.activation,
        action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir, goal_encoder_backbone=config.Goal_encoder.goal_encoder_backbone, goal_encoder_cache_dir=config.Goal_encoder.goal_encoder_cache_dir,
        seed=config.train.seed
    )

    trainer.offpolicy_train_loop(data_path_general=config.data.data_path_general, data_path_web_shop=config.data.data_path_web_shop, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio)

def TransitionModel_breman_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60 * 60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir=config.train.save_path)

    wandb.login(key=config.tools.wandb_key)

    K = config.train.K
    base_seed = config.train.seed
    bagging = getattr(config.train, "bagging", False)

    for k in range(K):
        run_name = f"{config.run_name or 'Transition'}_M{k}"

        wandb.init(project=config.project_name, name=run_name, config=dict(config))
        
        trainer = TransitionModelTrainer(
            accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval,
            state_dim=config.TransitionModel.state_dim, action_dim=config.TransitionModel.action_dim, goal_dim=config.TransitionModel.goal_dim, embed_dim=config.TransitionModel.embed_dim, num_attn_layers=config.TransitionModel.num_attn_layers, num_heads=config.TransitionModel.num_heads, activation=config.TransitionModel.activation,
            action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir, goal_encoder_backbone=config.Goal_encoder.goal_encoder_backbone, goal_encoder_cache_dir=config.Goal_encoder.goal_encoder_cache_dir,
            model_id=k, seed=base_seed + k
        )

        trainer.breman_train_loop(data_path_general=config.data.data_path_general, data_path_web_shop=config.data.data_path_web_shop, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio, bagging=bagging)


        wandb.finish()

        #To free GPU memory
        del trainer
        torch.cuda.empty_cache()
        accelerator.free_memory()

@hydra.main(config_name="train_transition", config_path="../../scripts/config/main", version_base="1.3")
def TransitionModel_train(config):
    TransitionModel_offpolicy_train(config)
    # TransitionModel_breman_train(config)

if __name__ == "__main__":
    TransitionModel_train()