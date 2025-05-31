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

import re
import hydra
import datetime
import wandb

from digiq.models.agent import Agent, process_action_str2tensor, process_action_tensor2str
from digiq.models.value_model import Value_Model
from digiq.models.transition_model import Transition_Model
from digiq.data.utils import ReplayBuffer
from digiq.data.utils import ReplayBufferDataset

def get_initpolicy_trainer(trainer_name:str, config, accelerator):
    if trainer_name=="BC":
        return BehaviorCloning_Trainer(
            accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval,
            state_dim=config.TransitionModel.stata_dim, goal_dim=config.TransitionModel.goal_dim, embed_dim=config.Transition_Model.embed_dom, num_sce_type=config.Agent.num_sce_type, latent_action_dim=config.Agent.latent_action_dim, num_attn_layers_first=config.Agent.num_attn_layers_first, num_heads_first=config.Agent.num_heads_first, num_attn_layers_second=config.Agent.num_attn_layers_second, num_heads_second=config.Agent.num_heads_second,
            goal_encoder_backbone=config.Goal_encoder.goal_encoder_backbone, goal_encoder_cache_dir=config.Goal_encoder.goal_encoder_cache_dir, action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir,
            loss_coef_alpha=config.train_init_policy.loss_coef_alpha, loss_coef_beta=config.train_init_policy.loss_coef_beta
        )
    elif trainer_name=="MCP":
        return MCP_Trainer(
            accelerator=accelerator, load_path=config.train.load_path, save_path=config.train.save_path, epoch=config.train.epoch, val_interval=config.train.val_interval,
            state_dim=config.TransitionModel.stata_dim, goal_dim=config.TransitionModel.goal_dim, embed_dim=config.Transition_Model.embed_dom, num_sce_type=config.Agent.num_sce_type, latent_action_dim=config.Agent.latent_action_dim, num_attn_layers_first=config.Agent.num_attn_layers_first, num_heads_first=config.Agent.num_heads_first, num_attn_layers_second=config.Agent.num_attn_layers_second, num_heads_second=config.Agent.num_heads_second,
            goal_encoder_backbone=config.Goal_encoder.goal_encoder_backbone, goal_encoder_cache_dir=config.Goal_encoder.goal_encoder_cache_dir, action_encoder_backbone=config.Action_encoder.action_encoder_backbone, action_encoder_cache_dir=config.Action_encoder.action_encoder_cache_dir,
            loss_coef_alpha=config.train_init_policy.loss_coef_alpha, loss_coef_beta=config.train_init_policy.loss_coef_beta,
            trial_times=config.train_init_policy.trial_times, num_attn_layers_transition=config.TransitionModel.num_attn_layers, num_heads_transition=config.TransitionModel.num_heads, activation=config.TransitionModel.activation, num_attn_layers_value=config.TransitionModel.num_attn_layers, num_heads_value=config.TransitionModel.num_heads, x_range_min=config.train_init_policy.x_range_min, x_range_max=config.train_init_policy.x_range_max, y_range_min=config.train_init_policy.y_range_min, y_range_max=config.train_init_policy.y_range_max
        )
    else:
        raise NotImplementedError()

class InitPolicy_Trainer:
    def __init__(
        self, accelerator:Accelerator=None, load_path:str=None, save_path:str=None, epoch:int=None, val_interval:int=None, learn_metric:str=None, advantage_estimation:str=None,
        state_dim:int=None, goal_dim:int=None, action_dim:int=None, embed_dim:int=None, num_sce_type:int=None, latent_action_dim:int=None,num_attn_layers_first:int=None, num_heads_first:int=None, num_attn_layers_second:int=None, num_heads_second:int=None,
        goal_encoder_backbone:str=None, goal_encoder_cache_dir:str=None, action_encoder_backbone:str=None, action_encoder_cache_dir:str=None,
        loss_coef_alpha:float=None, loss_coef_beta=None,
    ):
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.learn_metric = learn_metric
        self.advantage_estimation = advantage_estimation
        self.agent = Agent()
        
        self.alpha = loss_coef_alpha
        self.beta = loss_coef_beta

        self.agent = Agent(
            state_dim=state_dim, goal_dim=goal_dim, action_dim=action_dim, embed_dim=embed_dim, num_sce_type=num_sce_type, latent_action_dim=latent_action_dim, num_attn_layers_first=num_attn_layers_first, num_heads_first=num_heads_first, num_attn_layers_second=num_attn_layers_second, num_heads_second=num_heads_second,
            goal_encoder_backbone=goal_encoder_backbone, goal_encoder_cache_dir=goal_encoder_cache_dir, action_encoder_backbone=action_encoder_backbone, action_encoder_cache_dir=action_encoder_cache_dir, device=self.device
        )
        self.optimizer = optim.Adam(self.agent.parameters())
        self.agent, self.optimizer = self.accelerator.prepare(self.agent, self.optimizer)
        
        if self.learn_metric == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.learn_metric == "regression":
            self.criterion = torch.nn.MSELoss()

        self.load_path = load_path
        self.save_path = save_path
        self.epoch = epoch
        self.val_interval = val_interval
        self.load(load_path, self.device)

    def load(self, path:str, device:str):
        if path:
            self.agent.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            self.agent.init_weight()
            self.agent.to(device)

    def save(self, path:str):
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.agent.state_dict(), f"{path}/digiq_ValueModel_{time}.pth")

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
    
    def _loss(self, action_pre:torch.Tensor, action_type_logits:torch.Tensor, typing_type_logits:torch.Tensor, bottom_button_type_logits:torch.Tensor, action:torch.Tensor) -> dict[str:torch.Tensor]:
        loss = F.cross_entropy(action_type_logits, action[0])
        loss_ = 0.0

        if action[0] == 0: # typing
            loss_ = F.cross_entropy(typing_type_logits, action[1])
            loss = loss + self.alpha*loss_
        elif action[0] == 1: # bottom_button
            loss_ = F.cross_entropy(bottom_button_type_logits, action[2])
            loss = loss + self.alpha*loss_
        elif action[0] == 2: # touch
            loss_ = F.mse_loss(action_pre[3:5], action[3:5])
            loss = loss + self.beta*loss_
        elif action[0] == 3: # scroll
            loss_ = F.mse_loss(action_pre[5:], action[5:])
            loss = loss + self.beta*loss_
        
        return {"total loss": loss, "action loss": loss_}
    
    def loss(self, batch):
        raise NotImplementedError()

    def train_loop(self, data_path, batch_size=512, capacity=500000, train_ratio=0.8, val_ratio=0.2):
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

        best_loss = float("inf")
        for epoch in range(self.epoch):
            for batch in train_dataloader:
                train_info = self.loss(batch)
                wandb.log(train_info)
                self.optimizer.zero_grad()
                train_info["total loss"].backward()
                self.optimizer.step()

            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_info = self.loss(batch, validation=True)
                        wandb.log(val_info)

                    print(f"loss: {val_info["total loss"]}")
                    if self.accelerator.is_main_process and val_info["total loss"] < best_loss:
                        best_loss = val_info["total loss"]
                        self.save(self.save_path)
                        print(f'saved best model with loss: {best_loss}')

class BehaviorCloning_Trainer(InitPolicy_Trainer):
    def __init__(self, accelerator = None, load_path = None, save_path = None, epoch = None, val_interval = None, learn_metric = None, advantage_estimation = None, state_dim = None, goal_dim = None, action_dim = None, embed_dim = None, num_sce_type = None, latent_action_dim = None, num_attn_layers_first = None, num_heads_first = None, num_attn_layers_second = None, num_heads_second = None, goal_encoder_backbone = None, goal_encoder_cache_dir = None, action_encoder_backbone = None, action_encoder_cache_dir = None, loss_coef_alpha = None, loss_coef_beta=None):
        super().__init__(accelerator, load_path, save_path, epoch, val_interval, learn_metric, advantage_estimation, state_dim, goal_dim, action_dim, embed_dim, num_sce_type, latent_action_dim, num_attn_layers_first, num_heads_first, num_attn_layers_second, num_heads_second, goal_encoder_backbone, goal_encoder_cache_dir, action_encoder_backbone, action_encoder_cache_dir, loss_coef_alpha, loss_coef_beta)

    def loss(self, batch):
        observation, action, reward, next_observation, done, mc_return, state, next_state = batch
        reward = torch.Tensor(reward).to(self.device).flatten()
        done = torch.Tensor(done).to(self.device).flatten()
        mc_return = torch.Tensor(mc_return).to(self.device).flatten()
        past_action, goal = self.parse_obs(observation)
        next_past_action, next_goal = self.parse_obs(next_observation)
        
        action = process_action_str2tensor(action)
        action_pre, action_type_logits, typing_type_logits, bottom_button_type_logits = Agent.forward(state=state, goal=goal, past_action=past_action, determine=False)

        return self._loss(action_pre, action_type_logits, typing_type_logits, bottom_button_type_logits, action)

class MCP_Trainer(InitPolicy_Trainer):
    def __init__(
        self, accelerator = None, load_path = None, save_path = None, epoch = None, val_interval = None, learn_metric = None, advantage_estimation = None, state_dim = None, goal_dim = None, action_dim = None, embed_dim = None, num_sce_type = None, latent_action_dim = None, num_attn_layers_first = None, num_heads_first = None, num_attn_layers_second = None, num_heads_second = None, goal_encoder_backbone = None, goal_encoder_cache_dir = None, action_encoder_backbone = None, action_encoder_cache_dir = None, loss_coef_alpha = None, loss_coef_beta=None,
        trial_times:int = None, num_attn_layers_transition:int=3, num_heads_transition:int=5, activation:str="ReLU",num_attn_layers_value:int=None, num_heads_value:int=None, x_range_min:float=None, x_range_max:float=None, y_range_min:float=None, y_range_max:float=None
    ):
        super().__init__(accelerator, load_path, save_path, epoch, val_interval, learn_metric, advantage_estimation, state_dim, goal_dim, action_dim, embed_dim, num_sce_type, latent_action_dim, num_attn_layers_first, num_heads_first, num_attn_layers_second, num_heads_second, goal_encoder_backbone, goal_encoder_cache_dir, action_encoder_backbone, action_encoder_cache_dir, loss_coef_alpha, loss_coef_beta)
        self.transition = Transition_Model(state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers_transition, num_heads=num_heads_transition, activation=activation, device=self.device)
        self.value = Value_Model(state_dim=state_dim, goal_dim=goal_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers_value, num_heads=num_heads_value,goal_encoder_backbone=goal_encoder_backbone, goal_encoder_cache_dir=goal_encoder_cache_dir, action_encoder_backbone=action_encoder_backbone, action_encoder_cache_dir=action_encoder_cache_dir)
        self.trial_times = trial_times
        self.num_sce_type = num_sce_type
        self.x_range_min = x_range_min
        self.x_range_max = x_range_max
        self.y_range_min = y_range_min
        self.y_range_max = y_range_max

    def loss(self, batch):
        observation, action, reward, next_observation, done, mc_return, state, next_state = batch
        reward = torch.Tensor(reward).to(self.device).flatten()
        done = torch.Tensor(done).to(self.device).flatten()
        mc_return = torch.Tensor(mc_return).to(self.device).flatten()
        past_action, goal = self.parse_obs(observation)
        next_past_action, next_goal = self.parse_obs(next_observation)

        best_value = self.value.forward(state=next_state, goal=goal, past_action=action)
        best_action = process_action_str2tensor(best_action)
        for _ in range(self.trial_times):
            new_action = torch.tensor([
                torch.int(torch.clamp(best_action[0]+torch.normal(0, 1), 0, 3)),
                torch.int(torch.clamp(best_action[1]+torch.normal(0, 1), 0, self.num_sce_type-1)),
                torch.int(torch.clamp(best_action[2]+torch.normal(0, 1), 0, 2)),
                torch.clamp(best_action[3]+torch.normal(0, 1), self.x_range_min, self.x_range_max),
                torch.clamp(best_action[4]+torch.normal(0, 1), self.y_range_min, self.y_range_max),
                torch.clamp(best_action[5]+torch.normal(0, 1), self.x_range_min, self.x_range_max),
                torch.clamp(best_action[6]+torch.normal(0, 1), self.y_range_min, self.y_range_max),
                torch.clamp(best_action[7]+torch.normal(0, 1), self.x_range_min, self.x_range_max),
                torch.clamp(best_action[8]+torch.normal(0, 1), self.y_range_min, self.y_range_max),
            ])
            new_state, _, _ = self.transition.forward(state=state, action=new_action)
            new_value = self.value.forward(state=new_state, goal=goal, past_action=process_action_tensor2str(new_action))
            if new_value>best_value:
                best_action = new_action
                best_value = new_value
        
        action_pre, action_type_logits, typing_type_logits, bottom_button_type_logits = Agent.forward(state=state, goal=goal, past_action=past_action, determine=False)

        return self._loss(action_pre=action_pre, action_type_logits=action_type_logits, typing_type_logits=typing_type_logits, bottom_button_type_logits=bottom_button_type_logits, action=best_action)

@hydra.main(config_name="init_policy", config_path="../../scripts/config/main", version_base="1.3")
def InitPolicy_train(config):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], project_dir = config.train.save_path)

    wandb.login(key=config.tools.wandb_key)
    wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    trainer = get_initpolicy_trainer(config.train_init_policy.trainer_name, config, accelerator)
    trainer.train_loop(data_path=config.data.data_path, batch_size=config.data.batch_size, capacity=config.data.capacity, train_ratio=config.data.train_ratio, val_ratio=config.data.val_ratio)