import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from digiq.data import ReplayBufferDataset
import random
from concurrent.futures import ThreadPoolExecutor
from digiq.misc import colorful_print
import time

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class DigiQTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    critic_lr: float = 1e-3,\
                    lm_lr: float = 1e-5,\
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3,
                    advantage_estimation: str = "mc",
                    learn_metric: str = "classification",
                    num_action_resampling: int = 4,
                    critc_use_original_action_to_backup: bool = True,
                    task_set = "",
                    actor_always_include_original_action = True,
                    actor_loss_type = "best-of-n",
                    pg_multiplier = 10.0,
                    awr_beta = 0.05,
                    detach_model = False,
    ):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr = critic_lr)
        
        self.learn_metric = learn_metric
        self.critc_use_original_action_to_backup = critc_use_original_action_to_backup
        self.task_set = task_set
        self.advantage_estimation = advantage_estimation
        self.detach_model = detach_model
        
        self.actor_always_include_original_action = actor_always_include_original_action
        self.actor_loss_type = actor_loss_type
        self.pg_multiplier = pg_multiplier
        self.awr_beta = awr_beta

        if self.learn_metric == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.learn_metric == "regression":
            self.criterion = torch.nn.MSELoss()
        
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.softmax = torch.nn.Softmax(dim = -1)
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.num_action_resampling = num_action_resampling

    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)
        self.critic_optimizer = self.accelerator.prepare(self.critic_optimizer)
    
    def get_action_and_qrep_batched(self, observation, image_features, image_path, pi_version):
        # for i in range(len(image_path)):
        #     image_path[i] = image_path[i]
        # batched operation, usually getting 4 observations at once; for critic pi_theta or pi_b are the same
        colorful_print("Getting pi actions", fg='blue')
        start = time.time()
        pi_action = self.agent.get_pi_action_guarantee_valid(observation, image_features, pi_version, max_try=3)
        end = time.time()
        colorful_print(f"getting pi actions for {len(image_path)} images took {end-start:.2f}s", fg='blue')
        batch_size = len(image_path)
        world_size = self.accelerator.num_processes # 8
        workers_each_process = len(self.agent.clients) // world_size # 64 // 8 = 8
        num_iters = batch_size // workers_each_process # 8 // 8 = 1
        
        def process_single(i):
            # e.g. process 1 always occupies clients 8-15; but the input i will be 0-8 (b/c bs=8)
            relative_occupation_id = i % workers_each_process
            absolute_occupation_id = self.accelerator.local_process_index * workers_each_process + relative_occupation_id
            out = self.agent.get_q_reps(image_path[i], pi_action[i], self.agent.clients[absolute_occupation_id])
            out_tensor = torch.Tensor(out).to(
                self.accelerator.unwrap_model(self.agent.model).device,
                dtype=self.accelerator.unwrap_model(self.agent.model).dtype
            ).flatten()
            return out_tensor

        start = time.time()
        q_rep_out = []
        for iter in range(num_iters):
            start_id = iter * workers_each_process
            with ThreadPoolExecutor() as executor:
                q_rep_out.extend( list(executor.map(process_single, range(start_id, start_id + workers_each_process))) )
        q_rep_out = torch.stack(q_rep_out)
        end = time.time()
        colorful_print(f"getting qrep for {len(image_path)} images took {end-start:.2f}s", fg='green')
        return pi_action, q_rep_out

    def critic_loss(self, observation, image_features, action, action_list, reward, next_observation, 
                    next_image_features, done, mc_return, q_rep_out, q_rep_out_list, validation=False, **kwargs):
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()

        # both mc and bellman should obtain Q and V from the dataset (not from the model)
        q1, q2, v1, v2 = self.agent.critic(observation, image_features, action, q_rep_out, detach_model=False)

        if self.advantage_estimation == "bellman":
            with torch.no_grad():
                if self.critc_use_original_action_to_backup:
                    pi_theta_action = action
                    pi_q_rep_out = q_rep_out.detach()
                else:
                    num_actions = q_rep_out_list.shape[1]
                    action_list = [action.split("<split>") for action in action_list]
                    randomly_sampled_index = random.choice(range(num_actions))
                    pi_theta_action = []
                    for action_list_for_this_batch in action_list:
                        # print(f"{randomly_sampled_index}: {len(action_list_for_this_batch)}")
                        pi_theta_action.append(action_list_for_this_batch[randomly_sampled_index])
                    pi_q_rep_out = q_rep_out_list[:, randomly_sampled_index, :]
                
                q1_target, q2_target, _, _ = self.agent.target_critic(observation, image_features, pi_theta_action, pi_q_rep_out, detach_model=False)
                # action is dummy in the line below
                _, _, v1_target, v2_target = self.agent.target_critic(next_observation, next_image_features, action, pi_q_rep_out, detach_model=False)
            q1, q2, v1, v2, q1_target, q2_target, v1_target, v2_target = q1.flatten(), q2.flatten(), v1.flatten(), v2.flatten(), q1_target.flatten(), q2_target.flatten(), v1_target.flatten(), v2_target.flatten()
            v1_target = reward + (1 - done)*v1_target*self.gamma
            v2_target = reward + (1 - done)*v2_target*self.gamma
            v1_mc_return_mse = self.criterion(v1, mc_return)
            v2_mc_return_mse = self.criterion(v2, mc_return)
            q1_mc_return_mse = self.criterion(q1, mc_return)
            q2_mc_return_mse = self.criterion(q2, mc_return)
        elif self.advantage_estimation == "mc":
            if self.learn_metric == "classification":
                base_target = (mc_return.detach() > 0).long()
                v1_target = base_target.clone()
                v2_target = base_target.clone()
                q1_target = base_target.clone()
                q2_target = base_target.clone()

            elif self.learn_metric == "regression":
                base_target = mc_return.detach()
                v1_target = base_target.clone()
                v2_target = base_target.clone()
                q1_target = base_target.clone()
                q2_target = base_target.clone()

        q1_loss = self.criterion(q1, v1_target)
        q2_loss = self.criterion(q2, v2_target)
        v1_loss = self.criterion(v1, q1_target)
        v2_loss = self.criterion(v2, q2_target)

        if self.learn_metric == "classification":
            # classification uses CrossEntropyLoss, so we need to apply softmax for aggregation
            q1 = self.softmax(q1)[:, 1]
            q2 = self.softmax(q2)[:, 1]
            v1 = self.softmax(v1)[:, 1]
            v2 = self.softmax(v2)[:, 1]
        
        v_max = torch.maximum(v1, v2).flatten()
        q_max = torch.maximum(q1, q2).flatten()

        if not validation:
            self.accelerator.backward(v1_loss+v2_loss+q1_loss+q2_loss)
        q1_loss, q2_loss = q1_loss.detach().cpu().item(), q2_loss.detach().cpu().item()
        v1_loss, v2_loss = v1_loss.detach().cpu().item(), v2_loss.detach().cpu().item()
        q1, q2, v1, v2 = q1.detach().cpu(), q2.detach().cpu(), v1.detach().cpu(), v2.detach().cpu()
        v_max, q_max = v_max.detach().cpu(), q_max.detach().cpu()

        # calculate the probability for logging purpose
        info = {
                "q1.loss": q1_loss,\
                "q2.loss": q2_loss,\
                "q1.mean": torch.mean(q1).item(),\
                "q1.min": torch.min(q1).item(),\
                "q1.max": torch.max(q1).item(),\
                "q1.std": torch.std(q1).item(),\
                "q2.mean": torch.mean(q2).item(),\
                "q2.min": torch.min(q2).item(),\
                "q2.max": torch.max(q2).item(),\
                "q2.std": torch.std(q2).item(),\
                "q_max.std": torch.std(q_max).item(),\
                "v1.loss": v1_loss,\
                "v2.loss": v2_loss,\
                "v1.mean": torch.mean(v1).item(),\
                "v1.min": torch.min(v1).item(),\
                "v1.max": torch.max(v1).item(),\
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
                "q1_mc_return_mse": torch.mean(q1_mc_return_mse),
                "q2_mc_return_mse": torch.mean(q2_mc_return_mse),
                })

        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def actor_loss(self, observation, action_list, image_features, mc_return, reward, q_rep_out, 
                   q_rep_out_list, validation=False, **kwargs):
        # print(observation[0])
        
        num_action_resampling = self.num_action_resampling

        mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()

        with torch.no_grad():
            advantage_action_pairs = []
            action_list = [action.split("<split>") for action in action_list]

            action_id_list = random.sample(range(q_rep_out_list.shape[1]), num_action_resampling)
            if self.actor_always_include_original_action:
                action_id_list[0] = 0
            for action_id in action_id_list:
                # action_list: [4, 64]
                # q_rep_out_list: [4, 64, 4096]
                pi_action = [action_list_for_this_batch[action_id] for action_list_for_this_batch in action_list]
                q_rep_out = q_rep_out_list[:, action_id, :]
                q1, q2, v1, v2 = self.agent.critic(observation, image_features, pi_action, q_rep_out, detach_model=False)
                if self.learn_metric == "classification":
                    # classification uses CrossEntropyLoss, so we need to apply softmax for aggregation
                    q1 = self.softmax(q1)[:, 1]
                    q2 = self.softmax(q2)[:, 1]
                    v1 = self.softmax(v1)[:, 1]
                    v2 = self.softmax(v2)[:, 1]

                q = torch.maximum(q1, q2).flatten()
                v = torch.maximum(v1, v2).flatten()
                advantage = q - v
                
                for batch_position in range(len(pi_action)):
                    if not self.agent.is_action_valid(pi_action[batch_position]):
                        colorful_print(f"Invalid action {pi_action[batch_position]} detected, setting advantage to zero", fg='red')
                        advantage[batch_position] = 0

                advantage_action_pairs.append((advantage, pi_action))

            batch_size = len(q)
            max_index = torch.stack([adv[0] for adv in advantage_action_pairs], dim=1).argmax(dim=1)

            advantage = torch.zeros(batch_size)
            pi_action = [""] * batch_size

            for i in range(batch_size):
                advantage[i] = advantage_action_pairs[max_index[i]][0][i]
                pi_action[i] = advantage_action_pairs[max_index[i]][1][i]

        advantage = torch.clamp(advantage, 0, 1)
        if self.task_set == "general":
            threshold = 0.10
        elif self.task_set == "webshop":
            threshold = 0.05
        else:
            raise ValueError(f"Unknown task set {self.task_set}")
        
        if self.actor_loss_type == "best-of-n":
            advantage = (advantage > threshold).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        elif self.actor_loss_type == "pg":
            advantage = (advantage*self.pg_multiplier).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        elif self.actor_loss_type == "awr":
            advantage = torch.exp(advantage/self.awr_beta).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        elif self.actor_loss_type == "sft":
            advantage = torch.ones_like(advantage).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        
        learned_actions = []
        for i in range(len(advantage)):
            if advantage[i] == 1:
                learned_actions.append(pi_action[i])

        image_features = image_features.to(self.agent.device)
        log_prob = self.agent.get_pi_theta_log_prob(observation, image_features, pi_action).sum(dim = 1).flatten()
        advantage = torch.Tensor(advantage).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        advantages = advantage.flatten()
        pg_loss = -torch.mean(log_prob.flatten()*advantages)
        value_loss = torch.zeros_like(pg_loss)
        if not validation:
            self.accelerator.backward(pg_loss+value_loss)
        advantages = advantages.detach().cpu()
        info =  {"pg.loss": pg_loss.detach().cpu().item(),
                "advantages.mean": advantages.mean(),
                "advantages.max": torch.max(advantages),
                "advantages.min": torch.min(advantages),
                "advantages.std": torch.std(advantages),}
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def update_critic(self, replay_buffer, validation_buffer=None):
        self.step += 1
        info = {}
        
        # Create the dataset and DataLoader once per update.
        dataset = ReplayBufferDataset(replay_buffer)
        sampler = RandomSampler(dataset, replacement=True, num_samples=self.grad_accum_steps * replay_buffer.batch_size)
        dataloader = DataLoader(dataset, batch_size=replay_buffer.batch_size, sampler=sampler)
        dataloader = self.accelerator.prepare(dataloader)
        
        for epoch in tqdm(range(self.epochs), disable= not self.accelerator.is_main_process):
            info_list = []
            for batch in dataloader:
                with self.accelerator.accumulate(self.agent.critic):
                    info_list.append(self.critic_loss(**batch))
                    self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
            info.update(dict_mean(info_list))
            
            # update target network each epoch
            if self.advantage_estimation == "bellman":
                self.agent.soft_update_target_critic(tau=self.tau)
        
            if validation_buffer is not None:
                info_list = []
                val_dataset = ReplayBufferDataset(validation_buffer)
                val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=self.grad_accum_steps * validation_buffer.batch_size)
                val_dataloader = DataLoader(val_dataset, batch_size=validation_buffer.batch_size, sampler=val_sampler)
                val_dataloader = self.accelerator.prepare(val_dataloader)
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, disable=True):
                        info_list.append(self.critic_loss(validation=True, **batch))
                info.update(dict_mean(info_list))
        return info
        
    def update_policy(self, replay_buffer, validation_buffer = None, no_update_actor=False):
        self.step += 1
        info = {}
        
        # Create the dataset and DataLoader once per update.
        dataset = ReplayBufferDataset(replay_buffer)
        sampler = RandomSampler(dataset, replacement=True, num_samples=self.grad_accum_steps * replay_buffer.batch_size)
        dataloader = DataLoader(dataset, batch_size=replay_buffer.batch_size, sampler=sampler)
        dataloader = self.accelerator.prepare(dataloader)
        
        if not no_update_actor:
            print(">>>Training phase of actor")
            
            info_list = []
            for epoch in tqdm(range(self.actor_epochs), disable= not self.accelerator.is_main_process):
                for batch in dataloader:
                    with self.accelerator.accumulate(self.agent.model):
                        info_list.append(self.actor_loss(**batch))
                        self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                        self.lm_optimizer.step()
                        self.lm_optimizer.zero_grad()
            info.update(dict_mean(info_list))
            
        if validation_buffer is not None:
            print(">>>Validation phase of actor")
            info_list = []
            val_dataset = ReplayBufferDataset(validation_buffer)
            val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=self.grad_accum_steps * validation_buffer.batch_size)
            val_dataloader = DataLoader(val_dataset, batch_size=validation_buffer.batch_size, sampler=val_sampler)
            val_dataloader = self.accelerator.prepare(val_dataloader)
            info_list = []
            with torch.no_grad():
                for batch in tqdm(val_dataloader, disable=True):
                    info_list.append(self.actor_loss(validation=True, **batch))
            info.update(dict_mean(info_list))
            
        return info

    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)

    def load(self, path):
        self.accelerator.load_state(path)
        
        