import numpy as np
from tqdm import tqdm
from digiq.algorithms.digiq import DigiQTrainer
from digiq.algorithms.filteredbc import BCTrainer
from digiq.misc import colorful_print
from digiq.data import ReplayBuffer
import wandb
import os
import torch
from time import sleep
import copy
from digiq.environment.env_utils import add_mc_return
from digiq.algorithms.parallel_utils import remote_eval_offline_ckpt

def framestack(orig_trajs):
    trajs = copy.deepcopy(orig_trajs)
    
    for i in range(len(trajs)):
        for j in range(len(trajs[i])):
            if j == 0:
                trajs[i][j]["image_features"] = np.concatenate([orig_trajs[i][j]["image_features"], orig_trajs[i][j]["image_features"]], axis=-1)
            else:
                trajs[i][j]["image_features"] = np.concatenate([orig_trajs[i][j-1]["image_features"], orig_trajs[i][j]["image_features"]], axis=-1)
            trajs[i][j]["next_image_features"] = np.concatenate([orig_trajs[i][j]["image_features"], orig_trajs[i][j]["next_image_features"]], axis=-1)
    return trajs

def offpolicy_train_loop(agent,\
                tokenizer,\
                accelerator,\
                batch_size: int = 2,
                capacity: int = 500000,
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                num_action_resampling: int = 4,
                use_wandb: bool = False,
                actor_epochs: int = 3,
                train_mode: str = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                train_algorithm: str = "digiq",
                offline_data_path: str = None,
                offline_actor_iterations: int = 20,
                offline_critic_iterations: int = 20,
                task_mode: str = 'single',
                advantage_estimation: str = 'mc',
                learn_metric: str = 'classification',
                worker_temp_path=None, 
                worker_run_path=None,
                worker_ips=[], 
                worker_username=None,
                critc_use_original_action_to_backup=True,
                task_set="",
                actor_always_include_original_action=True,
                actor_loss_type="best-of-n",
                pg_multiplier=10.0,
                awr_beta=0.05,
                detach_model=False,
                **kwargs):

    torch.autograd.set_detect_anomaly(True)
    if train_algorithm == "digiq":
        trainer = DigiQTrainer(agent=agent,\
                                accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm,
                                advantage_estimation = advantage_estimation,
                                learn_metric = learn_metric,
                                num_action_resampling=num_action_resampling,
                                critc_use_original_action_to_backup=critc_use_original_action_to_backup,
                                task_set=task_set,
                                actor_always_include_original_action=actor_always_include_original_action,
                                actor_loss_type=actor_loss_type,
                                pg_multiplier=pg_multiplier,
                                awr_beta=awr_beta,
                                detach_model=detach_model)
    elif train_algorithm == "filteredbc":
        trainer = BCTrainer(agent=agent,\
                                tokenizer=tokenizer,\
                                accelerator=accelerator,
                                lm_lr = lm_lr,\
                                epochs = actor_epochs,\
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    all_trajectories = []
    
    # prepare the model
    agent.prepare()
    # prepare the optimizers
    trainer.prepare()

    loaded_trajs = False
    
    # omit this for online training
    if offline_data_path is not None:
        all_trajectories = torch.load(offline_data_path, weights_only=False)
        all_trajectories = framestack(all_trajectories)
        print(f"The number of offline trajectories is {len(all_trajectories)}")
        print(f"The average number of steps in each trajectory is {np.mean([len(t) for t in all_trajectories])}")
        all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]
        train_trajectories = all_trajectories[:int(len(all_trajectories)*0.8)]
        val_trajectories = all_trajectories[int(len(all_trajectories)*0.8):]
        loaded_trajs = 'scratch'
        
    # resume training from the saved checkpoint
    if os.path.exists(os.path.join(save_path, 'trainer.pt')):
        trainer.load(os.path.join(save_path, 'trainer.pt'))
        if use_wandb and accelerator.is_main_process:
            print("Loading from checkpoint")
        loaded_trajs = 'resume'
            
    if not loaded_trajs:
        train_trajectories = []
        val_trajectories = []
        all_trajectories = []

    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)

    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])
    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)

    # offline training
    info = {}
    if os.path.exists(os.path.join(save_path, 'trainer_offline.pt')):
        trainer.load(os.path.join(save_path, 'trainer_offline.pt'))
        print("Loading from offline trainer (critic+actor)")
    else:
        if offline_data_path is not None and train_mode != "online":
            colorful_print(">>>Offline Training", fg='green')
            
            if os.path.exists(os.path.join(save_path, 'digiq_critic.pt')):
                print("Loading from offline critic (no training)")
                trainer.load(os.path.join(save_path, 'digiq_critic.pt'))
            else:
                print(">>>Training critic")
                for i in tqdm(range(offline_critic_iterations), disable=not accelerator.is_main_process):
                    info = trainer.update_critic(replay_buffer=replay_buffer, validation_buffer=validation_buffer)
                    if use_wandb and accelerator.is_main_process:
                        wandb.log(info)
                accelerator.wait_for_everyone()
                trainer.save(os.path.join(save_path, 'digiq_critic.pt'))
                sleep(30)

            print(">>>Training Policy")
            for i in tqdm(range(offline_actor_iterations), disable=not accelerator.is_main_process):
                info.update(trainer.update_policy(replay_buffer=replay_buffer, validation_buffer=validation_buffer, no_update_actor=False))
                if use_wandb and accelerator.is_main_process:
                    wandb.log(info)
            accelerator.wait_for_everyone()
            trainer.save(os.path.join(save_path, 'trainer_offline.pt'))
                
    # submit evaluation to remote machine
    if task_mode == "train_and_remote_eval":
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            remote_eval_offline_ckpt(save_path=save_path, 
                        worker_temp_path=worker_temp_path, 
                        worker_run_path=worker_run_path,
                        worker_ips=worker_ips, 
                        worker_username=worker_username)
                