from digiq.environment import batch_interact_environment
from digiq.algorithms.digiq import DigiQTrainer
from digiq.algorithms.filteredbc import BCTrainer
import numpy as np
from digiq.misc import colorful_print
import copy
import wandb
import os
import torch
import time

def eval_loop(env,\
                agent,\
                accelerator,\
                tokenizer,\
                critic_lr,\
                lm_lr,\
                tau,\
                epochs,\
                actor_epochs,\
                grad_accum_steps,\
                max_grad_norm,
                gamma=None,\
                train_algorithm=None,\
                rollout_size: int = 50,\
                eval_iterations: int = 10,\
                use_wandb: bool = False,
                save_path: str = None,
                decode_f: callable = lambda x: x,
                **kwargs):
    if train_algorithm == "digiq":
        print(">>> Using Digi-Q trainer")
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
                                max_grad_norm=max_grad_norm)
    elif train_algorithm == "filteredbc":
        print(">>> Using Filtered BC trainer")
        trainer = BCTrainer(agent=agent,\
                                tokenizer=tokenizer,\
                                accelerator=accelerator,
                                lm_lr = lm_lr,\
                                epochs = actor_epochs,\
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)

    agent.prepare()
    
    if os.path.exists(os.path.join(save_path, 'trainer_offline.pt')):
        print(">>> Loading from previous checkpoint")
        trainer.load(os.path.join(save_path, 'trainer_offline.pt'))
    else:
        print(">>> No previous checkpoint found")

    colorful_print(">>> Evaluating Agent", fg='blue')
    
    if os.path.exists(os.path.join(save_path, 'trajectories_eval.pt')):
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories_eval.pt'), weights_only=False)
    else:
        all_trajectories = []
    for i in range(eval_iterations):
        trajectories = batch_interact_environment(agent = agent,\
                                            env = env,\
                                            num_trajectories= rollout_size,\
                                            accelerator = accelerator,\
                                            use_tqdm=False,
                                            decode_f = decode_f,
                                            gamma = gamma,
                                            iter=i)
        if accelerator.is_main_process:
            info = {"iteration": i,\
                    "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "walltime": time.time()}
            all_trajectories += trajectories
            
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories_eval.pt'))
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories_eval.pt'), weights_only=False)
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
            