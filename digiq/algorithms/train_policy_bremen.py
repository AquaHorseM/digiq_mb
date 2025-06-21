import torch
import torch.nn.functional as F
from torch.optim import Adam
from digiq.models.encoder import GoalEncoder, ActionEncoder
from digiq.models.transition_model import Transition_Model
from digiq.models.agent import Agent
from digiq.models.value_model import Value_Model

import argparse
import random
import yaml
import os

def collect_latent_rollout(
    policy, value_fn, trans_model,
    init_states, tasks, rollout_length, gamma, action_encoder, device="cuda"
):
    policy.eval(); value_fn.eval(); trans_model.eval()
    B = init_states.shape[0]
    s = init_states.to(device)

    states, actions, log_probs, rewards, values, past_actions_str = [], [], [], [], [], []
    
    current_past_action_str = [""] * B

    with torch.no_grad():
        for _ in range(rollout_length):
            dist = policy(s, tasks, current_past_action_str)
            
            a = policy.sample_action(dist)
            lp = policy.compute_log_prob(dist, a)

            a_str = [policy.process_action_tensor2str(a_i, goal_i) for a_i, goal_i in zip(a, tasks)]
            a_encoded = action_encoder(a_str).to(device)
            s_next, done, r = trans_model(s, a_encoded, tasks)
            v_next = v_next.squeeze(-1)

            states.append(s)
            values.append(v_next)
            actions.append(a)
            log_probs.append(lp)
            rewards.append(r.squeeze(-1))
            past_actions_str.append(current_past_action_str)
            s = s_next
            current_past_action_str = a_str

        v_final, _ = value_fn(s, tasks, current_past_action_str)
        v_final = v_final.squeeze(-1)

    return {
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions, dim=1),
        "log_probs": torch.stack(log_probs, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "values": torch.stack(values, dim=1),
        "v_final": v_final,
        "past_actions": past_actions_str,
        "tasks": tasks
    }

def compute_gae(rewards, values, v_final, gamma, lam):
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        next_val = v_final if t == T-1 else values[:, t+1]
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        adv[:, t] = last_gae

    returns = adv + values
    return adv, returns

def bremen_update(
    policy, optimizer, rollouts,
    adv, returns, clip_eps, ent_coef, max_grad_norm
):
    B, T = adv.shape
    old_lp = rollouts["log_probs"].reshape(-1)
    states = rollouts["states"].reshape(B*T, -1)
    actions = rollouts["actions"].reshape(B*T, -1)
    
    tasks_tensor = rollouts["tasks"].unsqueeze(1).expand(-1, T, -1).reshape(B*T, -1)

    past_actions_flat = [item for sublist in zip(*rollouts["past_actions"]) for item in sublist]

    adv_flat = adv.reshape(-1)

    dist = policy(states, tasks_tensor, past_actions_flat)
    new_lp = policy.compute_log_prob(dist, actions)
    entropy = dist.entropy().mean()

    ratio = (new_lp - old_lp).exp()
    surr1 = ratio * adv_flat
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_flat
    policy_loss = -torch.min(surr1, surr2).mean()

    loss = policy_loss - ent_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return policy_loss.item(), entropy.item()

def train_model_based_bremen(
    policy, value_fn, trans_model,
    num_iters, batch_size, rollout_length,
    data_file,
    gamma=0.99, lam=0.95,
    clip_eps=0.2,  ent_coef=0.01,
    bremen_epochs=4, lr=3e-4, max_grad_norm=0.5,
    goal_encoder=None, action_encoder=None,
    device="cuda"
):
    policy.to(device)
    value_fn.to(device)
    trans_model.to(device)
    optimizer = Adam(policy.parameters(), lr=lr)
    steps = torch.load(data_file, weights_only=False)

    for it in range(1, num_iters+1):
        
        init_states, tasks = sample_latent_starts(batch_size, steps, goal_encoder, device)

        batch = collect_latent_rollout(
            policy, value_fn, trans_model,
            init_states=init_states.to(device),
            tasks=tasks.to(device),
            rollout_length=rollout_length,
            gamma=gamma,
            action_encoder=action_encoder,
            device=device
        )

        adv, returns = compute_gae(
            batch["rewards"], batch["values"], batch["v_final"],
            gamma=gamma, lam=lam
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(bremen_epochs):
            p_loss, ent = bremen_update(
                policy, optimizer, batch,
                adv, returns,
                clip_eps, ent_coef,
                max_grad_norm=max_grad_norm
            )

        print(f"[Iter {it:3d}] π_loss={p_loss:.4f} ent={ent:.4f}")

    return policy

def sample_latent_starts(B, steps, goal_encoder, device):
    sampled_steps = random.sample(steps, B)
    init_states = torch.stack([step['s_rep'] for step in sampled_steps])
    tasks_str = [step['task'] for step in sampled_steps]
    tasks_encoded = goal_encoder(tasks_str).to(device)
    return init_states, tasks_encoded

# ──  Example of how to call it ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to offline data for sampling initial states.")
    parser.add_argument("--config_path", type=str, default="scripts/config/main/bremen_rl.yaml", help="Path to the main YAML config file.")
    parser.add_argument("--transition_model_path", type=str, required=True, help="Path to the pretrained transition model weights.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to the pretrained value model weights.")
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent_config = config['Agent']
    value_config = config['Value_Model']
    trans_config = config['TransitionModel']
    goal_enc_config = config['Goal_encoder']
    action_enc_config = config['Action_encoder']

    goal_encoder = GoalEncoder(
        backbone=goal_enc_config['goal_encoder_backbone'],
        cache_dir=goal_enc_config['goal_encoder_cache_dir'],
        device=device
    )
    action_encoder = ActionEncoder(
        backbone=action_enc_config['action_encoder_backbone'],
        cache_dir=action_enc_config['action_encoder_cache_dir'],
        device=device
    )

    policy = Agent(
        state_dim=agent_config['state_dim'],
        goal_dim=agent_config['goal_dim'],
        action_dim=agent_config['action_dim'],
        embed_dim=agent_config['embed_dim'],
        num_sce_type=agent_config['num_sce_type'],
        latent_action_dim=agent_config['latent_action_dim'],
        num_attn_layers_first=agent_config['num_attn_layers_first'],
        num_heads_first=agent_config['num_heads_first'],
        num_attn_layers_second=agent_config['num_attn_layers_second'],
        num_heads_second=agent_config['num_heads_second'],
        goal_encoder_backbone=goal_enc_config['goal_encoder_backbone'],
        goal_encoder_cache_dir=goal_enc_config['goal_encoder_cache_dir'],
        action_encoder_backbone=action_enc_config['action_encoder_backbone'],
        action_encoder_cache_dir=action_enc_config['action_encoder_cache_dir'],
        typing_lm=agent_config['typing_lm'],
        device=device
    )

    value_fn = Value_Model(
        state_dim=value_config['state_dim'],
        goal_dim=value_config['goal_dim'],
        action_dim=value_config['action_dim'],
        embed_dim=value_config['embed_dim'],
        num_attn_layers=value_config['num_attn_layers'],
        num_heads=value_config['num_heads'],
        goal_encoder_backbone=goal_enc_config['goal_encoder_backbone'],
        goal_encoder_cache_dir=goal_enc_config['goal_encoder_cache_dir'],
        action_encoder_backbone=action_enc_config['action_encoder_backbone'],
        action_encoder_cache_dir=action_enc_config['action_encoder_cache_dir'],
        device=device
    )

    trans_model = Transition_Model(
        state_dim=trans_config['state_dim'],
        action_dim=trans_config['action_dim'],
        goal_dim=trans_config['goal_dim'],
        embed_dim=trans_config['embed_dim'],
        num_attn_layers=trans_config['num_attn_layers'],
        num_heads=trans_config['num_heads'],
        activation=trans_config['activation'],
        device=device
    )

    trans_model.load_state_dict(torch.load(args.transition_model_path, map_location=device))
    value_fn.load_state_dict(torch.load(args.value_model_path, map_location=device))

    trained_policy = train_model_based_bremen(
        policy=policy,
        value_fn=value_fn,
        trans_model=trans_model,
        num_iters=config['train']['num_iters'],
        batch_size=config['train']['batch_size'],
        rollout_length=config['train']['rollout_length'],
        data_file=args.data_file,
        gamma=config['train']['gamma'],
        lam=config['train']['lam'],
        clip_eps=config['train']['clip_eps'],
        ent_coef=config['train']['ent_coef'],
        bremen_epochs=config['train']['bremen_epochs'],
        lr=config['train']['lr'],
        max_grad_norm=config['train']['max_grad_norm'],
        goal_encoder=goal_encoder,
        action_encoder=action_encoder,
        device=device,
    )