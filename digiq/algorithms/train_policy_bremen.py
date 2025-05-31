import torch
import torch.nn.functional as F
from torch.optim import Adam
from digiq.models.encoder import GoalEncoder, ActionEncoder

import argparse
import random

# ── 1) Rollout collection with latent-space “TD-reward” ────────────────────────
def collect_latent_rollout(
    policy, value_fn, trans_model,
    init_states, tasks, rollout_length, gamma, action_encoder, device="cuda"
):
    policy.eval(); value_fn.eval(); trans_model.eval()
    B = init_states.shape[0]
    s = init_states.to(device)

    states, actions, log_probs, rewards, values = [], [], [], [], []

    with torch.no_grad():
        for _ in range(rollout_length):
            dist  = policy(s)
            a     = policy.sample_action(dist)
            lp    = policy.compute_log_prob(a, dist)
            a_str = policy.process_action_tensor2str(a)
            a_encoded = action_encoder(a_str).to(device)  # encode action if needed
            s_next, done, r  = trans_model(tasks, s, a_encoded)
            v_next = value_fn(tasks, s_next).squeeze(-1)
            
            states.append(s)
            values.append(v_next)
            actions.append(a)
            log_probs.append(lp)
            rewards.append(r)
            s = s_next

        v_final = value_fn(tasks, s).squeeze(-1)

    return {
        "states":    torch.stack(states,    dim=1),
        "actions":   torch.stack(actions,   dim=1),
        "log_probs": torch.stack(log_probs, dim=1),
        "rewards":   torch.stack(rewards,   dim=1),
        "values":    torch.stack(values,    dim=1),
        "v_final":   v_final
    }

# ── 2) GAE advantage & return computation ─────────────────────────────────────
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

# ── 3) bremen-style update ───────────────────────────────────────────────────────
def bremen_update(
    policy, optimizer, rollouts,
    adv, returns, clip_eps, ent_coef, max_grad_norm
):
    B, T = adv.shape
    # flatten
    old_lp = rollouts["log_probs"].reshape(-1)
    states  = rollouts["states"].reshape(B*T, -1)
    actions = rollouts["actions"].reshape(B*T, -1)
    adv_flat = adv.reshape(-1)
    ret_flat = returns.reshape(-1)

    dist       = policy(states)
    new_lp     = dist.log_prob(actions).sum(-1)
    entropy    = dist.entropy().mean()

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

# ── 4) Main training loop ──────────────────────────────────────────────────────
def train_model_based_bremen(
    policy, trans_model,
    num_iters, batch_size, rollout_length,
    data_file,
    gamma=0.99, lam=0.95,
    clip_eps=0.2,  ent_coef=0.01,
    bremen_epochs=4, lr=3e-4,
    goal_encoder=None, action_encoder=None,
    device="cuda"
):
    policy.to(device); trans_model.to(device)
    optimizer = Adam(policy.parameters(), lr=lr)

    # load data
    steps = torch.load(data_file, weights_only=False)

    for it in range(1, num_iters+1):
        
        init_states, tasks = sample_latent_starts(batch_size, steps, goal_encoder)
        # 1) collect latent rollout
        batch = collect_latent_rollout(
            policy, trans_model,
            init_states   = init_states.to(device),
            tasks         = tasks.to(device),
            rollout_length= rollout_length,
            gamma         = gamma,
            action_encoder= action_encoder,
            device        = device
        )

        # 2) compute GAE & returns
        adv, returns = compute_gae(
            batch["rewards"], batch["values"], batch["v_final"],
            gamma=gamma, lam=lam
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 3) bremen updates
        for _ in range(bremen_epochs):
            p_loss, ent = bremen_update(
                policy, optimizer, batch,
                adv, returns,
                clip_eps, ent_coef,
                max_grad_norm=0.5
            )

        print(f"[Iter {it:3d}] π_loss={p_loss:.4f} ent={ent:.4f}")

    return policy

def sample_latent_starts(B, steps, goal_encoder):
    """
    This function should return a batch of initial latent states and encoded tasks
    """
    sampled_steps = random.sample(steps, B)
    init_states = torch.stack([step['s_rep'] for step in sampled_steps]) # 2d tensor
    tasks = [goal_encoder(step['task']) for step in sampled_steps] # list of str
    tasks = torch.stack(tasks, dim=0) 
    return init_states, tasks

# ──  Example of how to call it ────────────────────────────────────────────────
if __name__ == "__main__":
    # define or import your networks...
    # policy       = MyPolicyNet(STATE_DIM, ACTION_DIM)
    # policy.value_head = MyValueHead(...)
    # value_fn     = MyValueNet(STATE_DIM)
    # trans_model  = MyTransNet(STATE_DIM, ACTION_DIM)
    # def sample_latent_starts(B): ...
    #

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="", help="Path to offline data")
    args = parser.parse_args()
    
    #load the config from "scripts/config/main/bremen_rl.yaml" and initialize the goalencoder and action encoder
    import yaml
    with open("scripts/config/main/bremen_rl.yaml", 'r') as f:
        config = yaml.safe_load(f)
    goal_encoder_config = config['Goal_encoder']
    action_encoder_config = config['Action_encoder']
    goal_encoder = GoalEncoder(
        backbone=goal_encoder_config['goal_encoder_backbone'],
        cache_dir=goal_encoder_config['goal_encoder_cache_dir'],
        device="cuda"
    )
    action_encoder = ActionEncoder(
        backbone=action_encoder_config['action_encoder_backbone'],
        cache_dir=action_encoder_config['action_encoder_cache_dir'],
        device="cuda"
    )

    trained_policy = train_model_based_bremen(
        policy, trans_model,
        num_iters      = 1000,
        batch_size     = 64,
        rollout_length = 50,
        data_file = args.data_file
        gamma=0.99, lam=0.95,
        clip_eps=0.2, ent_coef=0.01,
        bremen_epochs=4, lr=3e-4,
        goal_encoder=goal_encoder,
        action_encoder=action_encoder,
        device="cuda"
    )
