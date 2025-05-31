import torch
import torch.nn.functional as F
from torch.optim import Adam

# ── 1) Rollout collection with latent-space “TD-reward” ────────────────────────
def collect_latent_rollout(
    policy, value_fn, trans_model,
    init_states, tasks, rollout_length, gamma, device="cuda"
):
    policy.eval(); value_fn.eval(); trans_model.eval()
    B = init_states.shape[0]
    s = init_states.to(device)

    states, actions, log_probs, rewards, values = [], [], [], [], []

    with torch.no_grad():
        for _ in range(rollout_length):
            dist  = policy(s)
            a     = dist.sample()
            lp    = dist.log_prob(a).sum(-1)
            s_next, done, r  = trans_model(tasks, s, a)
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
    gamma=0.99, lam=0.95,
    clip_eps=0.2,  ent_coef=0.01,
    bremen_epochs=4, lr=3e-4,
    device="cuda"
):
    policy.to(device); trans_model.to(device)
    optimizer = Adam(policy.parameters(), lr=lr)

    for it in range(1, num_iters+1):
        
        init_states, tasks = sample_latent_starts(batch_size)
        # 1) collect latent rollout
        batch = collect_latent_rollout(
            policy, trans_model,
            init_states   = init_states.to(device),
            tasks         = tasks.to(device),
            rollout_length= rollout_length,
            gamma         = gamma,
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

def sample_latent_starts(B):
    """
    This function should return a batch of initial latent states and encoded tasks
    """
    init_states = torch.randn(B, STATE_DIM)  # Example: random latent states
    tasks = torch.zeros(B, TASK_DIM)  # Example: zero tasks (or some task encoding)
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
    trained_policy = train_model_based_bremen(
        policy, trans_model,
        num_iters      = 1000,
        batch_size     = 64,
        rollout_length = 50,
        gamma=0.99, lam=0.95,
        clip_eps=0.2, ent_coef=0.01,
        bremen_epochs=4, lr=3e-4,
        device="cuda"
    )
