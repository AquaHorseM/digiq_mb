import os, re
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset

import wandb

from digiq.models.value_model import Value_Model
from digiq.data.utils import ReplayBuffer, ReplayBufferDataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ValueModelTrainerSimple:
    def __init__(
        self,
        save_path: str,
        state_dim: int,
        goal_dim: int,
        embed_dim: int,
        num_attn_layers: int,
        num_heads: int,
        goal_encoder_backbone: str,
        goal_encoder_cache_dir: str,
        lr: float,
        seed: int = 0,
    ):
        # reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        self.model = Value_Model(
            state_dim=state_dim,
            goal_dim=goal_dim,
            embed_dim=embed_dim,
            num_attn_layers=num_attn_layers,
            num_heads=num_heads,
            goal_encoder_backbone=goal_encoder_backbone,
            goal_encoder_cache_dir=goal_encoder_cache_dir,
            device=self.device,
        )
        #make sure the dtype is float32
        self.model.init_weight()
        self.model = self.model.to(dtype=torch.float32, device=self.device)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # where to save
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def compute_loss(self, batch):
        state = batch["s_rep"].to(dtype=torch.float32, device=self.device)           # [B, ...]
        observations = batch["observation"]
        if isinstance(observations, str):
            observations = [observations]
        goals = []
        for obs in observations:
            goal_match  = re.search(r'Goal: (.*?)</s>', obs)
            if goal_match:
                goal = goal_match.group(1)
            else:
                goal = ""
            goal = self.model.goal_encoder(goal).to(dtype=torch.float32, device=self.device)  # [B, goal_dim]
            goals.append(goal)
        goal = torch.stack(goals, dim=0)            # [B, goal_dim]
        mc = batch["mc_return"].to(dtype=torch.float32, device=self.device)      # [B]

        # forward; handles goal encoding internally
        pred = self.model(state, goal).view(-1).to(dtype=torch.float32, device=self.device)         # [B]
        return F.mse_loss(pred, mc)

    def train(
        self,
        data_path: str,
        batch_size: int = 512,
        epochs: int = 50,
        eval_every: int = 5,
    ):
        # 1. load data into buffer
        if not os.path.exists(data_path):
            print(f"Data path {data_path} is a huggingface dataset, loading it now...")
            raw = load_dataset(data_path, "general", split="train")
        else:
            raw = torch.load(data_path, weights_only=False)
        buffer = ReplayBuffer(batch_size, capacity=len(raw))
        for item in raw:
            buffer.insert(**item)

        dataset = ReplayBufferDataset(buffer)
        # random split into train and validation sets
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        # sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
        # loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # 2. optional: wandb init
        wandb.init(project="value-model", config={
            "batch_size": batch_size,
            "lr":        self.optimizer.param_groups[0]["lr"],
            "epochs":   epochs,
        })

        # 3. training loop
        print(f"Training Value Model for {epochs} epochs...")
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # batch.to(dtype=torch.float32, device=self.device)  # ensure batch is float32 and on the correct device
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch).float()  # compute loss, ensure it's float32
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches    += 1

            avg_loss = running_loss / n_batches
            print(f"[Epoch {epoch:02d}]  MSE Loss: {avg_loss:.6f}")
            wandb.log({"train/mse": avg_loss, "epoch": epoch})
            # 3.1. validation
            if epoch % eval_every == 0:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        # val_batch.to(dtype=torch.float32, device=self.device)  # ensure batch is float32 and on the correct device
                        loss = self.compute_loss(val_batch).float()
                        val_loss += loss.item()
                        n_val_batches += 1
                avg_val_loss = val_loss / n_val_batches
                print(f"[Epoch {epoch:02d}] Validation MSE Loss: {avg_val_loss:.6f}")
                wandb.log({"val/mse": avg_val_loss, "epoch": epoch})

                if avg_val_loss < best_loss:
                    # 3.2 save best model
                    best_loss = avg_val_loss
                    save_file = os.path.join(self.save_path, "value_model_best.pth")
                    torch.save(self.model.state_dict(), save_file)
                    print(f"Saved best model to {save_file}")

        # 4. save final weights
        save_file = os.path.join(self.save_path, "value_model_final.pth")
        torch.save(self.model.state_dict(), save_file)
        print(f"Saved final model to {save_file}")
        wandb.finish()


if __name__ == "__main__":
    # 🛠 adjust these to your config or pass via argument parsing
    cfg = {
        "save_path": "/data/mqj/models/value-ws",
        "state_dim": 3584,
        "goal_dim":  768,
        "embed_dim": 1024,
        "num_attn_layers": 0,
        "num_heads":  8,
        "goal_encoder_backbone": "roberta-base",
        "goal_encoder_cache_dir": None,
        "lr": 1e-4,
        "seed": 42,
        "data_path": "/data/mqj/datasets/rl/webshop-ft.pt",
        "batch_size": 512,
        "epochs": 30,
        "eval_every": 5,
    }

    trainer = ValueModelTrainerSimple(
        save_path=cfg["save_path"],
        state_dim=cfg["state_dim"],
        goal_dim=cfg["goal_dim"],
        embed_dim=cfg["embed_dim"],
        num_attn_layers=cfg["num_attn_layers"],
        num_heads=cfg["num_heads"],
        goal_encoder_backbone=cfg["goal_encoder_backbone"],
        goal_encoder_cache_dir=cfg["goal_encoder_cache_dir"],
        lr=cfg["lr"],
        seed=cfg["seed"],
    )
    trainer.train(
        data_path=cfg["data_path"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        eval_every=cfg["eval_every"],
    )
