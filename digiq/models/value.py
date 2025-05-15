import torch
import torch.nn as nn
import torch.optim as optim

from .transition_model import Transition_Model

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int):
        super().__init__()
        self.attn_goal_to_state = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_state = nn.LayerNorm(embed_dim)

    def forward(self, state:torch.Tensor, goal:torch.Tensor) -> torch.Tensor:
        attn_output_state, _ = self.attn_goal_to_state(query=state, key=goal, value=goal)
        state = self.norm_state(state + attn_output_state)

        return state

class ValueNet(nn.Module):
    def __init__(self, state_dim:int, goal_dim:int, embed_dim, num_attn_layers:int, num_heads:int, device:str):
        super().__init__()

        self.embedding_state = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)
        
        self.embedding_goal = nn.Sequential(
            nn.Linear(goal_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)

        self.attention = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads) for _ in range(num_attn_layers)]
        ).to(device)

        self.critic = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
        ).to(device)

    def forward(self, state:torch.Tensor, goal:torch.Tensor) -> torch.Tensor:
        # MODULE 0 : Embedding
        state = self.embedding_state(state)
        goal = self.embedding_goal(goal)
        # MODULE 1 : Attention Layer
        for attention_layer in self.attention:
            state = attention_layer(state, goal)
        # MODULE 2 : MLP
        value = self.critic(state)

        return value