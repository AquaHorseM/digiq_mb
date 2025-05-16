import torch
import torch.nn as nn
import torch.optim as optim

from .transition_model import Transition_Model
from .encoder import GoalEncoder, ActionEncoder

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int):
        super().__init__()
        self.attn_goal_to_state = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_state = nn.LayerNorm(embed_dim)

    def forward(self, state:torch.Tensor, goal:torch.Tensor) -> torch.Tensor:
        attn_output_state, _ = self.attn_goal_to_state(query=state, key=goal, value=goal)
        state = self.norm_state(state + attn_output_state)

        return state

class ValueModel(nn.Module):
    def __init__(self, state_dim:int, goal_dim:int, action_dim, embed_dim:int, num_attn_layers:int, num_heads:int,
                 goal_encoder_backbone:str, goal_encoder_cache_dir:str, action_encoder_backbone:str, action_encoder_cache_dir:str,
                 device:str):
        super().__init__()

        self.action_encoder = ActionEncoder(backbone=action_encoder_backbone, cache_dir=action_encoder_cache_dir, device=device)
        self.goal_encoder = GoalEncoder(backbone=goal_encoder_backbone, cache_dir=goal_encoder_cache_dir, device=device)

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

        self.embedding_past_action = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)

        self.embedding_others = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim),
        )

        self.attention = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads) for _ in range(num_attn_layers)]
        ).to(device)

        self.critic1 = nn.Sequential(
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

        self.critic2 = nn.Sequential(
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

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Embedding):
                with torch.no_grad():
                    one_hot = torch.eye(m.num_embeddings, m.embedding_dim)
                    noise = torch.randn_like(one_hot) * 0.1
                    m.weight.copy_(one_hot + noise)

    def forward(self, state:torch.Tensor, goal:str, past_action:str) -> torch.Tensor:
        # MODULE 0 : Embedding
        state = self.embedding_state(state)
        goal = self.embedding_goal(self.goal_encoder(goal))
        past_action = self.embedding_past_action(self.action_encoder(past_action))
        others = self.embedding_others(torch.cat(goal, past_action))
        # MODULE 1 : Attention Layer
        for attention_layer in self.attention:
            state = attention_layer(state, others)
        # MODULE 2 : MLP
        return self.critic1(state), self.critic2(state)