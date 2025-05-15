import torch
import torch.nn as nn
import torch.optim as optim

from .transition_model import Transition_Model
from .value_model import ValueModel

class RewardModel(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, goal_dim:int, embed_dim, 
                 num_attn_layers_transition:int, num_heads_transition:int, num_attn_layers_value:int, num_heads_value:int, device:str):
        super().__init__()

        self.transition = Transition_Model(state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers_transition, num_heads=num_heads_transition, device=device)
        self.value = ValueModel(state_dim=state_dim, goal_dim=goal_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers_value, num_heads=num_heads_value, device=device)
    
    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        next_state = self.transition.forward(state, action)
        value = self.value.forward(next_state)
        return value