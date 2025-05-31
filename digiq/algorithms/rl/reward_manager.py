import torch
from models.value_model import Value_Model
from models.transition_model import Transition_Model

class TransitionRewardManager:
    def __init__(
        self, state_dim:int, goal_dim:int, action_dim:int, embed_dim:int, 
        num_attn_layers_transition:int, num_heads_transition:int, num_attn_layers_value:int, num_heads_value:int,
        goal_encoder_backbone:str, goal_encoder_cache_dir:str, action_encoder_backbone:str, action_encoder_cache_dir:str,
        transition_load_path:str, value_load_path:str, activation:str="ReLU", device:str='cuda',
    ):
        self.value = Value_Model(state_dim=state_dim, goal_dim=goal_dim, action_dim=action_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers_value, num_heads=num_heads_value, goal_encoder_backbone=goal_encoder_backbone, goal_encoder_cache_dir=goal_encoder_cache_dir, action_encoder_backbone=action_encoder_backbone, action_encoder_cache_dir=action_encoder_cache_dir, device=device)
        self.transition = Transition_Model(state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim, num_attn_layers=num_attn_layers_transition, num_heads=num_heads_transition, activation=activation, device=device)

        self.value.load_state_dict(torch.load(value_load_path, map_location=device, weights_only=True))
        self.transition.load_state_dict(torch.load(transition_load_path, map_location=device, weights_only=True))

    def __call__(self, batch:torch.Tensor) -> torch.Tensor:
        state, action = batch
        with torch.no_grad():
            next_state, _, _ = self.transition.forward(state, action)
            reward = self.value(next_state)
        return reward