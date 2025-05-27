import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GoalEncoder, ActionEncoder

def init_weight(module:nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=torch.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / torch.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)

    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0)
        nn.init.xavier_uniform_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.constant_(module.out_proj.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif hasattr(module, 'weight') and module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int):
        super().__init__()
        self.attn_goal_to_state = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_state = nn.LayerNorm(embed_dim)

    def forward(self, state:torch.Tensor, goal:torch.Tensor) -> torch.Tensor:
        attn_output_state, _ = self.attn_goal_to_state(query=state, key=goal, value=goal)
        state = self.norm_state(state + attn_output_state)

        return state

class CrossLayer(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossLayer, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for i in range(self.num_layers):
            xw = torch.sum(x * self.weights[i], dim=1, keepdim=True)  # (batch, 1)
            x = x0 * xw + self.biases[i] + x
        return x

class FeatureSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super(FeatureSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        attn_output, _ = self.attention.forward(x, x, x)
        return attn_output.squeeze(1)

class CrossAttentionMLPModel(nn.Module):
    def __init__(self, input_dim, cross_layers=2, attn_heads=1, mlp_hidden=[64, 32], output_dim=1):
        super(CrossAttentionMLPModel, self).__init__()
        self.cross = CrossLayer(input_dim, cross_layers)
        self.attention = FeatureSelfAttention(input_dim, attn_heads)
        self.mlp = MLP(input_dim * 2, mlp_hidden, output_dim)  # concatenate cross + attn

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_cross = self.cross(x)
        x_attn = self.attention(x)
        x_cat = torch.cat([x_cross, x_attn], dim=1)
        out = self.mlp(x_cat)
        return out

class Agent(nn.Module):
    def __init__(
        self, state_dim:int, goal_dim:int, action_dim, embed_dim:int, num_sce_type:int, latent_action_dim:int, num_attn_layers_first:int, num_heads_first:int, num_attn_layers_second:int, num_heads_second:int,
        goal_encoder_backbone:str, goal_encoder_cache_dir:str, action_encoder_backbone:str, action_encoder_cache_dir:str,
        device:str
    ):
        super().__init__()

        self.action_encoder = ActionEncoder(backbone=action_encoder_backbone, cache_dir=action_encoder_cache_dir, device=device)
        self.goal_encoder = GoalEncoder(backbone=goal_encoder_backbone, cache_dir=goal_encoder_cache_dir, device=device)

        self.embedding_state = MLP(state_dim, hidden_dims=[embed_dim], output_dim=embed_dim).to(device)
        self.embedding_goal = MLP(input_dim=goal_dim, hidden_dims=[embed_dim], output_dim=embed_dim).to(device)
        self.embedding_past_action = MLP(input_dim=action_dim, hidden_dims=[embed_dim], output_dim=embed_dim).to(device)
        self.embedding_others = MLP(input_dim=embed_dim*2, hidden_dims=[embed_dim*2, embed_dim*2], output_dim=embed_dim).to(device)

        self.attention = nn.ModuleList([AttentionBlock(embed_dim, num_heads_first) for _ in range(num_attn_layers_first)]).to(device)
        self.latent_action = CrossAttentionMLPModel(input_dim=embed_dim*3, cross_layers=num_attn_layers_second, attn_heads=num_heads_second, mlp_hidden=[embed_dim*3, embed_dim*3], output_dim=latent_action_dim).to(device)

        self.action_type = MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim//2], output_dim=4).to(device)
        self.typing_type = MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim, latent_action_dim//2], output_dim=num_sce_type).to(device)
        self.bottom_button_type = MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim], output_dim=3).to(device)
        self.touch_coord = MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim, latent_action_dim, latent_action_dim], output_dim=2).to(device)
        self.scroll_from_coord = MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim, latent_action_dim, latent_action_dim], output_dim=2).to(device)
        self.scroll_to_coord = MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim, latent_action_dim, latent_action_dim], output_dim=2).to(device)

    def init_weight(self):
        self.apply(init_weight())

    def forward(self, state:torch.Tensor, goal:torch.Tensor, past_action:torch.Tensor, determine:bool=False) -> torch.Tensor:
        # MODULE 0 : Embedding
        state = self.embedding_state(state)
        goal = self.embedding_goal(self.goal_encoder(goal))
        past_action = self.embedding_past_action(self.action_encoder(past_action))
        others = self.embedding_others(torch.cat(goal, past_action))
        # MODULE 1 : Attention Layer
        for attention_layer in self.attention:
            state = attention_layer(state, others)
        # MODULE 2 : MLP
        latent_action = self.latent_action(torch.cat(state, others))
        # MODULE 3 : Action
        if determine:
            action_type = torch.argmax(self.action_type(latent_action))
        else:
            action_type = torch.multinomial(F.softmax(self.action_type(latent_action)), num_samples=1)
        
        typing_type = 0.0
        bottom_button_type = 0.0
        touch_coord_x, touch_coord_y = 0.0, 0.0
        scroll_from_x, scroll_from_y = 0.0, 0.0
        scroll_to_x, scroll_to_y = 0.0, 0.0
        
        if action_type == 0: # typing
            if determine:
                typing_type = torch.argmax(self.typing_type(latent_action))
            else:
                typing_type = torch.multinomial(F.softmax(self.typing_type(latent_action)), num_samples=1)
        elif action_type == 1: # bottom_button
            if determine:
                bottom_button_type = torch.argmax(self.bottom_button_type(latent_action))
            else:
                bottom_button_type = torch.multinomial(F.softmax(self.bottom_button_type(latent_action)), num_samples=1)
        elif action_type == 2: # touch
            touch_coord_x, touch_coord_y = self.touch_coord(latent_action)
        elif action_type == 3: # scroll
            scroll_from_x, scroll_from_y = self.scroll_from_coord(latent_action)
            scroll_to_x, scroll_to_y = self.scroll_to_coord(latent_action)

        action = torch.tensor([typing_type, bottom_button_type, touch_coord_x, touch_coord_y, scroll_from_x, scroll_from_y, scroll_to_x, scroll_to_y], dtype=state.dtype, device=state.device)
        return action