import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int):
        super().__init__()
        self.attn_state_to_action = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_action_to_state = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_state = nn.LayerNorm(embed_dim)
        self.norm_action = nn.LayerNorm(embed_dim)

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        attn_output_action, _ = self.attn_state_to_action(query=action, key=state, value=state)
        action = self.norm_action(action + attn_output_action)

        attn_output_state, _ = self.attn_action_to_state(query=state, key=action, value=action)
        state = self.norm_state(state + attn_output_state)

        return state, action

class Transition_Model(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, embed_dim:int, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", device:str='cuda'):
        super().__init__()
        
        self.device = device

        if activation=="ReLU":
            self.activation = nn.ReLU()
        elif activation=="ELU":
            self.activation = nn.ELU()
        elif activation=="GELU":
            self.activation = nn.GELU()

        self.embedding_state = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            self.activation,
            nn.Linear(embed_dim, embed_dim),
        ).to(device)

        self.embedding_action = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            self.activation,
            nn.Linear(embed_dim, embed_dim),
        ).to(device)

        self.cross_attention = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_attn_layers)]
        ).to(device)
        
        self.mlp_next_state = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, state_dim),
            self.activation,
            nn.Linear(state_dim, state_dim),
        ).to(device)

        self.mlp_termial = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, 1),
        ).to(device)

        self.mlp_reward = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, embed_dim*2),
            self.activation,
            nn.Linear(embed_dim*2, 1),
        ).to(device)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(self.activation, nn.LeakyReLU):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                elif isinstance(self.activation, nn.ELU):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                
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

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        # MODULE 0 : Embedding
        state = self.embedding_state(state)
        action = self.embedding_action(action)
        # MODULE 1 : Attention Layer
        for cross_attention_layer in self.cross_attention:
            state, action = cross_attention_layer(state, action)
        # MODULE 2 : MLP
        cat = torch.cat([state, action], dim=-1)
        next_state = self.mlp_next_state(cat)
        terminal = torch.softmax(self.mlp_termial(cat))
        reward = self.mlp_reward(cat)

        return next_state, terminal, reward