import torch
import torch.nn as nn

class MLPTransition(nn.Module):
    def __init__(self, state_dim:int=None, action_dim:int=None,
                 num_hidden_layers: int=6, dropout: float=0.1, res_per_layers: int=2, 
                 activation:str="ReLU", device:str='cuda'):
        super().__init__()
        self.device = device

        if activation=="ReLU":
            self.activation = nn.ReLU()
        elif activation=="ELU":
            self.activation = nn.ELU()
        elif activation=="GELU":
            self.activation = nn.GELU()
        self.res_per_layers = res_per_layers

        self.input_dim = state_dim + action_dim
        hidden_dim = self.input_dim * 2

        self.input_proj = nn.Linear(self.input_dim, hidden_dim).to(device)
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ).to(device))
        self.output_proj = nn.Linear(hidden_dim, state_dim).to(device)
    
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

    def forward(self, state:torch.Tensor, action:torch.Tensor, goal:torch.Tensor=None) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)

        for i, layer in enumerate(self.hidden_layers):
            if self.res_per_layers > 0 and (i + 1) % self.res_per_layers == 0:
                residual = x
            x = layer(x)
            if self.res_per_layers > 0 and (i + 1) % self.res_per_layers == 0:
                x = x + residual
        
        return self.output_proj(x)
        
        state = self.embedding_state(state)
        action = self.embedding_action(action)

        for cross_attention_layer in self.cross_attention:
            state, action = cross_attention_layer(state, action)
        next_state = self.mlp_next_state(torch.cat([state, action], dim=-1))
        
        if goal is not None:
            goal = self.embedding_goal(goal)
            for cross_attention_layer in self.cross_attention_with_goal:
                state, goal = cross_attention_layer(state, goal)
            terminal = torch.sigmoid(self.mlp_termial(torch.cat([state, goal], dim=-1)))
            reward = self.mlp_reward(torch.cat([state, goal], dim=-1))
            return next_state, terminal, reward
        else:
            return next_state
        

class TransformerTransition(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 model_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 device:str='cuda'):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, model_dim).to(device)
        self.action_proj = nn.Linear(action_dim, model_dim).to(device)
        self.pos_embedding = nn.Parameter(torch.zeros(2, model_dim)).to(device)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4*model_dim,
            dropout=dropout,
            activation='relu'
        ).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        self.output_proj = nn.Linear(model_dim, state_dim).to(device)

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

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 投影
        s = self.state_proj(state)  # (batch, model_dim)
        a = self.action_proj(action)    # (batch, model_dim)
        # 构造序列 (seq_len=2, batch, model_dim)
        seq = torch.stack([s, a], dim=0)
        seq = seq + self.pos_embedding.unsqueeze(1)
        # Transformer 编码
        enc_out = self.transformer(seq)  # (2, batch, model_dim)
        # 取首 token（state）输出作为聚合特征
        feat = enc_out[0]
        return self.output_proj(feat).squeeze(-1)