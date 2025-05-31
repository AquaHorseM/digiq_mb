import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import ast

from digiq.models.encoder import GoalEncoder, ActionEncoder

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
        goal_encoder_backbone:str, goal_encoder_cache_dir:str, action_encoder_backbone:str, action_encoder_cache_dir:str, typing_lm:str,
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
        self.action = nn.Sequential([
            CrossAttentionMLPModel(input_dim=embed_dim*3, cross_layers=num_attn_layers_second, attn_heads=num_heads_second, mlp_hidden=[embed_dim*3, embed_dim*3], output_dim=latent_action_dim).to(device),
            MLP(input_dim=latent_action_dim, hidden_dims=[latent_action_dim, latent_action_dim, latent_action_dim, latent_action_dim], output_dim=18).to(device)
        ])

        # LLM for typing
        self.tokenizer = AutoTokenizer.from_pretrained(typing_lm)
        self.typing_lm = AutoModelForCausalLM.from_pretrained(typing_lm, torch_dtype="auto", device_map="auto")
        self.step_goals = ["asking questions", "searching for things", "visiting websites"]

    def init_weight(self):
        self.apply(init_weight())

    def forward(self, state:torch.Tensor, goal:str|torch.Tensor, past_action:str|torch.Tensor, determine:bool=False) ->torch.Tensor:
        # MODULE 0 : Embedding
        state = self.embedding_state(state)
        if isinstance(goal, str):
            goal = self.embedding_goal(self.goal_encoder(goal))
        if isinstance(past_action, str):
            past_action = self.embedding_past_action(self.action_encoder(past_action))
        others = self.embedding_others(torch.cat(goal, past_action))
        # MODULE 1 : Attention Layer
        for attention_layer in self.attention:
            state = attention_layer(state, others)
        # MODULE 2 : MLP
        # 18维向量
        # 0表示typing, 1表示导航栏1, 2表示导航栏2, 3表示导航栏3, 4表示touch, 5表示scroll. 均理解为logits.
        # 剩下的维度单个坐标的mu和sigma, 有 3*2*2 = 12个
        action_dist = self.action(torch.cat(state, others))

    def sample_action(self, action_dist: torch.Tensor) -> torch.Tensor:
        batch_size = action_dist.size(0)

        type_logits = action_dist[:,:6]
        type_dist = torch.distributions.Categorical(logits=type_logits)
        type_one_hot = F.one_hot(type_dist.sample(), num_classes=6).float()

        mus = action_dist[:, 6::2]
        log_stds = action_dist[:, 7::2]
        stds = torch.exp(log_stds)
        coord_dist = torch.distributions.Normal(mus, stds)
        coords = coord_dist.rsample()

        action = torch.cat((type_one_hot, coords), dim=1)
        
        return action

    def compute_log_prob(self, action_dist: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        type_logits = action_dist[:, :6]
        type_dist = torch.distributions.Categorical(logits=type_logits)
        action_type = action[:,:6].argmax(dim=-1)
        log_prob_type = type_dist.log_prob(action_type)

        mus = action_dist[:, 6::2]
        log_stds = action_dist[:, 7::2]
        stds = torch.exp(log_stds)
        action_coords = action[:,6:]
        coord_dist = torch.distributions.Normal(mus, stds)
        log_prob_coords = coord_dist.log_prob(action_coords).sum(dim=-1)

        log_prob = log_prob_type + log_prob_coords
        
        return log_prob
    
    def get_typed_text(self, goal:str, step_goal) -> str:
        prompt = f"A phone user wants to acheive the following goal: \"{goal}\". The user needs to enter text in the input box of the user interface at a single step: **{step_goal}**. Provide the text the user should input. Your response should contain only the text the user should enter, without any additional information."

        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.typing_lm.device)

        generated_ids = self.typing_lm.generate(
            **model_inputs,
            max_new_tokens=32
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def get_pi_action(self, observation:str, image_features:torch.Tensor) -> str:
        goal = observation.split("Goal: ")[-1]
        past_action = observation.split("Previous Actions: ")[-1].split("Goal: ")[0]
        action, action_type_logits, typing_type_logits, bottom_button_type_logits = self.forward(image_features, goal, past_action)
        return self.process_action_tensor2str(action, goal)

    def process_action_str2tensor(action_str: str) -> dict[str:torch.Tensor|tuple[torch.Tensor, torch.Tensor]]:
        action_str = action_str.replace("Action Decision: ", "").strip()

        components = action_str.split(", ")
        action = {}

        action_type_str = components[0].split(": ")[1].replace("\"", "")
        if action_type_str == "TYPE":
            action['action_type'] = 0
        elif action_type_str == "PRESS_HOME":
            action['action_type'] = 1
            action['button_type'] = 0
        elif action_type_str == "PRESS_BACK":
            action['action_type'] = 1
            action['button_type'] = 1
        elif action_type_str == "PRESS_ENTER":
            action['action_type'] = 1
            action['button_type'] = 2
        elif action_type_str == "DUAL_POINT":
            action['action_type'] = 2

        touch_point_str = components[1].split(": ")[1].replace("\"", "")
        touch_point = ast.literal_eval(touch_point_str)
        action['touch_point'] = torch.tensor(touch_point)

        lift_point_str = components[2].split(": ")[1].replace("\"", "")
        lift_point = ast.literal_eval(lift_point_str)
        if lift_point != [0.0, 0.0]:
            action['action_type'] = 3
            action['scroll_from'] = torch.tensor(touch_point)
            action['scroll_to'] = torch.tensor(lift_point)

        return action

    def process_action_tensor2str(self, action:dict[str:torch.Tensor|tuple[torch.Tensor, torch.Tensor]], goal:str) -> str:
        raw_action = "Action Decision: "
        # Handle action type
        raw_action += "\"action_type\": "
        action_type = action['action_type']
        if action_type == 0: # typing
            raw_action += "\"TYPE\""
        elif action_type == 1: # bottom_button
            button_type = action['button_type']
            if button_type == 0:
                raw_action += "\"PRESS_HOME\""
            elif button_type == 1:
                raw_action += "\"PRESS_BACK\""
            elif button_type == 2:
                raw_action += "\"PRESS_ENTER\""
        elif action_type == 2 or action_type == 3: # touch or scroll
            raw_action += "\"DUAL_POINT\""
        # handle touch point
        raw_action += ", \"touch_point\": "
        touch_point = "\"[0.0, 0.0]\""
        if action_type == 2: # touch
            touch_point = f"\"{str(action['touch_point'])}\""
        elif action_type == 3: # scroll
            touch_point = f"\"{str(action['scroll_from'])}\""
        raw_action += touch_point
        # handle lift point
        raw_action += ", \"lift_point\": "
        lift_point = "\"[0.0, 0.0]\""
        if action_type == 3: # scroll
            lift_point = f"\"{str(action['scroll_to'])}\""
        raw_action += lift_point
        # handle typed text
        raw_action += ", \"typed_text\": "
        typed_text = "\"\""
        if action_type == 0: # type
            typed_text = self.get_typed_text(goal, self.step_goals[action['typing_type']])
        raw_action += typed_text

        return raw_action