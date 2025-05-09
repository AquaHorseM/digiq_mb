import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models

import re
import ast

class StateEncoderVLM(nn.Module):
    def __init__(self):
        pass

    def forward(self, image:torch.Tensor) -> torch.Tensor:
        pass

class StateEncoderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(image)
        return x.view(x.size(0), -1) # 2048

class ActionEncoder(nn.Module):
    def __init__(self, backbone:str, cache_dir:str, device:str):
        self.device = device
        
        self.base_lm_type = AutoModel.from_pretrained(backbone, cache_dir=cache_dir).to(device)
        self.base_tokenizer_type = AutoTokenizer.from_pretrained(backbone, cache_dir=cache_dir)
        self.base_tokenizer_type.truncation_side = 'left'
        
        self.base_lm_text = AutoModel.from_pretrained(backbone, cache_dir=cache_dir).to(device)
        self.base_tokenizer_text = AutoTokenizer.from_pretrained(backbone, cache_dir=cache_dir)
        self.base_tokenizer_text.truncation_side = 'left'

    def parse_action_string(self, action_list:str|list[str]):
        if type(action_list) == str:
            action_list = [action_list]
        
        type_list = []
        text_list = []
        
        for action in action_list:
            if self.is_action_valid(action):
                parsed_action = self.parse_action(action)
                type_list.append(parsed_action['action_type'])
                text_list.append(parsed_action['typed_text'])
            else:
                type_list.append(action)
                text_list.append(action)
        return type_list, text_list

    def parse_action(self, action):
        # the given actions is a string look like 'Action Plan: ... ; Action Decision: "action_type": "...", "touch_point": "...", "lift_point": "...", "typed_text": "..."'
        fill = ["action_type", "touch_point", "lift_point", "typed_text"]
        action_dict = {}
        for key in fill:
            # extract the value of each key in the action (exmaple shown in template)
            pattern = f'"{key}": "(.*?)"'
            match = re.search(pattern, action)
            action_dict[key] = match.group(1) if match else None
        
        return action_dict
    
    def is_action_valid(self, raw_action):
        try:
            parsed_action = self.parse_action(raw_action)
            touch_point_ratio = ast.literal_eval(parsed_action['touch_point'])
            lift_point_ratio = ast.literal_eval(parsed_action['lift_point'])
            return True
        except:
            return False

    def forward(self, action:str) -> torch.Tensor:
        type_list, text_list = self.parse_action_string(action)
        action_type_ids = self.base_tokenizer_type(type_list, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        action_text_ids = self.base_tokenizer_text(text_list, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        action_type_states = self.base_lm_type(**action_type_ids).pooler_output
        action_text_states = self.base_lm_text(**action_text_ids).pooler_output

        return torch.cat([action_type_states, action_text_states])