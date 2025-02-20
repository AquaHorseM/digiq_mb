import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import ast
import re

class VLMDoubleCritic(torch.nn.Module):
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        """
        VLM critic using image features
        """
        super(VLMDoubleCritic, self).__init__()
        self.device = device
        self.accelerator = accelerator
        
        self.base_lm_task = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer_task = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer_task.truncation_side = 'left'
        
        self.base_lm_type = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer_type = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer_type.truncation_side = 'left'
        
        self.base_lm_text = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer_text = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer_text.truncation_side = 'left'


        v_rep_dim = 1408 * 2
        q_rep_dim = 4096 + 1408 * 2
        v_in_dim = in_dim * 3
        q_in_dim = in_dim * 5
        
        # for q
        self.q_critic1 = nn.Sequential(
                                    nn.Linear(q_in_dim+q_rep_dim, q_in_dim+q_rep_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim+q_rep_dim, q_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim, q_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim, q_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim, out_dim)).to(device)
        self.q_critic2 = nn.Sequential(
                                    nn.Linear(q_in_dim+q_rep_dim,  q_in_dim+q_rep_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim+q_rep_dim, q_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim, q_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim, q_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(q_in_dim, out_dim)).to(device)
        
        # for v
        self.v_critic1 = nn.Sequential(
                                    nn.Linear(v_in_dim+v_rep_dim, v_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(v_in_dim, v_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(v_in_dim, v_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(v_in_dim, out_dim)).to(device)
        self.v_critic2 = nn.Sequential(
                                    nn.Linear(v_in_dim+v_rep_dim, v_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(v_in_dim, v_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(v_in_dim, v_in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(v_in_dim, out_dim)).to(device)
        
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
            # print("action invalid, raw action: ", raw_action)
            return False
        
    def parse_obs(self, observation):
        if type(observation) == str:
            observation = [observation]
        
        # obs example: Previous Actions: Goal: Go to newegg.com</s>
        previous_actions = []
        goals = []
        for obs in observation:
            previous_action_match = re.search(r'Previous Actions: (.*?)Goal:', obs)
            goal_match = re.search(r'Goal: (.*?)</s>', obs)
            
            # Map None to an empty string if no match is found
            previous_actions.append(previous_action_match.group(1) if previous_action_match else "")
            goals.append(goal_match.group(1) if goal_match else "")
            
        return previous_actions, goals
    
    def parse_action_string(self, action_string_list):
        if type(action_string_list) == str:
            action_string_list = [action_string_list]
        
        type_content_list = []
        text_content_list = []
        for action_string in action_string_list:
            if self.is_action_valid(action_string):
                # print("action valid")
                parsed_action = self.parse_action(action_string)
                type_content_list.append(parsed_action['action_type'])
                text_content_list.append(parsed_action['typed_text'])
            else:
                type_content_list.append(action_string)
                text_content_list.append(action_string)
        return type_content_list, text_content_list
        
    def forward(self, observation, image_features, action, q_rep_out, detach_model=False):
        prev_actions, goal = self.parse_obs(observation)
        goal_ids = self.base_tokenizer_task(goal, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        
        prev_type_content, prev_text_content = self.parse_action_string(prev_actions)
        prev_action_type_ids = self.base_tokenizer_type(prev_type_content, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        prev_action_text_ids = self.base_tokenizer_text(prev_text_content, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        
        type_content, text_content = self.parse_action_string(action)
        action_type_ids = self.base_tokenizer_type(type_content, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        action_text_ids = self.base_tokenizer_text(text_content, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
            
        if detach_model:
            with torch.no_grad():
                prev_action_type_states = self.base_lm_type(**prev_action_type_ids).pooler_output
                prev_action_text_states = self.base_lm_text(**prev_action_text_ids).pooler_output
                goal_states = self.base_lm_task(**goal_ids).pooler_output
                action_type_states = self.base_lm_type(**action_type_ids).pooler_output
                action_text_states = self.base_lm_text(**action_text_ids).pooler_output
        else:
            prev_action_type_states = self.base_lm_type(**prev_action_type_ids).pooler_output
            prev_action_text_states = self.base_lm_text(**prev_action_text_ids).pooler_output
            goal_states = self.base_lm_task(**goal_ids).pooler_output
            action_type_states = self.base_lm_type(**action_type_ids).pooler_output
            action_text_states = self.base_lm_text(**action_text_ids).pooler_output
            
        q_states = torch.cat([prev_action_type_states, prev_action_text_states, goal_states, image_features, action_type_states, action_text_states, q_rep_out], dim = 1)
        v_states = torch.cat([prev_action_type_states, prev_action_text_states, goal_states, image_features], dim = 1)
        # q_states = torch.cat([prev_action_type_states, prev_action_text_states, goal_states, image_features, action_type_states, action_text_states], dim = 1)
        # v_states = torch.cat([prev_action_type_states, prev_action_text_states, goal_states, image_features], dim = 1)
        
        return self.q_critic1(q_states), self.q_critic2(q_states), self.v_critic1(v_states), self.v_critic2(v_states)
    