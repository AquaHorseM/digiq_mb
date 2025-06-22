import torch
from transformers import AutoTokenizer
from digiq.models.critic import VLMDoubleCritic
from digiq.models.value_model import Value_Model
from digiq.models.encoder import ActionEncoder
from digiq.models.transition_model import TransformerTransition
from .model import T5ForMultimodalGeneration
import numpy as np
from gradio_client import Client, file
import ast
import signal
import time
import copy
import re
from PIL import Image, ImageDraw
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, wait_chain

# Suppress httpx info logs
logging.getLogger('httpx').setLevel(logging.WARNING)

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class AutoUIAgent(torch.nn.Module):
    def __init__(self, device, accelerator, click_icon_path, policy_lm = "gpt2", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, 
                eos_str = None, learn_metric="classification", advantage_estimation="bellman", 
                api_endpoints = [], 
                transition_path = None, value_path = None):
        super(AutoUIAgent, self).__init__()
        if use_bfloat16:
            self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir,
                                                              torch_dtype = torch.bfloat16).to(device)
            self.init_model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir,
                                                              torch_dtype = torch.bfloat16).to(device)
        else:
            # pi_theta
            self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
            # pi_b
            self.init_model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Using LoRA")
            self.model.print_trainable_parameters()
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.advantage_estimation = advantage_estimation
        if learn_metric == "classification":
            out_dim = 2
        elif learn_metric == "regression":
            out_dim = 1
        else:
            raise ValueError(f"Unknown metric {learn_metric}")
        
        in_dim = 768

        self.critic = VLMDoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = in_dim, out_dim = out_dim)
        if advantage_estimation == "bellman":
            self.target_critic = VLMDoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = in_dim, out_dim = out_dim)  
        
        self.transition = TransformerTransition(
            state_dim=3584, action_dim=1536, device=device
        )
        self.transition.load_state_dict(torch.load(transition_path, map_location=device))
        self.action_encoder = ActionEncoder(backbone='roberta-base', cache_dir=None, device=device)
        self.value = Value_Model(3584, 768, 1024, 0, 8, critic_lm, None, device)
        self.value.load_state_dict(torch.load(value_path, map_location=device))

        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
        self.click_icon = Image.open(click_icon_path).convert("RGBA")
        if api_endpoints:
            self.clients = [Client(endpoint) for endpoint in api_endpoints]
        else:
            self.clients = []

    def prepare(self):
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)
        if self.advantage_estimation == "bellman":
            self.target_critic = self.accelerator.prepare(self.target_critic)

    def get_pi_action_guarantee_valid(self, observation, image_features, pi_version="pi_b", max_try=3, original_action=None, allow_going_home=True):
        pi_b_action = self.get_pi_action(observation, image_features, pi_version)
        actions_is_valid = [self.is_action_valid(a, allow_going_home) for a in pi_b_action]
        try_count = 0
        # resample invalid actions
        while not all(actions_is_valid) and try_count < max_try:
            for i in range(len(pi_b_action)):
                if not actions_is_valid[i]:
                    print(f"action invalid, resampling - raw action: {pi_b_action[i]}")
                    pi_b_action[i] = self.get_pi_action([observation[i]], image_features[i].unsqueeze(0), pi_version)[0]
                    print(f"resampled action: {pi_b_action[i]}")
                    actions_is_valid[i] = self.is_action_valid(pi_b_action[i], allow_going_home)
            try_count += 1
        
        if not all(actions_is_valid):
            print(f"Failed to get valid actions after {max_try} tries. Returning original action: {original_action}")
            return original_action
        return pi_b_action

    def select(self, goals, s_reps, actions, sample_per_input):
        s_reps = s_reps.to(self.device)
        ret_actions = []
        for i in range(len(actions) // sample_per_input):
            batch_action = self.action_encoder(actions[i*sample_per_input:(i+1)*sample_per_input])
            self.transition.eval()
            with torch.no_grad():
                pred_next_state = self.transition.forward(torch.stack([s_reps[i]] * sample_per_input, dim=0), batch_action)
            values = self.value(pred_next_state, goals)
            selected = values.argmax()
            ret_actions.append(actions[i*sample_per_input + selected])
        return ret_actions

    def get_goal(self, observation):
        if isinstance(observation, str):
            observation = [observation]
        goals = []
        for obs in observation:
            goal_match  = re.search(r'Goal: (.*?)</s>', obs)
            if goal_match:
                goal = goal_match.group(1)
            else:
                goal = ""
            goal = self.value.goal_encoder(goal).to(dtype=torch.float32, device=self.device)  # [B, goal_dim]
            goals.append(goal)
        goals = torch.stack(goals, dim=0)            # [B, goal_dim]
        return goals

    def get_pi_action(self, observation, image_features, s_reps, pi_version="pi_b"):
        if pi_version == "pi_b":
            policy = self.init_model
        elif pi_version == "pi_theta":
            policy = self.accelerator.unwrap_model(self.model)
        else:
            raise ValueError(f"Unknown pi_version {pi_version}")
        image_features = image_features[..., -1408:]
        for _ in range(3):
            try:
                with timeout(seconds=60):
                    with torch.no_grad():
                        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
                        image_features = image_features.to(self.device)
                        outputs = policy.generate(**obs_ids, image_ids = image_features,
                                                    max_new_tokens=self.max_new_tokens, 
                                                    do_sample=self.do_sample, temperature = self.temperature, num_return_sequences=2,
                                                    pad_token_id = self.tokenizer.eos_token_id).cpu()
                    break
            except TimeoutError:
                print("Timeout while accessing actions")
                continue
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        raw_action = self.select(self.get_goal(observation), s_reps, raw_action, 2)
        for _ in range(3):
            raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
        if self.eos_str is not None:
            return [raw_a.split(self.eos_str)[0] for raw_a in raw_action]
        else:
            return raw_action
        
    def parse_action(self, action):
        # the given actions is a string look like 'Action Plan: [...] ; Action Decision: "action_type": "...", "touch_point": "...", "lift_point": "...", "typed_text": "..."'
        fill = ["Action Plan", "action_type", "touch_point", "lift_point", "typed_text"]
        action_dict = {}
        for key in fill:
            if key == "Action Plan":
                # extract the value of the action plan
                pattern = r'{key}: \[(.*?)\]'
                match = re.search(pattern, action)
                action_dict[key] = match.group(1) if match else None
            else:
                # extract the value of each key in the action (exmaple shown in template)
                pattern = f'"{key}": "(.*?)"'
                match = re.search(pattern, action)
                action_dict[key] = match.group(1) if match else None
                
        if action_dict["Action Plan"]:
            # split the action plan into a list of actions
            action_dict["Action Plan"] = action_dict["Action Plan"].split(",")
        
        return action_dict
    
    def is_action_valid(self, raw_action, allow_going_home=True):
        try:
            if allow_going_home:
                allowed_actions = ["DUAL_POINT", "TYPE", "PRESS_BACK", "PRESS_ENTER", "STATUS_TASK_COMPLETE", "PRESS_HOME"]
            else:
                allowed_actions = ["DUAL_POINT", "TYPE", "PRESS_BACK", "PRESS_ENTER", "STATUS_TASK_COMPLETE"]
            parsed_action = self.parse_action(raw_action)
            if not parsed_action['Action Plan']:
                return False
            if type(parsed_action['typed_text']) != str:
                return False
            # if len(parsed_action['typed_text']) > 30:
            #     print(f"Typed text too long: {parsed_action['typed_text']}")
            #     return False
            if len(parsed_action['Action Plan']) > 8:
                print(f"Action Plan too long: {parsed_action['Action Plan']}")
                return False
            for action in parsed_action['Action Plan']:
                if action not in allowed_actions:
                    return False
            
            touch_point_ratio = ast.literal_eval(parsed_action['touch_point'])
            lift_point_ratio = ast.literal_eval(parsed_action['lift_point'])
            return True
        except:
            return False
    
    def add_cursor(self, image_path, raw_action):
        image_path = image_path
        # the raw action is guaranteed to be valid by upstream code
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            # for each step, first label the action onto the image, then explain the action by calling the llava API
            parsed_action = self.parse_action(raw_action)
            touch_point_ratio = ast.literal_eval(parsed_action['touch_point'])
            lift_point_ratio = ast.literal_eval(parsed_action['lift_point'])
            touch_point = (int(touch_point_ratio[0] * img.height), int(touch_point_ratio[1] * img.width))
            lift_point = (int(lift_point_ratio[0] * img.height), int(lift_point_ratio[1] * img.width))
            if (touch_point[0] == lift_point[0]) and (touch_point[1] == lift_point[1]) and parsed_action['action_type'] == "DUAL_POINT":
                x = touch_point[1]
                y = touch_point[0]
                icon_size = (250, 250)  # Adjust this size as needed
                resized_icon = self.click_icon.resize(icon_size, Image.Resampling.LANCZOS)
                icon_x = x - icon_size[0] // 2
                icon_y = y - icon_size[1] // 2
                img.paste(resized_icon, (icon_x, icon_y), resized_icon)

            img = img.resize((int(img.width * 0.25), int(img.height * 0.25)))
            walltime = time.time()
            # the path should be changed if the current dir is too small; but this function will never be called so it's okay to keep it as is for now
            q_image_path = f"{walltime}.png"
            img.save(q_image_path)
            
        except Exception as e:
            print(f"Error at step: {e}, Action: {raw_action}")
            parsed_action = None
            q_image_path = image_path
        return parsed_action, q_image_path
    
    @retry(stop=stop_after_attempt(3), wait=wait_chain(*[wait_fixed(1), wait_fixed(2), wait_fixed(4)]))
    def attempt_predict(self, client, q_image_path, query):
        return client.predict(file(q_image_path), query, api_name="/predict")
    
    def make_prompt(self, action_dict):
        command_string = "Respond only 'Yes' or 'No' (without period / quotation marks) and don't respond anything else."
        # command_string = ""
        if action_dict is not None:
            if (action_dict['action_type'] == "DUAL_POINT") and (action_dict['touch_point'] == action_dict['lift_point']):
                prompt = f"""You're given a user interface. There is a cursor in the screen. The touch point is located at {action_dict['touch_point']} Is this cursor Clicking on any interactive elements?""" + command_string
            elif action_dict['action_type'] == "TYPE":
                prompt = f"""You're given a user interface. If a user now Types {action_dict['typed_text']}, will this Type action effectively input the text into somewhere on the Screenshot?""" + command_string
            elif action_dict['action_type'] == "PRESS_HOME":
                prompt = """You're given a user interface. If a user now Presses the <HOME> button, will this action effectively navigate the user to the Home screen?""" + command_string
            elif action_dict['action_type'] == "PRESS_BACK":
                prompt = """You're given a user interface. If a user now Presses the <BACK> button, will this action effectively navigate the user to the previous screen?""" + command_string
            elif action_dict['action_type'] == "PRESS_ENTER":
                prompt = """You're given a user interface. If a user now Presses the <ENTER> button, will this action effectively submit the form?""" + command_string
            else:
                prompt = """You're given a user interface. Is this action effective?""" + command_string
        else:
            prompt = """You're given a user interface. Is this action effective?""" + command_string
            
        return prompt

    def get_q_reps(self, first_image_path, pi_action, client):
        action_dict, q_image_path = self.add_cursor(first_image_path, pi_action)
        print("the q image path is: ", q_image_path)

        if action_dict is not None:
            query = self.make_prompt(action_dict)

            try:
                q_prediction = self.attempt_predict(client, q_image_path, query)
                q_rep_out = np.array(q_prediction[0]['data'][0], dtype=np.float32)
            except Exception as e:
                print(f"Failed after retries: {e}. Returning zero vectors.")
                q_rep_out = np.zeros(4096, dtype=np.float32)
        else:
            print("Action dict is None. Returning zero vectors.")
            q_rep_out = np.zeros(4096, dtype=np.float32)

        return q_rep_out

    def get_pi_theta_log_prob(self, observation, image_features, action):
        image_features = image_features[...,-1408:]
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        outputs = self.model(input_ids = obs_ids["input_ids"],
                            image_ids = image_features,
                            attention_mask = obs_ids["attention_mask"],
                            labels = action_ids["input_ids"])
        
        # # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        # input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        # attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
        #                         dim = 1)
        # outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        # values = None
        # if isinstance(outputs, Tuple):
        #     values, outputs = outputs

        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs,\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        return torch.log(selected_prediction_probs)*action_ids["attention_mask"]

    def get_pi_theta_log_prob_distr(self, observation, image_features, action):
        """
        Computes the log probability distribution over the entire vocabulary for each token in the `action` sequence.

        Args:
            observation (List[str]): List of observation texts.
            image_features (torch.Tensor): Tensor of image features.
            action (List[str]): List of action texts.

        Returns:
            torch.Tensor: Log probabilities with shape (batch_size, seq_length, vocab_size).
        """
        # **1. Image Feature Processing**
        image_features = image_features[..., -1408:]
        
        # **2. Template Application (Optional)**
        if self.template:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        outputs = self.model(input_ids = obs_ids["input_ids"],
                            image_ids = image_features,
                            attention_mask = obs_ids["attention_mask"],
                            labels = action_ids["input_ids"])
        prediction_probs = self.softmax(outputs.logits)
        log_probs = torch.log(prediction_probs + 1e-12)  # Prevent log(0)
        masked_log_probs = log_probs * action_ids["attention_mask"].unsqueeze(-1)
        
        return masked_log_probs


    
    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
    def copy_pi_theta_to_pi_b(self):
        for pi_theta_param, pi_b_param in zip(self.model.parameters(), self.init_model.parameters()):
            pi_b_param.data.copy_(pi_theta_param.data)
