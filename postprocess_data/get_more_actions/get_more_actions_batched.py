import re
import ast
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from digiq.models.model import T5ForMultimodalGeneration
from accelerate import Accelerator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", type=str, default="", help="Path to collcted trajectories")
parser.add_argument("--output_data_path", type=str, default="", help="Path to store formatted trajectories")
parser.add_argument("--policy_lm", type=str, default="", help="Path to policy language model that collects more actions")
parser.add_argument("--num_actions", type=str, default="", help="Total number of actions to collect (the original actions will be reserved, so putting num_actions=64 will collect 63 actions)")
parser.add_argument("--num_gpus", type=str, default="", help="Number of GPUs to use for parallel processing")
args = parser.parse_args()

# Define a function to process a slice of trajectories
def process_slice(trajs_slice, gpu_id):
    policy_lm = args.policy_lm
    accelerator = Accelerator()
    device = f"cuda:{gpu_id}"
    policy = T5ForMultimodalGeneration.from_pretrained(policy_lm, device_map=device, cache_dir="'/home/ubuntu/.cache'")
    tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir="'/home/ubuntu/.cache'")
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    num_actions_to_get = num_actions - 1
    for i in tqdm(range(len(trajs_slice)), disable=gpu_id != 0):
        for j in range(len(trajs_slice[i])):
            action_list = [trajs_slice[i][j]['action']]
            repeated_observation = [trajs_slice[i][j]['observation']] * num_actions_to_get
            repeated_image_features = torch.from_numpy(trajs_slice[i][j]['image_features']).repeat(num_actions_to_get, 1)
            all_actions = get_pi_action_guarantee_valid(
                repeated_observation,
                repeated_image_features,
                policy=policy,
                tokenizer=tokenizer,
                device=device
            )
            action_list.extend(all_actions)
            trajs_slice[i][j]['action_list'] = action_list
    return trajs_slice

# Modified function to get actions with policy and tokenizer arguments
def get_pi_action_guarantee_valid(observation, image_features, policy, tokenizer, device, max_try=3):
    pi_0_action = get_pi_0_action(observation, image_features, policy, tokenizer, device)
    actions_is_valid = [is_action_valid(raw_action=a) for a in pi_0_action]
    try_count = 0
    # resample invalid actions
    while not all(actions_is_valid) and try_count < max_try:
        num_invalid = sum([not a for a in actions_is_valid])
        print(f"resampling {num_invalid} invalid actions, first invalid action: {pi_0_action[actions_is_valid.index(False)]}")
        for i in range(len(pi_0_action)):
            if not actions_is_valid[i]:
                pi_0_action[i] = get_pi_0_action([observation[i]], image_features[i].unsqueeze(0), policy, tokenizer, device)[0]
                actions_is_valid[i] = is_action_valid(raw_action=pi_0_action[i])
        try_count += 1
        
    if not all(actions_is_valid):
        print(f"Failed to get valid actions after {max_try} tries. Returning the last action.")
    return pi_0_action

# Modified function to use policy and tokenizer
def get_pi_0_action(observation, image_features, policy, tokenizer, device):
    image_features = image_features[..., -1408:]
    with torch.no_grad():
        obs_ids = tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(device)
        image_features = image_features.to(device)
        outputs = policy.generate(**obs_ids, 
                                    image_ids = image_features,
                                    max_new_tokens=128, 
                                    do_sample=True, 
                                    temperature=1.2, # larger temperature -> more randomness
                                    pad_token_id=tokenizer.eos_token_id
                                    ).cpu()
    raw_action = tokenizer.batch_decode(outputs, skip_special_tokens  = True)
    for _ in range(3):
        raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
    return raw_action

def parse_action(action):
    # the given actions is a string look like 'Action Plan: [...] ; Action Decision: "action_type": "...", "touch_point": "...", "lift_point": "...", "typed_text": "..."'
    fill = ["Action Plan", "action_type", "touch_point", "lift_point", "typed_text"]
    action_dict = {}
    for key in fill:
        if key == "Action Plan":
            # extract the value of the action plan
            pattern = f'{key}: \[(.*?)\]'
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

def is_action_valid(raw_action, allow_going_home=True):
    try:
        if allow_going_home:
            allowed_actions = ["DUAL_POINT", "TYPE", "PRESS_BACK", "PRESS_ENTER", "STATUS_TASK_COMPLETE", "PRESS_HOME"]
        else:
            allowed_actions = ["DUAL_POINT", "TYPE", "PRESS_BACK", "PRESS_ENTER", "STATUS_TASK_COMPLETE"]
        parsed_action = parse_action(raw_action)
        if not parsed_action['Action Plan']:
            return False
        if type(parsed_action['typed_text']) != str:
            return False
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

offline_data_path_1action = args.input_data_path
num_actions = int(args.num_actions)
num_gpus = int(args.num_gpus)
trajs_1action = torch.load(offline_data_path_1action, weights_only=False)

# Split trajectories into slices
slice_size = len(trajs_1action) // num_gpus
trajs_slices = [trajs_1action[i * slice_size:(i + 1) * slice_size] for i in range(num_gpus)]
if len(trajs_1action) % num_gpus != 0:
    trajs_slices[-1] += trajs_1action[num_gpus * slice_size:]

# Process each slice in parallel using a process pool
with ProcessPoolExecutor(max_workers=num_gpus) as executor:
    futures = [executor.submit(process_slice, trajs_slices[i], i) for i in range(num_gpus)]
    results = [f.result() for f in as_completed(futures)]

# Merge the results in the original order
merged_trajs = []
for i in range(num_gpus):
    merged_trajs.extend(results[i])

reordered_trajs = []
traj_id_to_merged_traj = {merged_traj[0]['traj_id']: merged_traj for merged_traj in merged_trajs}
for i in range(len(trajs_1action)):
    reordered_trajs.append(traj_id_to_merged_traj[i]) 

# Save the results
torch.save(reordered_trajs, args.output_data_path)
