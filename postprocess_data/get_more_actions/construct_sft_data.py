import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import json
import ast
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", type=str, default="", help="Path to input trajectories")
parser.add_argument("--output_contrastive_data_path", type=str, default="", help="Path to store the contrastive learning data")
args = parser.parse_args()

trajs = torch.load(args.input_data_path, weights_only=False)

# collect datapoints with format:
# input: (annotated Q image, prompt)
# output: (answer)
datapoints = []

def judge(image_path, next_image_path):
    image_path = image_path.replace("images", "smaller_images")
    next_image_path = next_image_path.replace("images", "smaller_images")
    # thresholding method: if the difference between the two images is greater than 0.5, then the answer is "Yes", else "No"
    with Image.open(image_path) as img1_src, Image.open(next_image_path) as img2_src:   
        img1 = np.array(img1_src)
        img2 = np.array(img2_src)

    diff = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    # print(diff)
    threshold = 30
    if diff > threshold:
        return "Yes"
    else:
        return "No"

def parse_action(action):
    # the given actions is a string look like 'Action Plan: ... ; Action Decision: "action_type": "...", "touch_point": "...", "lift_point": "...", "typed_text": "..."'
    fill = ["action_type", "touch_point", "lift_point", "typed_text"]
    action_dict = {}
    for key in fill:
        # extract the value of each key in the action (exmaple shown in template)
        pattern = f'"{key}": "(.*?)"'
        match = re.search(pattern, action)
        action_dict[key] = match.group(1) if match else None
    
    return action_dict

def is_action_valid(raw_action):
    try:
        parsed_action = parse_action(raw_action)
        touch_point_ratio = ast.literal_eval(parsed_action['touch_point'])
        lift_point_ratio = ast.literal_eval(parsed_action['lift_point'])
        return True
    except:
        return False

def make_prompt(action):
    command_prompt = "Respond only 'Yes' or 'No' (without period / quotation marks) and don't respond anything else."
    # command_prompt = ""
    if action is not None:
        if action['action_type'] == "DUAL_POINT":
            prompt = f"<image>\nYou're given a user interface. There is a cursor in the screen. The touch point is located at {action['touch_point']} Is this cursor Clicking on any interactive elements?"
        elif action['action_type'] == "TYPE":
            prompt = f"<image>\nYou're given a user interface. If a user now Types {action['typed_text']}, will this Type action effectively input the text into somewhere on the Screenshot?"""
        elif action['action_type'] == "PRESS_HOME":
            prompt = "<image>\nYou're given a user interface. If a user now Presses the <HOME> button, will this action effectively navigate the user to the Home screen?"
        elif action['action_type'] == "PRESS_BACK":
            prompt = "<image>\nYou're given a user interface. If a user now Presses the <BACK> button, will this action effectively navigate the user to the previous screen?"
        elif action['action_type'] == "PRESS_ENTER":
            prompt = "<image>\nYou're given a user interface. If a user now Presses the <ENTER> button, will this action effectively submit the form?"
        else:
            prompt = "<image>\nYou're given a user interface. Is this action effective?"
    else:
        prompt = "<image>\nYou're given a user interface. Is this action effective?"
    return prompt + "\n" + command_prompt

for traj_id, traj in tqdm(enumerate(trajs)):
    for step_id, step in enumerate(traj):
        datapoint = {
            "id": f"{step['traj_id']}-{step['step_id']}",
            "image": step["q_image_path"].split("/")[-1],
            "conversations": [
                {"from": "human", "value": make_prompt(step['parsed_action'])},
                {"from": "gpt", "value": judge(step["image_path"], step["next_image_path"])}
            ]
        }
        datapoints.append(datapoint)

# reserve all training data in low-resource setting
train_data = datapoints

# Save the train and validation datasets
train_save_path = args.output_contrastive_data_path

with open(train_save_path, "w") as f:
    json.dump(train_data, f)

print(f"Training data saved to {train_save_path}")
