import torch
from PIL import Image
import argparse
import re
import ast
import os
import concurrent.futures
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", type=str, default="", help="Path to input data")
parser.add_argument("--layovered_image_path", type=str, default="", help="Path to store q_image_paths")
parser.add_argument("--output_data_path", type=str, default="", help="Path to store data with the 'q_image_paths' field")
parser.add_argument("--click_icon_path", type=str, default="", help="Path to the click icon (provided in the repo)")
parser.add_argument("--num_slices", type=str, default="", help="Number of slices for parallel processing, should satisfy len(trajs) % num_slices == 0")
args = parser.parse_args()

# Load the dataset
trajs = torch.load(args.input_data_path, weights_only=False)
save_path = args.output_data_path
click_icon = Image.open(args.click_icon_path).convert("RGBA")
num_slices = int(args.num_slices)

assert len(trajs) % num_slices == 0, "Number of slices should divide the number of trajectories"

total_steps = sum(len(traj) for traj in trajs)

def parse_action(action):
    fill = ["action_type", "touch_point", "lift_point", "typed_text"]
    action_dict = {}
    for key in fill:
        pattern = f'"{key}": "(.*?)"'
        match = re.search(pattern, action)
        if match:
            try:
                if key in ["touch_point", "lift_point"]:
                    points = ast.literal_eval(match.group(1))
                    if isinstance(points, (list, tuple)) and len(points) == 2:
                        action_dict[key] = points
                    else:
                        return None
                else:
                    action_dict[key] = match.group(1)
            except (ValueError, SyntaxError):
                return None
        else:
            return None
    return action_dict

def process_image(step):
    image_path = step['image_path']

    action_list = step['action_list']
    parsed_action_list = []
    q_image_path_list = []
    for i, raw_action in enumerate(action_list):
        img = Image.open(image_path).convert("RGB")
        parsed_action = parse_action(raw_action)
        parsed_action_list.append(parsed_action)
        if parsed_action is None:
            print(f"Skipping invalid action for trajectory {step['traj_id']} step {step['step_id']} action {i}: {raw_action}")
        else:
            touch_point = (int(parsed_action['touch_point'][0] * img.height), int(parsed_action['touch_point'][1] * img.width))
            lift_point = (int(parsed_action['lift_point'][0] * img.height), int(parsed_action['lift_point'][1] * img.width))

            if (touch_point[0] == lift_point[0]) and (touch_point[1] == lift_point[1]) and parsed_action['action_type'] == "DUAL_POINT":
                icon_size = (250, 250)
                resized_icon = click_icon.resize(icon_size, Image.Resampling.LANCZOS)
                icon_x = touch_point[1] - icon_size[0] // 2
                icon_y = touch_point[0] - icon_size[1] // 2
                img.paste(resized_icon, (icon_x, icon_y), resized_icon)
        q_image_path = os.path.join(args.layovered_image_path, f"temp_{step['traj_id']}_{step['step_id']}_{i}.png")
        q_image_path_list.append(q_image_path)
        img = img.resize((int(img.width * 0.25), int(img.height * 0.25)))
        img.save(q_image_path)

    step['parsed_action_list'] = parsed_action_list
    step['q_image_path_list'] = q_image_path_list
    step['parsed_action'] = parsed_action_list[0]
    step['q_image_path'] = q_image_path_list[0]
    return step

def process_slice(slice_trajs, slice_index):
    processed_slice = []
    for i in tqdm(range(len(slice_trajs)), disable=(slice_index != 0)):
        for j in range(len(slice_trajs[i])):
            slice_trajs[i][j] = process_image(slice_trajs[i][j])
        processed_slice.append(slice_trajs[i])
    return processed_slice

# Split the trajs into slices
slice_interval = len(trajs) // num_slices
trajs_slices = [trajs[i * slice_interval:(i + 1) * slice_interval] for i in range(num_slices)]

# Process each slice in parallel using a thread pool
with concurrent.futures.ProcessPoolExecutor(max_workers=num_slices) as executor:
    futures = [executor.submit(process_slice, trajs_slices[i], i) for i in range(num_slices)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

# Merge the results in the original order
merged_trajs = []
for result in results:
    merged_trajs.extend(result)

# Save the processed trajectories
torch.save(merged_trajs, save_path)

print("All websites filtered and saved to", save_path)
