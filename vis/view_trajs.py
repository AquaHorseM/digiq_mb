import gradio as gr
from PIL import Image, ImageFont, ImageDraw
import torch
import copy
import transformers
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
import re
import numpy as np
import argparse

transformers.logging.set_verbosity_error()

from digiq.models import AutoUIAgent
from digiq.algorithms.digiq import DigiQTrainer

def framestack(orig_trajs):
    trajs = copy.deepcopy(orig_trajs)
    
    for i in range(len(trajs)):
        for j in range(len(trajs[i])):
            if j == 0:
                trajs[i][j]["image_features"] = np.concatenate([orig_trajs[i][j]["image_features"], orig_trajs[i][j]["image_features"]], axis=-1)
            else:
                trajs[i][j]["image_features"] = np.concatenate([orig_trajs[i][j-1]["image_features"], orig_trajs[i][j]["image_features"]], axis=-1)
            trajs[i][j]["next_image_features"] = np.concatenate([orig_trajs[i][j]["image_features"], orig_trajs[i][j]["next_image_features"]], axis=-1)
    return trajs

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--agent_path", type=str, default="", help="Path to agent model")
parser.add_argument("--data_path", type=str, default="", help="Path to processed trajectories")
parser.add_argument("--image_path", type=str, default="", help="Path to the images/ directory")
parser.add_argument("--hf_cache_path", type=str, default="", help="Path to the huggingface cache file")
parser.add_argument("--autoui_path", type=str, default="", help="Path to the AutoUI model")
parser.add_argument("--click_icon_path", type=str, default="", help="Path to the click icon image")
parser.add_argument("--vis_state_values", type=str, default="True", help="Whether to visualize state values")
parser.add_argument("--advantage_estimation", type=str, default="bellman", help="Whether to visualize the advantages")
parser.add_argument("--parse_action", type=str, default="True", help="Whether to parse the action")
parser.add_argument("--learn_metric", type=str, default="regression", help="Whether to learn a classification or regression metric")
parser.add_argument("--vis_last_step_nv", type=str, default="False", help="Whether to visualize the last step's next state value")
args = parser.parse_args()

agent_path = args.agent_path
path = args.data_path

# Initialize accelerators
accelerators = {}
agents = {}
trainers = {}
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs], cpu=False)

agent = AutoUIAgent(device=accelerator.device, accelerator=accelerator, 
                    temperature=1.0, do_sample=True, 
                    policy_lm=args.autoui_path, critic_lm="roberta-base",
                    cache_dir=args.hf_cache_path, max_new_tokens=256, learn_metric=args.learn_metric,
                    advantage_estimation=args.advantage_estimation, click_icon_path=args.click_icon_path)

if args.vis_state_values == "True":
    tokenizer = agent.tokenizer
    trainer = DigiQTrainer(agent=agent,
                            accelerator=accelerator,
                            tokenizer=tokenizer,
                            critic_lr=1e-4,
                            lm_lr=1e-4,
                            gamma=0.6,
                            tau=0.1,
                            epochs=50,
                            actor_epochs=3,
                            grad_accum_steps=32,
                            max_grad_norm=1.0)
    agent.prepare()
    agent.accelerator.load_state(agent_path)

data = torch.load(path, weights_only=False)
data = framestack(data)

# print(data[0][0]['observation'])
# print(data[0][0]['image_features'].shape)

walmart_trajs = [traj for traj in data if "walmart" in traj[0]['observation']]
print("List of Go to walmart.com, search for 'logitech g933' trajectories:", [i for i, traj in enumerate(data) if "Go to walmart.com, search for 'logitech g933'" in traj[0]['observation'] and traj[-1]['reward'] > 0])

success_trajs = [traj for traj in data if traj[-1]['reward'] > 0]
# print(f"Number of successful trajectories: {len(success_trajs)}")
print("List of successful trajectories IDs:", [i for i, traj in enumerate(data) if traj[-1]['reward'] > 0])


def parse_action(action):
    fill = ["action_type", "touch_point", "lift_point", "typed_text"]
    action_dict = {}
    for key in fill:
        pattern = f'"{key}": "(.*?)"'
        match = re.search(pattern, action)
        action_dict[key] = match.group(1) if match else None
    return action_dict

def display_images(trajectory_id):
    trajectory_id = int(trajectory_id)
    if trajectory_id < 0 or trajectory_id >= len(data):
        return ["Invalid index. Please enter a valid index."] + [None] * len(agents)
    
    images = []
    task = data[trajectory_id][0]['task']
    
    list_qv_differences = []
    for step_id in range(len(data[trajectory_id])):
        step_data = data[trajectory_id][step_id]
        action_id = 0
        pi_action = step_data['action_list'].split("<split>")[action_id]
        step_data['image_path'] = step_data['image_path'].replace("<path_to_images>", args.image_path)
        # print(step_data['image_path'])
        img_base = Image.open(step_data['image_path']).convert("RGB")
        img_copy = img_base.copy()
        draw = ImageDraw.Draw(img_copy)
        # get q_rep_out by inferring llava on-the-fly
        if args.vis_state_values == "True":
            out_tensor = torch.from_numpy(step_data['q_rep_out_list'][action_id]).to(agent.device)
            
            out_np = out_tensor.cpu().detach().numpy().tolist()
            out_np = [round(x, 2) for x in out_np]
            q1, q2, v1, v2 = agent.critic(step_data["observation"],
                                            torch.from_numpy(step_data['image_features']).unsqueeze(0).to(agent.device),
                                            pi_action, 
                                            out_tensor.unsqueeze(0).to(agent.device))
            
            if args.advantage_estimation == "mc":
                q1, q2, v1, v2 = torch.nn.Softmax(dim=-1)(q1)[:, 1], torch.nn.Softmax(dim=-1)(q2)[:, 1], torch.nn.Softmax(dim=-1)(v1)[:, 1], torch.nn.Softmax(dim=-1)(v2)[:, 1]
            elif args.advantage_estimation == "bellman":
                q1, q2, v1, v2 = q1.flatten(), q2.flatten(), v1.flatten(), v2.flatten()

            v, q = torch.maximum(v1, v2).flatten(), torch.maximum(q1, q2).flatten()
            qv_difference = q - v
            qv_difference = round(qv_difference.cpu().detach().numpy().item(), 2)
            
            list_qv_differences.append(qv_difference)
            v, q = round(v.cpu().detach().numpy().item(), 2), round(q.cpu().detach().numpy().item(), 2)
            v1, v2, q1, q2 = round(v1.cpu().detach().numpy().item(), 2), round(v2.cpu().detach().numpy().item(), 2), round(q1.cpu().detach().numpy().item(), 2), round(q2.cpu().detach().numpy().item(), 2)
            draw.text((100, 500), f"V: {v}\nQ: {q}\nQ-V: {qv_difference}\nQrep:{out_np[:3]}", fill=(0, 0, 0), font=ImageFont.load_default(75))
            
        if args.parse_action == "True":
            action_dict = parse_action(pi_action)
            action = "\n".join([f"{key}: {value}" for key, value in action_dict.items()])
            if action_dict['action_type'] == 'DUAL_POINT':
                try:
                    import ast
                    touch_point_ratio = ast.literal_eval(action_dict['touch_point'])
                    lift_point_ratio = ast.literal_eval(action_dict['lift_point'])
                    touch_point = (int(touch_point_ratio[0] * img_copy.height), int(touch_point_ratio[1] * img_copy.width))
                    lift_point = (int(lift_point_ratio[0] * img_copy.height), int(lift_point_ratio[1] * img_copy.width))
                    if touch_point[0] == lift_point[0] and touch_point[1] == lift_point[1]:
                        x = touch_point[1]
                        y = touch_point[0]
                        r = 20
                        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 0, 0))
                    else:
                        touch_point_x = touch_point[1]
                        touch_point_y = touch_point[0]
                        lift_point_x = lift_point[1]
                        lift_point_y = lift_point[0]
                        draw.line([touch_point_x, touch_point_y, lift_point_x, lift_point_y], fill=(255, 0, 0), width=15)
                        r = 20
                        draw.ellipse([(lift_point_x - r, lift_point_y - r), (lift_point_x + r, lift_point_y + r)], fill=(255, 0, 0))
                except Exception as e:
                    print(e)

            draw.text((100, 1000), str(action), fill=(0, 0, 0), font=ImageFont.load_default(50))
            
            images.append(img_copy)

        if args.vis_last_step_nv == "True" and args.vis_state_values == "True" and step_id == len(data[trajectory_id]) - 1:
            next_img = Image.open(step_data['next_image_path']).convert("RGB")
            _, _, nv1, nv2 = agent.critic(step_data["next_observation"],
                                            torch.from_numpy(step_data['next_image_features']).unsqueeze(0).to(agent.device),
                                            pi_action, 
                                            torch.from_numpy(step_data['q_rep_out']).unsqueeze(0).to(agent.device))
            if args.advantage_estimation == "mc":
                nv1, nv2 = torch.nn.Softmax(dim=-1)(nv1)[:, 1], torch.nn.Softmax(dim=-1)(nv2)[:, 1]
            elif args.advantage_estimation == "bellman":
                nv1, nv2 = nv1.flatten(), nv2.flatten()
            nv = (nv1 + nv2) / 2
            nv = round(nv.cpu().detach().numpy().item(), 2)
            
            next_img_copy = next_img.copy()
            draw = ImageDraw.Draw(next_img_copy)
            images.append(next_img_copy)
                
    if images:
        max_images_per_row = 5
        num_rows = (len(images) + max_images_per_row - 1) // max_images_per_row
        
        row_images = []
        max_row_width = 0
        total_height = 0

        for row in range(num_rows):
            start_index = row * max_images_per_row
            end_index = min(start_index + max_images_per_row, len(images))
            
            row_images_subset = images[start_index:end_index]
            
            total_width = sum(img.width for img in row_images_subset)
            max_height = max(img.height for img in row_images_subset)
            
            row_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in row_images_subset:
                row_image.paste(img, (x_offset, 0))
                x_offset += img.width
            
            row_images.append(row_image)
            max_row_width = max(max_row_width, total_width)
            total_height += max_height
        
        combined_image = Image.new('RGB', (max_row_width, total_height))
        y_offset = 0
        for row_img in row_images:
            combined_image.paste(row_img, (0, y_offset))
            y_offset += row_img.height

        return [task] + [combined_image]

with gr.Blocks() as interface:
    
    gr.Markdown("## Trajectory Viewer (Trajectory-level, Single Action)")
    
    with gr.Row():
        with gr.Column():
            index_input = gr.Number(label="Index")
        with gr.Column():
            task_description_output = gr.Text(label="Task Description")
    
    image_outputs = []
    with gr.Row():
        with gr.Column():
            image_outputs.append(gr.Image(label=f"Trajectory Images Labeled with Q/V Values"))
    
    index_input.change(display_images, inputs=index_input, outputs=[task_description_output] + image_outputs)

print("Launching the Single-action Trajectory-level Viewer")
interface.launch(share=True)