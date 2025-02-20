import torch
import numpy as np
from PIL import Image
import time
import numpy as np
from vllm import LLM, SamplingParams
import PIL.Image
from tqdm import tqdm
import argparse

def get_responses_representations(user_inputs):
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     top_p=args.top_p,
                                     )
    responses = []
    representations = []
    outputs = llm.generate(user_inputs, sampling_params)
    for o in outputs:
        responses.append(o.outputs[0].text)
        representations.append(o.outputs[0].hidden_states[1].detach().cpu().numpy())
    return responses, representations

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

def get_prompt(action):
    command_prompt = "Respond only 'Yes' or 'No' (without period / quotation marks) and don't respond anything else."
    # command_prompt = ""
    if action is not None:
        if action['action_type'] == "DUAL_POINT":
            prompt = f"You're given a user interface. There is a cursor in the screen. The touch point is located at {action['touch_point']} Is this cursor Clicking on any interactive elements?"
        elif action['action_type'] == "TYPE":
            prompt = f"You're given a user interface. If a user now Types {action['typed_text']}, will this Type action effectively input the text into somewhere on the Screenshot?"""
        elif action['action_type'] == "PRESS_HOME":
            prompt = "You're given a user interface. If a user now Presses the <HOME> button, will this action effectively navigate the user to the Home screen?"
        elif action['action_type'] == "PRESS_BACK":
            prompt = "You're given a user interface. If a user now Presses the <BACK> button, will this action effectively navigate the user to the previous screen?"
        elif action['action_type'] == "PRESS_ENTER":
            prompt = "You're given a user interface. If a user now Presses the <ENTER> button, will this action effectively submit the form?"
        else:
            prompt = "You're given a user interface. Is this action effective?"
    else:
        prompt = "You're given a user interface. Is this action effective?"
    return prompt + "\n" + command_prompt

def construct_inputs(trajs, num_actions=32):
    user_inputs = []
    system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    for traj in tqdm(trajs, desc="Constructing inputs"):
        for step in traj:
            for i in range(num_actions):
                image_path = step["q_image_path_list"][i]
                user_image = PIL.Image.open(image_path, 'r').convert("RGB")
                prompt = get_prompt(step['parsed_action_list'][i])
                user_inputs.append(
                    {
                        "prompt": f"{system_prompt} USER: <image>\n{prompt} ASSISTANT:",
                        "multi_modal_data": {"image": user_image},
                    }
                )
    return user_inputs

def update_trajectories(trajs, responses, representations, num_actions=32):
    for i in tqdm(range(len(trajs))):
        for j in range(len(trajs[i])):
            # action-level fields
            trajs[i][j]['q_rep_out_list'] = []
            trajs[i][j]['q_response_list'] = []
            for _ in range(num_actions):
                trajs[i][j]['q_rep_out_list'].append(representations.pop(0))
                trajs[i][j]['q_response_list'].append(responses.pop(0))
            # stack q_rep_out_list
            trajs[i][j]['q_rep_out_list'] = np.stack(trajs[i][j]['q_rep_out_list'], axis=0)
            
            # step-level fields
            trajs[i][j]['q_rep_out'] = trajs[i][j]['q_rep_out_list'][0]
            trajs[i][j]['q_response'] = trajs[i][j]['q_response_list'][0]
            trajs[i][j]['pred_q_response'] = "unknown"
            trajs[i][j]['state_summarization'] = "unknown"
            trajs[i][j]['new_state_summarization'] = "unknown"
    return trajs

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=0.2)
parser.add_argument("--input-data-path", type=str, default=0.2)
parser.add_argument("--output-data-path", type=str, default=0.2)
parser.add_argument("--num-finegrained-slices", type=float, default=0.2)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=5)
parser.add_argument("--slice-id", type=int, default=0)
parser.add_argument("--num-slices", type=int, default=8)
parser.add_argument("--num-actions", type=int, default=32)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
args = parser.parse_args()

llm = LLM(model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=102400,
        max_model_len=2048,
        enforce_eager=True,
        return_hidden_states=True,
        )

load_path = args.input_data_path
save_path = args.output_data_path + f"-slice{args.slice_id}.pt"

# Load the dataset and define API clients
trajs = torch.load(load_path) # list
slice_length = len(trajs) // args.num_slices
traj_slices = [trajs[i:i+slice_length] for i in range(0, len(trajs), slice_length)]
traj_slice = traj_slices[args.slice_id]

# further slice into slices for sequential processing
num_finegrained_slices = int(args.num_finegrained_slices)
finegrained_slice_length = len(traj_slice) // num_finegrained_slices
output_traj_slice = []
for i in range(num_finegrained_slices):
    finegrained_traj_slice = traj_slice[i*finegrained_slice_length:(i+1)*finegrained_slice_length]
    user_inputs = construct_inputs(finegrained_traj_slice, num_actions=args.num_actions)
    responses, representations = get_responses_representations(user_inputs)
    finegrained_traj_slice = update_trajectories(finegrained_traj_slice, responses, representations, num_actions=args.num_actions)
    output_traj_slice.extend(finegrained_traj_slice)

print(f"Saving trajectories slice {args.slice_id}...")
torch.save(output_traj_slice, save_path)
print("Saved trajectories...")