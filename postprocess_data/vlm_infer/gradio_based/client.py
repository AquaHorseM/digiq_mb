import time
import torch
import numpy as np
from gradio_client import Client, handle_file
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", type=str, default="", help="Path to collcted trajectories")
parser.add_argument("--output_data_path", type=str, default="", help="Path to store formatted trajectories")
args = parser.parse_args()

load_path = args.input_data_path
save_path = args.output_data_path

# Load the dataset and define API endpoints
trajs = torch.load(load_path, weights_only=False)
api_endpoints = [
"https://6f83bf18f080f93dd2.gradio.live",
"https://9de1e5a95766b49099.gradio.live",
"https://2bf86516f5f650d38a.gradio.live",
"https://d0d78a8fcbcecabb3c.gradio.live",
"https://7e055bde31e57fec51.gradio.live",
"https://8be4726f4555ccf516.gradio.live",
"https://2fbb035168e5b311e8.gradio.live",
"https://a9f6fc7313d5767cbd.gradio.live",
]

def make_prompt(action_dict):
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


def guaranteed_predict(client, q_image_path, prompt):
    q_prediction = client.predict(handle_file(q_image_path), prompt, api_name="/predict")
    return q_prediction

def process_step(args):
    step, endpoint, traj_id, step_id = args
    step['traj_id'] = traj_id
    step['step_id'] = step_id
    client = Client(endpoint)

    q_rep_out_list = []
    q_response_list = []

    default_embedding_dim = 4096  # Set to the expected embedding dimension

    # Process each action individually
    for idx, (action_dict, q_image_path) in enumerate(zip(step['parsed_action_list'], step['q_image_path_list'])):
        try:
            prompt = make_prompt(action_dict)
            q_prediction = guaranteed_predict(client, q_image_path, prompt)

            q_rep_out = np.array(q_prediction[0]['data'][0], dtype=np.float32)
            q_response = q_prediction[2]

            q_rep_out_list.append(q_rep_out)
            q_response_list.append(q_response)

            # If this is the first action and it succeeds, set 'q_rep_out' and 'q_response'
            if idx == 0:
                step['q_rep_out'] = q_rep_out
                step['q_response'] = q_response

        except Exception as e:
            error_msg = str(e)
            print(f"Error processing action {idx} in traj_id: {traj_id}, step_id: {step_id} due to error: {error_msg}")

            # Determine embedding dimension
            if q_rep_out_list:
                embedding_dim = q_rep_out_list[-1].shape
            else:
                embedding_dim = (default_embedding_dim,)

            default_q_rep_out = np.zeros(embedding_dim, dtype=np.float32)
            q_rep_out_list.append(default_q_rep_out)
            q_response_list.append(f"Error: {error_msg}")

            # If this is the first action and it fails, set 'q_rep_out' to zeros
            if idx == 0:
                step['q_rep_out'] = default_q_rep_out
                step['q_response'] = q_response_list[0]

    # Set the lists in the step
    step['q_rep_out_list'] = q_rep_out_list
    step['q_response_list'] = q_response_list

    return step

def flatten_trajectories(trajs):
    return [dict(step, traj_id=i, step_id=j) for i, traj in enumerate(trajs) for j, step in enumerate(traj)]

def reconstruct_trajectories(flat_steps, num_trajs):
    trajs = [[] for _ in range(num_trajs)]
    for step in flat_steps:
        trajs[step['traj_id']].append(step)
    for traj in trajs:
        traj.sort(key=lambda x: x['step_id'])
    return trajs

def process_steps_with_timeout(steps, api_endpoints, start, timeout=120):
    completed_steps = []
    args_list = [(step, api_endpoints[i % len(api_endpoints)], step['traj_id'], step['step_id']) for i, step in enumerate(steps)]
    
    with ProcessPoolExecutor(max_workers=len(api_endpoints)) as executor:
        futures = {executor.submit(process_step, args): idx for idx, args in enumerate(args_list)}
        for idx, future in enumerate(as_completed(futures)):
            i = futures[future]
            step = steps[i]
            try:
                result = future.result(timeout=timeout)
                completed_steps.append(result)
                if idx % 50 == 0:
                    elapsed = time.time() - start
                    print(f"Finished {idx}, total: {len(steps)}, progress: {idx / len(steps) * 100:.2f}%, time: {elapsed:.2f}s, velocity: {idx / elapsed:.2f} steps/s")
            except TimeoutError as e:
                error_msg = f"Timeout Error: {str(e)}"
                print(f"Timeout processing traj_id: {step['traj_id']}, step_id: {step['step_id']}. Error: {error_msg}")
                completed_steps.append(step)
            except Exception as e:
                error_msg = f"Exception: {str(e)}"
                print(f"Error processing traj_id: {step['traj_id']}, step_id: {step['step_id']}. Error: {error_msg}")
                completed_steps.append(step)
    return completed_steps

if __name__ == "__main__":
    flat_steps = flatten_trajectories(trajs)
    start_time = time.time()
    processed_steps = process_steps_with_timeout(flat_steps, api_endpoints, start_time)
    reconstructed_trajs = reconstruct_trajectories(processed_steps, len(trajs))
    print("Saving trajectories...")
    torch.save(reconstructed_trajs, save_path)
    print("Saved trajectories.")
