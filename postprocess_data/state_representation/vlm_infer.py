import argparse

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm

prompt = "You're given a user interface. List all clickable items' positions. Positions are defined as coordinates in the screen which need to be normalized to 0 to 1, e.g. (0.8, 0.2)."

def flatten_trajectories(trajs):
    return [dict(step, traj_id=i, step_id=j) for i, traj in enumerate(trajs) for j, step in enumerate(traj)]

def inference(model_path, image_dir, batch_size, steps):
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path).to("cuda")

    for i in tqdm(range(0, len(steps), batch_size), desc="Processing"):
        def get_path(image_path):
            if image_path.startswith("<path_to_images>"):
                return image_dir + image_path.split("<path_to_images>")[1]
            else:
                return image_dir + image_path.split("images")[1]
        image_paths = [get_path(steps[j]['image_path']) for j in range(i, min(len(steps), i + batch_size))]
        batch_images = [Image.open(p).convert("RGB") for p in image_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            } for image in batch_images
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = model(
                **inputs,
                return_dict=True
            )
            #last_hidden_state = outputs.hidden_states[-1]
            reps = outputs.logits[:, -1, :]
        
        reps = reps.cpu().numpy()
        for rep, step in zip(reps, steps[i:min(len(steps), i+batch_size)]):
            step['s_rep'] = rep

def reconstruct_trajectories(flat_steps, num_trajs):
    trajs = [[] for _ in range(num_trajs)]
    for step in flat_steps:
        trajs[step['traj_id']].append(step)
    for traj in trajs:
        traj.sort(key=lambda x: x['step_id'])
    return trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the VLM model")
    parser.add_argument("--input_data_path", type=str, default="", help="Path to collcted trajectories")
    parser.add_argument("--image_dir", type=str, default="", help="Path to images")
    parser.add_argument("--output_data_path", type=str, default="", help="Path to store trajectories with state representations")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    load_path = args.input_data_path
    save_path = args.output_data_path

    # Load the dataset
    print("Loading dataset...")
    trajs = torch.load(load_path, weights_only=False)
    flat_steps = flatten_trajectories(trajs)
    inference(args.model_path, args.image_dir, args.batch_size, flat_steps)
    reconstructed_trajs = reconstruct_trajectories(flat_steps, len(trajs))
    print("Saving trajectories...")
    torch.save(reconstructed_trajs, save_path)
    print("Saved trajectories.")

