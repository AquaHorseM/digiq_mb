import argparse
import torch
import os
import shutil
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_paths", type=str, nargs='+', default="", help="Path to trajectories")
    parser.add_argument("--image_dirs", type=str, nargs='+', default="", help="Path to images")
    parser.add_argument("--output_data_path", type=str, default="", help="Path to store splited data")
    args = parser.parse_args()

    assert len(args.input_data_paths) == len(args.image_dirs), "different lengths of input_data_paths and image_dirs"

    os.makedirs(args.output_data_path + "/embeddings", exist_ok=True)
    os.makedirs(args.output_data_path + "/images", exist_ok=True)

    idx = 0
    for input_data_path, image_dir in zip(args.input_data_paths, args.image_dirs):
        steps = torch.load(input_data_path, weights_only=False)
        for step in steps:
            image_path = f"{image_dir}/{step['image_path'].split('/')[-2]}/{step['image_path'].split('/')[-1]}"
            dst_file = os.path.join(args.output_data_path, f"{args.output_data_path}/images/IMG-{idx}.{image_path.split('.')[-1]}")
            shutil.copy2(image_path, dst_file)

            np.save(f"{args.output_data_path}/embeddings/IMG-{idx}.npy", step['s_rep'])
            idx += 1