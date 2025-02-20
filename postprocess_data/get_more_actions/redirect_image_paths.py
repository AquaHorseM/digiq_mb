import torch
import copy
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", type=str, default="", help="Path to collcted trajectories")
parser.add_argument("--image_path", type=str, default="", help="Path to images that this trajectory directs to")
parser.add_argument("--output_data_path", type=str, default="", help="Path to store formatted trajectories")
args = parser.parse_args()

orig_trajs = torch.load(args.input_data_path, weights_only=False)
for i in range(len(orig_trajs)):
    for j in range(len(orig_trajs[i])):
        import re
        import os
        print("replacing", orig_trajs[i][j]['image_path'])
        orig_trajs[i][j]['image_path'] = os.path.join(args.image_path, re.sub(r".*?/images/?", "", orig_trajs[i][j]['image_path']))
        print("to       ", orig_trajs[i][j]['image_path'])
        
torch.save(orig_trajs, args.output_data_path)
