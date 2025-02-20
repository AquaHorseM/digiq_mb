import torch
import copy
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", type=str, default="", help="Path to collcted trajectories")
parser.add_argument("--output_data_path", type=str, default="", help="Path to store formatted trajectories")
args = parser.parse_args()

orig_trajs = torch.load(args.input_data_path, weights_only=False)
trajs = copy.deepcopy(orig_trajs)

for i, traj in enumerate(orig_trajs):
    for j, step in enumerate(traj):
        trajs[i][j]['next_image_path'] = trajs[i][j]['image_path']
    
    trajs[i][0]['image_path'] = trajs[i][0]['image_path'].replace("_1.png", "_0.png")
    for j in range(1, len(traj)):
        trajs[i][j]['image_path'] = trajs[i][j-1]['next_image_path']
    
    for j in range(len(traj)-2, 0, -1):
        trajs[i][j]['mc_return'] = trajs[i][j]['reward'] + trajs[i][j+1]['mc_return'] * 0.95

for traj_id in range(len(trajs)):
    for step_id in range(len(trajs[traj_id])):
        trajs[traj_id][step_id]['traj_id'] = traj_id
        trajs[traj_id][step_id]['step_id'] = step_id

torch.save(trajs, args.output_data_path)
print(f"Saved formatted trajectories to {args.output_data_path}")
