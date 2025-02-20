import argparse
import torch
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--llavarep_path_base", type=str, default=8)
parser.add_argument("--num_slices", type=int, default=8)
args = parser.parse_args()

# wait until all slices are generated
while True:
    if all([os.path.exists(f"{args.llavarep_path_base}-slice{i}.pt") for i in range(args.num_slices)]):
        break
    time.sleep(1)

trajs = []
for slice_id in range(args.num_slices):
    path = f"{args.llavarep_path_base}-slice{slice_id}.pt"
    traj_slice = torch.load(path, weights_only=False)
    trajs.extend(traj_slice)
    
# make action_list a string for later learning
for i in range(len(trajs)):
    for j in range(len(trajs[i])):
        trajs[i][j]['action_list'] = "<split>".join(trajs[i][j]['action_list'])

save_path = f"{args.llavarep_path_base}.pt"
torch.save(trajs, save_path)
print("Merged slices to", save_path)