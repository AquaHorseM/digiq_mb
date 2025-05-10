import argparse
import torch

def reconstruct_data(trajs):
    data = []
    for traj in trajs:
        for step, next_step in zip(traj, traj[1:]):
            simple_action = step["action"].split("Action Decision: ")[1]
            data.append(dict(state=step["s_rep"], action=simple_action, next_state=next_step["s_rep"]))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="", help="Path to collcted trajectories (with state representations)")
    parser.add_argument("--output_data_path", type=str, default="", help="Path to store transition data")
    args = parser.parse_args()

    input_data_path = args.input_data_path
    output_data_path = args.output_data_path

    # Load the dataset
    print("Loading dataset...")
    trajs = torch.load(input_data_path, weights_only=False)
    print("Reconstructing dataset...")
    output_data = reconstruct_data(trajs)

    # Save the dataset
    torch.save(output_data, output_data_path)