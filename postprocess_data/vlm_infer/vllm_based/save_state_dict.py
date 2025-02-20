import torch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_model_path", type=str, default="", help="Path to collcted trajectories")
parser.add_argument("--output_state_dict_path", type=str, default="", help="Path to store formatted trajectories")
args = parser.parse_args()

# load model
disable_torch_init()
model_path = args.input_model_path 
model_name = get_model_name_from_path(model_path)
model_base = None
model = tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, "cuda:0")
print(model.config)

# Save state dict
torch.save(model.state_dict(), args.output_state_dict_path)
