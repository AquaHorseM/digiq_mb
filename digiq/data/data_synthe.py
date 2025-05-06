import torch

from models.transition_model import Transition_Model

class Data_Synthesizer_with_TransitionMoel:
    def __init__(self, state_dim:int, action_dim:int, embed_dim:int, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", device:str='cuda', checkpoint_path:str=None, data_path:str=None):
        self.trainsition_model = Transition_Model(state_dim, action_dim, embed_dim, num_attn_layers, num_heads, activation, device)
        assert(checkpoint_path), "If you want synthesize data, you should have a trained transition model."
        self.trainsition_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    def synthesize(self):
        pass