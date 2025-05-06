import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.transition_model import Transition_Model

class TransitionModel_Trainer:
    def __init__(self, state_dim:int, action_dim:int, embed_dim:int, num_attn_layers:int=3, num_heads:int=5, activation:str="ReLU", device:str='cuda', checkpoint_path:str=None):
        self.state_encoder = None # TODO
        self.action_encoder = None # TODO

        self.trainsition_model = Transition_Model(state_dim, action_dim, embed_dim, num_attn_layers, num_heads, activation, device)
        self.optimizer = optim.Adam(self.trainsition_model.parameters)

        self.init_weight(checkpoint_path, device)

    def init_weight(self, path:str, device:str):
        if path:
            self.trainsition_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            self.trainsition_model.init_weight()

    def train(self, batch, mini_batch_size, epochs):
        batch_size = batch.size
        indices = np.arange(batch_size)
        states_ori, actions_ori, next_states_ori = batch
        with torch.no_grad:
            states = self.state_encoder(states_ori)
            actions = self.action_encoder(actions_ori)
            next_states = self.state_encoder(next_states_ori)

        for _ in range(epochs):
            end = start + self.mini_batch_size
            minibatch_idx = indices[start:end]
            
            for start in range(0, batch.size, mini_batch_size):
                next_states_pre = self.trainsition_model.forward(states[minibatch_idx], actions[minibatch_idx])
                loss = F.mse_loss(next_states_pre, next_states[minibatch_idx])
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self, batch):
        pass