import numpy as np
from torch.utils.data import Dataset

class ReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer

    def __len__(self):
        # Return the number of items stored (or max_size if you want to sample uniformly)
        return len(self.buffer)

    def __getitem__(self, idx):
        # Return a dictionary of one transition, indexed by idx.
        # (Make sure that the replay_buffer’s arrays are already allocated.)
        return {
            "observation": self.buffer.observations[idx],
            "action": self.buffer.actions[idx],
            # "action_list": self.buffer.action_lists[idx],
            # "image_features": self.buffer.image_features[idx],
            # "next_image_features": self.buffer.next_image_features[idx],
            "reward": self.buffer.rewards[idx],
            "next_observation": self.buffer.next_observations[idx],
            "done": self.buffer.dones[idx],
            "mc_return": self.buffer.mc_returns[idx],
            # "q_rep_out": self.buffer.q_reps_out[idx],
            # "q_rep_out_list": self.buffer.q_reps_out_list[idx],
            "state": self.buffer.state[idx],
            "next_state": self.buffer.next_state[idx],
            "terminal": self.buffer.terminal[idx],
        }


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.mc_returns = None
        # self.image_features = None
        # self.next_image_features = None
        # self.q_reps_out = None
        # self.q_reps_out_list = None
        self.state = None
        self.next_state = None
        self.terminal = None

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation,
        action,
        # action_list,
        # image_features: np.ndarray,
        # next_image_features: np.ndarray,
        reward: np.ndarray,
        next_observation,
        done: np.ndarray,
        mc_return,
        # q_rep_out,
        # q_rep_out_list,
        state,
        next_state,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(done, bool):
            done = np.array(done)

        if self.observations is None:
            self.observations = np.array(['']*self.max_size, dtype = 'object')
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            # self.action_lists = np.array(['']*self.max_size, dtype = 'object')
            # self.image_features = np.empty((self.max_size, *image_features.shape), dtype=image_features.dtype)
            # self.next_image_features = np.empty((self.max_size, *next_image_features.shape), dtype=next_image_features.dtype)
            # self.q_reps_out = np.empty((self.max_size, *q_rep_out.shape), dtype=q_rep_out.dtype)
            # self.q_reps_out_list = np.empty((self.max_size, *q_rep_out_list.shape), dtype=q_rep_out_list.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array(['']*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)
            self.state = np.empty((self.max_size, *state.shape), dtype=state.dtype)
            self.next_state = np.empty((self.max_size, *next_state.shape), dtype=next_state.dtype)
            self.terminal = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)

        assert reward.shape == ()
        assert done.shape == ()

        # in some cases the q get rep errors out, need to get rid of this situation
        self.observations[self.size % self.max_size] = observation
        # self.image_features[self.size % self.max_size] = image_features
        # self.next_image_features[self.size % self.max_size] = next_image_features
        # self.q_reps_out[self.size % self.max_size] = q_rep_out
        # self.q_reps_out_list[self.size % self.max_size] = q_rep_out_list
        self.actions[self.size % self.max_size] = action
        # self.action_lists[self.size % self.max_size] = action_list
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.mc_returns[self.size % self.max_size] = mc_return
        self.state[self.size % self.max_size] = state
        self.next_state[self.size % self.max_size] = next_state
        self.terminal[self.size % self.max_size] = 0 if next_state is None else 0
        self.size += 1

class TransitionReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer

    def __len__(self):
        # Return the number of items stored (or max_size if you want to sample uniformly)
        return len(self.buffer)

    def __getitem__(self, idx):
        # Return a dictionary of one transition, indexed by idx.
        # (Make sure that the replay_buffer’s arrays are already allocated.)
        return {
            "state": self.buffer.states[idx],
            "action": self.buffer.actions[idx],
            "next_state": self.buffer.next_states[idx],
        }


class TransitionReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.states = None
        self.actions = None
        self.next_states = None
        self.batch_size = batch_size
        self.max_size = capacity
        self.size = 0

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.
        """

        if self.states is None:
            self.states = np.empty((self.max_size, *state.shape), dtype=state.dtype)
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            self.next_states = np.empty((self.max_size, *next_state.shape), dtype=next_state.dtype)

        self.states[self.size % self.max_size] = state
        self.actions[self.size % self.max_size] = action
        self.next_states[self.size % self.max_size] = next_state

        self.size += 1