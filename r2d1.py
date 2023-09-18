# import cv2
import numpy as np
from time import gmtime, strftime
import pandas as pd
import dataclasses

import torch
import numpy as np
import random
import torch.functional as F
from torch.optim import lr_scheduler

# device = 'cuda:0'
device = 'cpu'

to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.to(
    device)


# Task Embedders
# one hot embedder for actions, last column is the reward
class OAREmbedder:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    # BabyAI embedding according to the observation is not provided
    def __call__(self, action, observation, reward, observations=None):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=device)

        # Inputs are in the form action, observation, reward
        actions = torch.nn.functional.one_hot(action, num_classes=self.num_actions)

        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=device)

        reward = reward.unsqueeze(-1)

        reward = torch.tanh(reward)
        return torch.cat((actions, reward), axis=-1)


import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from collections import deque
import os
from time import time, sleep
import gc
import fasteners
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from collections import deque
import os
from time import time, sleep
import gc
import fasteners
import pickle


class NStepMemory(dict):
    def __init__(self, memory_size=5, gamma=0.997):
        self.memory_size = memory_size
        self.gamma = gamma

        self.q_value = deque(maxlen=memory_size)
        self.state = deque(maxlen=memory_size)
        self.hs = deque(maxlen=memory_size)
        self.cs = deque(maxlen=memory_size)
        self.target_hs = deque(maxlen=memory_size)
        self.target_cs = deque(maxlen=memory_size)
        self.action = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.stack_count = deque(maxlen=memory_size)
        self.phi = deque(maxlen=memory_size)

    @property
    def size(self):
        return len(self.state)

    def add(self, q_value, state, hs, cs, target_hs, target_cs, action, reward, stack_count, phi):
        self.q_value.append(q_value)
        self.state.append(state)
        self.hs.append(hs)
        self.cs.append(cs)
        self.target_hs.append(target_hs)
        self.target_cs.append(target_cs)
        self.action.append(action)
        self.reward.append(reward)
        self.stack_count.append(stack_count)
        self.phi.append(phi)

    def get(self):
        q_value = self.q_value.popleft()
        state = self.state.popleft()
        hs = self.hs.popleft()
        cs = self.cs.popleft()
        target_hs = self.target_hs.popleft()
        target_cs = self.target_cs.popleft()
        action = self.action.popleft()
        stack_count = self.stack_count.popleft()
        reward = sum([self.gamma ** i * r for i, r in enumerate(self.reward)])
        phi = self.phi.popleft()
        return q_value, state, hs, cs, target_hs, target_cs, action, reward, stack_count, phi

    def is_full(self):
        return len(self.state) == self.memory_size


class ReplayMemory:
    def __init__(self, memory_size=100_000, batch_size=32, n_step=3, state_size=(3, 80, 80), cell_size=256,
                 action_repeat=1, n_stacks=1, alpha=0.4):
        # Pong and Atari needs action_repeat and n_stacks = 4
        self.index = 0
        self.memory_size = memory_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.state_size = (action_repeat,) + state_size
        self.action_repeat = action_repeat
        self.n_stacks = n_stacks // action_repeat
        self.alpha = alpha
        self.beta = 0.4
        self.beta_step = 0.00025 / 4
        self.eta = 0.9
        self.burn_in_length = 0  # According to hyperparameter paper in Carvalho2023 0
        self.learning_length = 40
        self.sequence_length = self.burn_in_length + self.learning_length

        self.phi_feature_size = 4

        self.memory = dict()
        self.memory['state'] = np.zeros((self.memory_size, *self.state_size), dtype=np.uint8)
        self.memory['hs_cs'] = np.zeros((self.memory_size, cell_size * 2), dtype=np.float32)
        self.memory['target_hs_cs'] = np.zeros((self.memory_size, cell_size * 2), dtype=np.float32)
        self.memory['action'] = np.ones((self.memory_size, 1), dtype=np.int8)
        self.memory['reward'] = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.memory['done'] = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.memory['stack_count'] = np.zeros((self.memory_size,), dtype=np.int8)
        self.memory['priority'] = np.zeros((self.memory_size,), dtype=np.float32)
        self.memory['sequence_priority'] = np.zeros((self.memory_size,), dtype=np.float32)
        self.memory['is_seq_start'] = np.zeros((self.memory_size,), dtype=np.uint8)
        self.memory['phi'] = np.zeros((self.memory_size, self.phi_feature_size), dtype=np.float32)
        self.arange = np.arange(self.memory_size)

    @property
    def size(self):
        return min(self.index, self.memory_size)

    def add(self, state, hs, cs, target_hs, target_cs, action, reward, done, stack_count, priority, phi):
        index = self.index % self.memory_size
        self.memory['state'][index] = state * 255
        self.memory['hs_cs'][index, :self.cell_size] = hs
        self.memory['hs_cs'][index, self.cell_size:] = cs
        self.memory['target_hs_cs'][index, :self.cell_size] = target_hs
        self.memory['target_hs_cs'][index, self.cell_size:] = target_cs
        self.memory['action'][index] = action
        self.memory['reward'][index] = reward
        self.memory['done'][index] = 1 if done else 0
        self.memory['stack_count'][index] = stack_count
        self.memory['priority'][index] = priority
        self.memory['phi'][index] = phi
        self.index = (self.index + 1) % self.memory_size  # Kiyo fix

    def extend(self, memory):
        start_index = self.index % self.memory_size
        last_index = (start_index + memory['state'].shape[0]) % self.memory_size
        if start_index < last_index:
            index = [i for i in range(start_index, last_index)]
        else:
            index = [i for i in range(start_index, self.memory_size)] + [i for i in range(last_index)]
        index = np.array(index)

        for key in self.memory.keys():
            self.memory[key][index] = memory[key]

        self.index += memory['state'].shape[0]

    def fit(self):
        for key in self.memory.keys():
            self.memory[key] = self.memory[key][:self.size]

    def save(self, path, actor_id):
        path = os.path.join(path, f'memory{actor_id}.pt')
        lock = fasteners.InterProcessLock(path)

        while True:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                if lock.acquire(blocking=False):
                    try:
                        memory = torch.load(path, map_location=lambda storage, loc: storage)
                        self.extend(memory)
                        self.fit()
                        torch.save(self.memory, path)
                    except:
                        os.remove(path)
                    lock.release()
                    gc.collect()
                    return
            else:
                if lock.acquire(blocking=False):
                    try:
                        self.fit()
                        torch.save(self.memory, path)
                    except:
                        os.remove(path)
                    lock.release()
                    gc.collect()
                    return
            sleep(np.random.random() + 2)

    def load(self, path, actor_id):
        path = os.path.join(path, f'memory{actor_id}.pt')
        lock = fasteners.InterProcessLock(path)

        while True:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                if lock.acquire(blocking=False):
                    try:
                        memory = torch.load(path, map_location=lambda storage, loc: storage)
                        self.extend(memory)
                    except:
                        pass
                    os.remove(path)
                    lock.release()
                    gc.collect()
                    return
                else:
                    sleep(np.random.random())
            return

    def update_priority(self, index, priority):
        self.memory['priority'][index] = priority

    def set_hs_cs(self, index, hs, cs, target_hs, target_cs):
        self.memory['hs_cs'][index, :self.cell_size] = hs
        self.memory['hs_cs'][index, self.cell_size:] = cs
        self.memory['target_hs_cs'][index, :self.cell_size] = target_hs
        self.memory['target_hs_cs'][index, self.cell_size:] = target_cs

    def update_sequence_priority(self, index, update_pre_next_seq_priority=False):
        for idx in index:
            indices = np.arange(idx, idx + self.sequence_length) % self.memory_size
            priority = self.memory['priority'][idx: idx + self.sequence_length]

            if len(priority) < 1:
                print(priority, indices, idx, 'Wrong priority')
                continue

            self.memory['sequence_priority'][idx] = self.eta * priority.max() + (1 - self.eta) * priority.mean()

            if update_pre_next_seq_priority:
                update = False
                for i in range(1, self.sequence_length + 1):
                    if i < 0:
                        i = self.memory_size + i
                    if self.memory['is_seq_start'][(idx - i) % self.memory_size] == 1:
                        pre_idx = idx - i
                        update = True
                        break
                if update:
                    indices = np.arange(pre_idx, pre_idx + self.sequence_length) % self.memory_size
                    priority = self.memory['priority'][indices]
                    self.memory['sequence_priority'][pre_idx] = self.eta * priority.max() + (
                                1 - self.eta) * priority.mean()

                update = False
                for i in range(1, self.sequence_length + 1):
                    if self.memory['is_seq_start'][(idx + i) % self.memory_size] == 1:
                        next_idx = idx - i
                        update = True
                        break
                if update:
                    indices = np.arange(next_idx, next_idx + self.sequence_length) % self.memory_size
                    priority = self.memory['priority'][indices]
                    self.memory['sequence_priority'][next_idx] = self.eta * priority.max() + (
                                1 - self.eta) * priority.mean()

    def get_stacked_state(self, index):
        stack_count = self.memory['stack_count'][index]
        start_index = index - (self.n_stacks - stack_count)
        if start_index < 0:
            start_index = self.memory_size + start_index
        stack_index = [start_index for _ in range(stack_count)] + [(start_index + 1 + i) % self.memory_size for i in
                                                                   range(self.n_stacks - stack_count)]
        stacked_state = np.concatenate([self.memory['state'][i] for i in stack_index])
        return stacked_state

    def sample(self, device='cpu'):

        seq_start_index = self.arange[self.memory['is_seq_start'] == 1]

        # Return void when not enough data to complete batch size
        if self.size < self.batch_size or len(seq_start_index) < 1:
            return None, None, None  # Return tuple

        priority = self.memory['sequence_priority'][seq_start_index]

        seq_index = WeightedRandomSampler(
            priority / np.sum(priority),
            self.batch_size,
            replacement=True)
        seq_index = np.array(list(seq_index))
        seq_index = seq_start_index[seq_index]
        next_seq_index = (seq_index + self.n_step) % self.memory_size

        batch = dict()
        batch['state'] = [np.stack([self.get_stacked_state(i % self.memory_size) for i in seq_index + s]) for s in
                          range(self.sequence_length)]
        batch['next_state'] = [np.stack([self.get_stacked_state(i % self.memory_size) for i in next_seq_index + s]) for
                               s in range(self.sequence_length)]
        batch['hs'] = self.memory['hs_cs'][seq_index, :self.cell_size]
        batch['cs'] = self.memory['hs_cs'][seq_index, self.cell_size:]
        batch['target_hs'] = self.memory['target_hs_cs'][next_seq_index, :self.cell_size]
        batch['target_cs'] = self.memory['target_hs_cs'][next_seq_index, self.cell_size:]
        # This replay buffer only works for this Environment. 100 is the no op action.
        # batch['prev_action'] = [[[100]] * len(seq_index)] + [
        #     self.memory['action'][(seq_index - 1 + self.burn_in_length + s) % self.memory_size] for s in
        #     range(1, self.learning_length)]
        batch['prev_action'] = [self.memory['action'][(seq_index - 1 + self.burn_in_length + s) % self.memory_size] for s in
            range(self.learning_length)]
        batch['action'] = [self.memory['action'][(seq_index + self.burn_in_length + s) % self.memory_size] for s in
                           range(self.learning_length)]
        batch['reward'] = [self.memory['reward'][(seq_index + self.burn_in_length + s) % self.memory_size] for s in
                           range(self.learning_length)]
        batch['done'] = [self.memory['done'][(seq_index + self.burn_in_length + s) % self.memory_size] for s in
                         range(self.learning_length)]
        batch['phi'] = [self.memory['phi'][(seq_index + self.burn_in_length + s) % self.memory_size] for s in
                        range(self.learning_length)]

        for key in batch.keys():
            if key not in ['hs', 'cs', 'target_hs', 'target_cs']:
                batch[key] = np.stack(batch[key])

        for key in batch.keys():
            batch[key] = np.stack(batch[key])

        index = np.stack([(seq_index + s) % self.memory_size for s in range(self.sequence_length)])

        for key in ['state', 'next_state']:
            batch[key] = batch[key].astype(np.float32) / 255.

        for key in batch.keys():
            batch[key] = torch.FloatTensor(batch[key]).to(device)
        batch['action'] = batch['action'].long()

        return batch, seq_index, index

    def indexing_sample(self, start_index, last_index, device='cpu'):
        index = np.arange(start_index, last_index) % self.memory_size
        next_index = (index + self.n_step) % self.memory_size

        batch = dict()
        batch['state'] = np.stack([[self.get_stacked_state(i)] for i in index])
        batch['next_state'] = np.stack([[self.get_stacked_state(i % self.memory_size)] for i in next_index])
        batch['action'] = self.memory['action'][index]
        batch['reward'] = self.memory['reward'][index]
        batch['done'] = self.memory['done'][index]

        batch['state'] = batch['state'].astype(np.float32) / 255.
        batch['next_state'] = batch['next_state'].astype(np.float32) / 255.
        return batch, index


class Logger:
    SOURCE_TASK = 'source'
    TARGET_TASK = 'target'

    HEADERS = ['task_id', 'reward', 'step', 'accum_loss', 'q_loss', 'psi_loss', 'phi_loss']

    def __init__(self, root_path):
        self.source_tasks_file = root_path + f'results/source_performance_{strftime("%d_%b_%Y_%H_%M_%S", gmtime())}.csv'
        self.target_tasks_file = root_path + f'results/target_performance_{strftime("%d_%b_%Y_%H_%M_%S", gmtime())}.csv'

    def log_agent_performance(self, task, reward, step, accum_loss, *args, **kwargs):
        values = np.array([task, reward, step, accum_loss, *args])
        type_task = kwargs.get('type_task', self.SOURCE_TASK)
        filename = self.source_tasks_file if type_task == self.SOURCE_TASK else self.target_tasks_file

        print(filename)

        with open(filename, 'a') as f:
            np.savetxt(f, np.column_stack(values), delimiter=',', newline='\n')

    def load_text(self, type_task='source'):
        filename = self.source_tasks_file if type_task == self.SOURCE_TASK else self.target_tasks_file

        return pd.DataFrame(np.loadtxt(filename, delimiter=','))


@dataclasses.dataclass
class CommonConfig:
    dimension: int = 16


@dataclasses.dataclass
class DQNAgentConfig:
    epsilon: int = 0.99 # 0.99 # Borsa2020 keeps 0.1 fixed
    gym_legacy: bool = False
    batch_size: int = 32  # 32 # 128
    # n_training_steps: int = 50_000
    n_training_steps: int = 4_000_000
    evaluation_n_training_steps: int = 100_000
    log_performance_n_training_steps: int = 2_500  # This is not part of Carvalho2023
    n_step_q_learning: int = 5  # Default in Carvalho2023
    episode_length_seconds: int = 60  # According to Barreto2018, Barreto2017 this is 1 minute.
    max_gradient_norm: float = 80

    # Replay buffer
    n_replay_samples: int = 100_000
    trace_length: int = 40
    overlap_length: int = 20  # Default according R2D2 paper
    priority_alpha: float = 0.0 # According to Carvalho2023 is 0.
    priority_epsilon: float = 1e-6
    burn_in_length: int = 0

    # DQN config
    learning_rate: float = 1e-2
    min_learning_rate_factor: float = 1e-4
    total_iters_learning_rate: int = 1_250_000 # the number of sgd steps
    gamma: float = 0.99

    # Target network updates hyperparameters
    n_steps_update_target_model: int = 1_000
    use_target_soft_update: bool = False
    target_update_tau: float = 1e-3

    q_loss_coefficient: float = 1.0 # Carvalho2023 0.5
    psi_loss_coefficient: float = 0.5 # Carvalho2023 1.0
    phi_loss_coefficient: float = 0.0 # Carvalho2023 1.0

    use_full_loss: bool = False

@dataclasses.dataclass
class DQNConfig:
    actions_dim: int = 1  # Different from number of actions
    states_dim: int = 128  # This is the same as the VisionConfig output_layers_dim

    # n_actions: int = 2 # Cartpole
    n_actions: int = 5  # Gridworld is 7. For GoToAndAvoid is 4, and 5 with the no-op

@dataclasses.dataclass
class VisionConfig(CommonConfig):
    # input_channel: int = 16 # Can be the same as CommonConfig?
    input_channel: int = 3  # Input channels are 3 RGB
    output_channel: int = 16  # According to code hyperparameters
    output_layers_dim: int = 128  # 0 According to code hyperparameters
    flatten: bool = True

# DQN ConvNet
# Vision Torso for Go To Environment
class DQNConvolutionalNetwork(torch.nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        input_dim = config.input_channel
        conv_dim = config.output_channel
        out_dim = config.output_layers_dim

        layers = [
            torch.nn.Conv2d(input_dim, 128, (8, 8), stride=8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, (3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, (3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, conv_dim, (1, 1), stride=1)

            # torch.nn.Conv2d(input_dim, 128, (8,8), stride=8), # Original "code" says the vision-torso is (8,8)
            # torch.nn.Conv2d(128, 128, (3,3), stride=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(128, 128, (3,3), stride=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(128, conv_dim, (1,1), stride=1)
        ]

        self._network = torch.nn.Sequential(*layers)
        self.flatten = config.flatten
        if out_dim:
            # self.out_net = torch.nn.Linear(256, out_dim) # 256 using stride 8
            self.out_net = torch.nn.Linear(576, out_dim)  # 256 using stride 8
        else:
            self.out_net = lambda x: x

    def forward(self, inputs: torch.Tensor):
        inputs_rank = inputs.ndim
        batched_inputs = inputs_rank == 4  # Batch is 4

        outputs = self._network(inputs)
        if not self.flatten:
            return outputs

        if batched_inputs:
            flat = torch.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
        else:
            flat = torch.reshape(outputs, [-1])  # [D]

        return self.out_net(flat)


# DQN_USF_MODEL
# DQN_USF_MODEL

class DQN_Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.vision_config = VisionConfig()

        self._vision_net = DQNConvolutionalNetwork(self.vision_config).to(device=device)
        self.dqn_config = DQNConfig()

        self.recurrent_hidden_state_dim = 256  # TODO check this as config constant
        self.recurrent_hidden_state_rollout = None
        self.recurrent_cell_state_rollout = None
        self.recurrent_hidden_state_train = None
        self.recurrent_cell_state_train = None


        # self._vision_net = torch.nn.Linear(4, 128)

        # self._recurrent_net = torch.nn.LSTM(self.sf_config.states_dim + self.sf_config.actions_dim, self.sf_config.states_dim, batch_first=True, hidden_size=256)
        # I changed the batch_first to align the result with the replay buffer.
        self._recurrent_net = torch.nn.LSTM(self.dqn_config.states_dim + self.dqn_config.actions_dim, batch_first=False,
                                            hidden_size=self.recurrent_hidden_state_dim)
        self._f_recurrent = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_hidden_state_dim, self.dqn_config.states_dim),
            torch.nn.ReLU()
        )

        self._dqn_model = torch.nn.Sequential(
            torch.nn.Linear(self.dqn_config.states_dim, 128),
            torch.nn.ReLU(),
            # torch.nn.Linear(128 * 2, 256),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, self.dqn_config.n_actions)
        )

    def reset_recurrent_hidden_state(self):
        self.recurrent_hidden_state_train = None
        self.recurrent_cell_state_train = None

    def set_recurrent_hidden_state(self, hidden_state, cell_state):  # This set train hidden state
        if hidden_state.ndim == 2:
            hidden_state = hidden_state.unsqueeze(0)  # Add trace dimension [n_trace, n_batch...]
            cell_state = cell_state.unsqueeze(0)

        self.recurrent_hidden_state_train = hidden_state
        self.recurrent_cell_state_train = cell_state

    def get_recurrent_hidden_state(self):
        return self.recurrent_hidden_state_train, self.recurrent_cell_state_train

    def forward(self, obs, action_taken, mode='rollout'):
        # Prepare inputs
        inputs = to_tensor(obs)  # [n_trace, n_batch, ...] if coming from replay buffer else [n_batch, n_channel,...]
        action_taken = to_tensor(action_taken)

        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(0)  # [n_trace, n_batch, n_channel, ...]

        if inputs.ndim == 6:  # coming from replay buffer and axis=2 is the number of stacked frames.
            inputs = inputs.squeeze(2)  # to have [n_trace, n_batch, n_channel] Remove the stack frames

        # Apply Convolutions to images
        n_batch = inputs.shape[1]
        n_trace = inputs.shape[0]

        vision_input = inputs.reshape(-1, self.vision_config.input_channel, inputs.shape[3], inputs.shape[4])
        vision_output = self._vision_net(vision_input).view(n_trace, n_batch, -1)  # [n_batch, n_trace, dim] restore trace dim

        # Initialize recurrent hideen states and cell states
        initial_hidden_state, initial_cell_state = None, None
        if mode == 'train':
            if self.recurrent_cell_state_train is None:
                self.recurrent_hidden_state_train = torch.zeros((1, n_batch, self.recurrent_hidden_state_dim)).to(
                    device)  # This is the same dim as hidden in LSTM
                self.recurrent_cell_state_train = torch.zeros((1, n_batch, self.recurrent_hidden_state_dim)).to(
                    device)  # This is the same dim as hidden in LSTM

            initial_hidden_state = self.recurrent_hidden_state_train
            initial_cell_state = self.recurrent_cell_state_train

        elif mode == 'rollout':
            if self.recurrent_cell_state_rollout is None:
                self.recurrent_hidden_state_rollout = torch.zeros((1, n_batch, self.recurrent_hidden_state_dim)).to(
                    device)  # This is the same dim as hidden in LSTM
                self.recurrent_cell_state_rollout = torch.zeros((1, n_batch, self.recurrent_hidden_state_dim)).to(
                    device)  # This is the same dim as hidden in LSTM

            initial_hidden_state = self.recurrent_hidden_state_rollout
            initial_cell_state = self.recurrent_cell_state_rollout

        else:
            raise Exception('Undefined forward mode. Must be train or rollout')

        for i in range(vision_output.ndim - action_taken.ndim):
            action_taken = action_taken.unsqueeze(-1)

        # Recurrent Torso
        recurrent_input = torch.concat([vision_output, action_taken], dim=-1)
        recurrent_output, (hidden_state, cell_state) = self._recurrent_net(recurrent_input,
                                                                           (initial_hidden_state, initial_cell_state))

        recurrent_output = self._f_recurrent(recurrent_output)  # [n_trace, n_batch, hidden_lstm_size]

        q_values = self._dqn_model(recurrent_output).view(n_trace, n_batch, self.dqn_config.n_actions)
        # [n_trace, n_batch, d_zamples + 1, n_actions, d_features]
        # Borsa2020 paper said the rollout computes with the reward maper.
        if mode == 'rollout':
            q_values = q_values[-1, :]
            hidden_state = hidden_state[-1, :]
            cell_state = cell_state[-1, :]
        # elif mode == 'train':
        #     z_samples = z_samples.view(n_trace, n_batch, self.sf_config.d_z_samples, self.sf_config.features_dim).unsqueeze(-2).repeat(1,1,1,1,1)
        #     q_values = torch.matmul(sf_values, z_samples)

        return q_values, hidden_state, cell_state


class R2D1_NStep:
    """
    Modular Successor Feature Approximator Agent
    """

    def __init__(self, source_tasks, target_tasks, vector_tasks, config):
        self.source_tasks = source_tasks
        self.target_tasks = target_tasks
        self.vector_tasks = vector_tasks  # This is for the USFA heads.
        self.logger = Logger('')

        self.config = config
        self.dqn_config = DQNConfig()

        # self.n_acton
        self.epsilon = config.epsilon
        self.min_epsilon = 0.1

        self.gym_legacy = config.gym_legacy
        self.n_batch = config.batch_size
        self.n_training_steps = config.n_training_steps
        self.evaluation_n_training_steps = config.evaluation_n_training_steps
        self.log_performance_n_training_steps = config.log_performance_n_training_steps
        self.priority_alpha = config.priority_alpha
        self.priority_epsilon = config.priority_epsilon

        # config DQN
        self.n_step_q_learning = config.n_step_q_learning
        if config.n_step_q_learning > 0:
            # self.replay_buffer = NStepQLearningMemory(config)
            # self.replay_buffer = RecurrentNStepQLearningMemory(config)
            self.n_step_replay_buffer = NStepMemory(config.n_step_q_learning, gamma=config.gamma)
            self.replay_buffer = ReplayMemory(config.n_replay_samples, config.batch_size, config.n_step_q_learning,
                                              action_repeat=1)

        # Task configuration
        first_task = self.source_tasks[0]
        self.task_embedder = OAREmbedder(first_task.action_space.n)

        # Initialise models DQN test
        # self.msf = RecurrentNetwork()
        self.loss = torch.nn.MSELoss()
        # self.target_network = DQN_SF_Model()
        # self.policy_network = DQN_SF_Model()

        self.target_network = DQN_Model().to(device)
        self.policy_network = DQN_Model().to(device)

        with torch.no_grad():
            self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.optimizer_scheduler = lr_scheduler.LinearLR(self.optimizer,
                                                         start_factor=1.0,
                                                         end_factor=config.min_learning_rate_factor,
                                                         total_iters=config.total_iters_learning_rate)

        self.gamma = config.gamma

        self.use_target_soft_update = config.use_target_soft_update
        self.target_update_tau = config.target_update_tau
        self.n_steps_update_target_model = config.n_steps_update_target_model  # If soft update is True this is not needed.

        self.policy_network.train()
        self.target_network.eval()

        # Any n_step_q_learning
        self.minimum_exploration_steps = 1_000
        self.number_of_sgd_steps = 40

        # Debug purposes
        self.debug_gradients = True
        self.current_state_from_render = False

    def process_current_state(self, obs, env):

        if isinstance(obs, tuple):
            obs, *_ = obs

        if self.current_state_from_render:
            # Get image from observations render
            current_obs = {'image': cv2.resize(env.render(), [64, 64])}  # For CARTPOLE
            current_state = to_tensor(current_obs['image']).unsqueeze(0)  # B H W C
        else:
            # Get image from observations using
            current_obs = obs
            current_state = to_tensor(current_obs['image']).unsqueeze(0)

        current_state = current_state.permute([0, 3, 1, 2])  # B C H W

        return current_obs, current_state

    def compute_priority(self, q_value, action, reward, next_q_value, target_next_q_value, done):

        q_value = q_value.view(-1)[action]

        if done:
            target_q_value = reward
        else:
            next_action = next_q_value.argmax(-1)
            target_next_q_value = target_next_q_value.view(-1)[next_action]
            target_q_value = reward + (self.gamma ** self.n_step_q_learning) * target_next_q_value

        # to cpu after computation
        priority = np.abs(q_value - target_q_value) + self.priority_epsilon
        priority = priority ** self.priority_alpha

        return priority

    def set_seq_start_index(self):
        # Aware index should
        last_index = self.replay_buffer.index
        start_index = self.episode_start_index

        stop_index = max(last_index - self.config.trace_length, start_index)

        seq_start_index = [i for i in range(start_index, stop_index, self.config.overlap_length)]
        seq_start_index.append(stop_index)
        seq_start_index = np.array(seq_start_index)
        self.replay_buffer.update_sequence_priority(seq_start_index)
        try:
            self.replay_buffer.memory['is_seq_start'][seq_start_index] = 1
        except:
            raise Exception(f'Error indexing {last_index, start_index, seq_start_index}')

    def execute(self):
        # reset agent

        # Train agent with all source tasks
        for source_task_id, env in enumerate(self.source_tasks):
            ### Reset Environment
            obs = env.reset()
            # env.reset() # Do not store obs for USFA

            current_obs, current_state = self.process_current_state(obs, env)
            # current_obs = None
            # current_state = None
            # current_action = None
            next_action = None  # Change name to be more accurate.
            previous_action = env.get_no_op_action()  #

            # episode rewards
            # reset_episode this case we will manage only one episode. This would be like a accum_rewards
            self.episode_rewards = 0.0
            self.episode_loss = 0.0
            self.episode_q_loss = 0.0
            self.episode_psi_loss = 0.0
            self.episode_phi_loss = 0.0
            self.episode_greedy_action_selection = {k: 0 for k in range(env.action_space.n)} # This is for logging purposes

            self.n_episode = 0
            self.episode_start_index = 0  # This is fot the Replay Buffer
            # Statistics
            self.number_greedy_actions = 0

            # Replay buffers or slide window for N-Step Q learning
            accum_loss = torch.tensor(0, device=device)

            for training_step in range(self.n_training_steps):
                #########################################################
                ### Training in source tasks
                ### Actor using epsilon greedy
                with torch.no_grad():
                    # To train R2D2 based need to store the recurrent state
                    # Be aware. USFA needs the previous action taken. If current action is None is an initial state
                    # we need to sample the action.

                    # According to Carvalho they fit the successor feature using the vector_to_reward or reward mapper
                    # Here current_action was the previous action that leads to the current state
                    current_q_value, hidden_state, cell_state = self.policy_network(current_state, previous_action)
                    target_q_value, target_hidden_state, target_cell_state = self.target_network(current_state, previous_action)

                    if (random.random() <= self.epsilon):
                        next_action = env.action_space.sample()
                    else:
                        next_action = torch.argmax(current_q_value, dim=1).item()  # [n_batch, n_actions ]

                        self.number_greedy_actions += 1
                        self.episode_greedy_action_selection[next_action] += 1

                    if self.gym_legacy:  # Step the environoment with the sampled random action
                        next_obs, reward, terminated, info = env.step(next_action)
                    else:
                        next_obs, reward, terminated, truncated, info = env.step(next_action)

                    # Embedded vector
                    # embedded_vector = self.task_embedder(a, obs, reward)
                    next_obs, next_state = self.process_current_state(next_obs, env)

                # Oracle phi
                phi_value = info['features']  # If phi is fixed is coming from Env.

                # Only store if current state
                self.n_step_replay_buffer.add(current_q_value.cpu(),
                                              current_state.cpu(),
                                              hidden_state.cpu(),
                                              cell_state.cpu(),
                                              target_hidden_state.cpu(),
                                              target_cell_state.cpu(),
                                              next_action,
                                              reward,
                                              0,
                                              phi_value)

                # Accumulate episode rewards and losses.
                self.episode_rewards += reward

                # Go to the next state
                current_obs = next_obs
                current_state = next_state
                previous_action = next_action

                # Linear epsilon discount
                if training_step >= self.minimum_exploration_steps:
                    self.epsilon = max(self.epsilon * (1 - 1e-6), self.min_epsilon)  # Have a minimum of exploration. Be aware epsilon.

                if training_step > self.n_step_q_learning:
                    pre_q_value, state, h, c, target_h, target_c, action, reward, stack_count, phi = self.n_step_replay_buffer.get()
                    priority = self.compute_priority(pre_q_value, action, reward, current_q_value.cpu(), target_q_value.cpu(), terminated)
                    self.replay_buffer.add(state, h, c, target_h, target_c, action, reward, terminated, stack_count, priority, phi)

                # Empty n_step_replay_buffer
                #########################################################
                ### Logging source and target tasks performance
                if terminated:
                    self.n_episode += 1
                    while self.n_step_replay_buffer.size > 0:
                        pre_q_value, state, h, c, target_h, target_c, action, reward, stack_count, phi = self.n_step_replay_buffer.get()
                        priority = self.compute_priority(pre_q_value, action, reward, current_q_value.cpu(), target_q_value.cpu(), terminated)
                        self.replay_buffer.add(state, h, c, target_h, target_c, action, reward, terminated, stack_count, priority, phi)

                    self.policy_network.reset_recurrent_hidden_state()
                    self.target_network.reset_recurrent_hidden_state()

                    self.set_seq_start_index()  # Set the start index to train.
                    self.episode_start_index = self.replay_buffer.index

                    self.n_step_replay_buffer = NStepMemory(self.n_step_q_learning,
                                                            self.gamma)  # This is to reset the Buffer memory.
                    previous_action = env.get_no_op_action()

                if training_step % self.config.log_performance_n_training_steps == 0:
                    self.logger.log_agent_performance(source_task_id,
                                                      self.episode_rewards,
                                                      self.n_episode,  # n_number of times terminated
                                                      training_step,
                                                      self.episode_loss,
                                                      self.episode_q_loss,
                                                      self.episode_psi_loss,
                                                      self.number_greedy_actions,
                                                      self.episode_phi_loss,
                                                      *dict(sorted(self.episode_greedy_action_selection.items())).values()
                                                      )

                    # current_obs, _ = env.reset()
                    # current_obs, current_state = self.process_current_state(current_obs, env)
                    # current_action = None # reinit action

                    # Reset episode
                    # self.episode_rewards = 0.0
                    # self.episode_loss = 0.0
                    # self.episode_q_loss = 0.0
                    # self.episode_psi_loss = 0.0
                    # self.number_greedy_actions = 0

                ######################################################################################
                # Training process models
                ######################################################################################
                if (training_step > self.minimum_exploration_steps) and (training_step % 10) == 0:

                    replay_buffer, seq_index, index = self.replay_buffer.sample()

                    if replay_buffer is not None:

                        # Update models in DQN style
                        batch_states = to_tensor(replay_buffer['state'])
                        batch_prev_actions = to_tensor(replay_buffer['prev_action'])
                        batch_actions = to_tensor(replay_buffer['action'])
                        batch_rewards = to_tensor(replay_buffer['reward'])
                        batch_next_states = to_tensor(replay_buffer['next_state'])
                        batch_terminated_masks = to_tensor(replay_buffer['done']).to(dtype=torch.int)
                        batch_phis = to_tensor(replay_buffer['phi'])

                        batch_hidden_states = to_tensor(replay_buffer['hs'])
                        batch_cell_states = to_tensor(replay_buffer['cs'])
                        batch_target_hidden_states = to_tensor(replay_buffer['target_hs'])
                        batch_target_cell_states = to_tensor(replay_buffer['target_cs'])

                        self.policy_network.set_recurrent_hidden_state(batch_hidden_states, batch_cell_states)
                        self.target_network.set_recurrent_hidden_state(batch_target_hidden_states,
                                                                       batch_target_cell_states)

                        # For periodic environments
                        with torch.no_grad():
                            # Terminated states have 0 value
                            gammas = self.gamma ** (self.n_step_q_learning)
                            target_q_value = self.target_network(batch_next_states, batch_actions, 'train')[0]  # DQN needs states and actions?
                            target_q_value_policy_network = self.policy_network(batch_next_states, batch_actions, 'train')[0] # index 1 is the q_value

                            # Computa target q values
                            # [n_batch, n_trace, d_z_samples, n_actions]
                            # max_target_q_values_indices = torch.max(target_q_value_sf, axis=2, keepdim=True)
                            # max_target_q_value_sf = max_target_q_values_indices.values  # Perform GPI over z_samples
                            next_actions = torch.argmax(target_q_value_policy_network , axis=2, keepdim=True)  # GPI to get actions Double DQN
                            max_target_q_value = target_q_value.gather(2, next_actions)
                            target_q_values = batch_rewards + (gammas * (1 - batch_terminated_masks)) * max_target_q_value.view(self.config.trace_length, self.n_batch, -1)  # Remove last dimension no needed

                        # next actions? For the Q value network we should have the next action
                        current_q_value, *_ = self.policy_network(batch_states, batch_prev_actions, 'train')
                        #  [n_trace, n_batch, d_z_sample, n_actions, d_features]
                        # current q value
                        batch_actions = to_tensor(batch_actions) # Same dimensions in the samples.
                        current_q_value = current_q_value.gather(2, batch_actions).view(self.config.trace_length, self.n_batch, -1)

                        # First loss Q-Loss
                        q_loss_n_trace = 0.5 * torch.sum(torch.square(target_q_values - current_q_value), dim=0)  # [B, ]
                        q_loss = torch.mean(q_loss_n_trace, dim=0)  # mean over traces

                        # Second Loss psi-loss
                        psi_loss = self.loss(to_tensor(torch.tensor(0, dtype=torch.float64)), to_tensor(torch.tensor(0, dtype=torch.float64)))

                        # TODO Third loss phi-loss
                        phi_loss = self.loss(to_tensor(torch.tensor(0, dtype=torch.float64)), to_tensor(torch.tensor(0, dtype=torch.float64)))

                        accum_loss = self.config.q_loss_coefficient * q_loss + self.config.psi_loss_coefficient * psi_loss + self.config.phi_loss_coefficient * phi_loss

                        self.optimizer.zero_grad()
                        accum_loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_gradient_norm)

                        for _  in range(self.number_of_sgd_steps):
                            self.optimizer.step()

                        self.optimizer_scheduler.step() # Using Learning Rate Scheduler

                        # Add episode loss when update statistics
                        self.episode_loss += accum_loss.item()
                        self.episode_q_loss += q_loss.item()
                        self.episode_psi_loss += psi_loss.item()
                        self.episode_phi_loss += phi_loss.item()

                        # Kiyo initiative: Update priority with mean in the last dimension
                        priority = (np.abs((current_q_value - target_q_values).detach().cpu().numpy().mean(axis=-1)).reshape(
                            -1) + self.priority_epsilon) ** self.priority_alpha
                        self.replay_buffer.update_priority(index[self.config.burn_in_length:].reshape(-1), priority)
                        self.replay_buffer.update_sequence_priority(seq_index, True)

                        # For debugging purposes
                        if self.debug_gradients:
                            accum_norm = 0
                            accum_norm_params = 0
                            for param in self.policy_network.parameters():
                                accum_norm += torch.linalg.norm(param.data)
                                accum_norm_params += torch.mean(param.grad)

                            print('after', training_step, accum_norm, accum_norm_params, accum_loss)

                        # free memory
                        torch.cuda.empty_cache()

                #########################################################
                ### Update target network. Using soft-update or hard-update
                if self.use_target_soft_update:
                    with torch.no_grad():
                        # if soft update the params based on DPPG
                        for target_param, source_param in zip(self.target_network.parameters(),
                                                              self.policy_network.parameters()):
                            target_param.data.copy_((self.target_update_tau * source_param.data) + (
                                        1 - self.target_update_tau) * target_param.data)
                else:
                    if training_step % self.n_steps_update_target_model == 0:
                        with torch.no_grad():
                            # self.target_network.load_state_dict(self.policy_network.state_dict())
                            for target_param, source_param in zip(self.target_network.parameters(),
                                                                  self.policy_network.parameters()):
                                target_param.data.copy_(source_param.data)
                #########################################################
                ### Evaluation in target tasks
                if (training_step % self.evaluation_n_training_steps == 0):
                    print('Mock testing every', self.evaluation_n_training_steps)
                    # self.evaluate_target_tasks(training_step)
    @torch.no_grad()
    def evaluate_target_tasks(self, training_step):
        import time

        max_n_episodes = 10  # This can be part of the config
        number_of_minutes = self.config.episode_length_seconds * 1
        epsilon_target_tasks = 1e-3

        for target_task_id, target_task in enumerate(self.target_tasks):
            target_task_vector_tensor = to_tensor(target_task.vector_to_reward)
            total_reward = 0

            for n_episode in range(max_n_episodes):
                max_episode_len = time.time() + number_of_minutes

                obs = target_task.reset()
                current_obs, current_state = self.process_current_state(obs, target_task)
                action = target_task.get_no_op_action()

                while time.time() <= max_episode_len:
                    if (random.random() <= epsilon_target_tasks):
                        action = target_task.action_space.sample()
                    else:
                        # Using GPI according to Borsa2019. max z_samples and max over training tasks.
                        # [1, d_z_samples, n_actions]
                        source_q_value, *_ = self.policy_network(current_state, action)
                        action = torch.argmax(source_q_value, dim=1).item()  # [n_trace, n_batch, n_actions]

                    obs, reward, *_ = target_task.step(action)
                    current_obs, current_state = self.process_current_state(obs, target_task)

                    total_reward += reward

            self.logger.log_agent_performance(target_task_id,
                                              total_reward / max_n_episodes,
                                              training_step,
                                              0.0,
                                              type_task='target')


if __name__ == '__main__':
    import gymnasium as gym
    from minigrid.wrappers import RGBImgObsWrapper
    from envs.babyai.gotoavoid import PickupAndAvoid

    # First is
    vector_to_reward = [1, 0, 0, 0]
    number_of_objects = len(vector_to_reward) * 3

    training_envs = [RGBImgObsWrapper(PickupAndAvoid(number_of_objects, vector_to_reward))]
    testing_envs = [RGBImgObsWrapper(PickupAndAvoid(number_of_objects, [1, 0, 0, 0])),
                    RGBImgObsWrapper(PickupAndAvoid(number_of_objects, [0, 0, 1, 1]))]

    config = DQNAgentConfig()

    msf = R2D1_NStep(training_envs, testing_envs, [vector_to_reward], config)
    msf.execute()

    # for env in training_envs:
    #     env.reset()
    #     env.render()

    # for env in training_envs:
    #     env.close()
