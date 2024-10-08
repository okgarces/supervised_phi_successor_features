import numpy as np
import torch as th


class ReplayBuffer:

    def __init__(self, obs_dim, action_dim, rew_dim=1, max_size=100000, obs_dtype=np.float32, action_dtype=np.float32, device=None):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        if isinstance(obs_dim, tuple):
            self.obs = np.zeros((max_size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=obs_dtype)
            self.next_obs = np.zeros((max_size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=obs_dtype)
        else:
            self.obs = np.zeros((max_size, obs_dim), dtype=obs_dtype)
            self.next_obs = np.zeros((max_size, obs_dim), dtype=obs_dtype)

        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.scalar_rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.gammas = np.zeros((max_size, 1), dtype=np.float32)
        self.device = device

    def add(self, obs, action, reward, next_obs, done, scalar_reward=0, gamma=0.95):
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.scalar_rewards[self.ptr] = scalar_reward
        self.gammas[self.ptr] = gamma

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, scalar_rewards=False):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience

        experience_tuples = (
        self.obs[inds], self.actions[inds], self.rewards[inds], self.next_obs[inds], self.dones[inds],
        self.gammas[inds])
        if scalar_rewards:
            experience_tuples = experience_tuples + (self.scalar_rewards[inds],)

        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(self.device), experience_tuples))
        else:
            return experience_tuples

    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if to_tensor:
            return th.tensor(self.obs[inds]).to(device)
        else:
            return self.obs[inds]

    def get_all_data(self, max_samples=None, scalar_rewards=False):
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)

        experience_tuples = (
        self.obs[inds], self.actions[inds], self.rewards[inds], self.next_obs[inds], self.dones[inds],
        self.gammas[inds])
        if scalar_rewards:
            experience_tuples = experience_tuples + (self.scalar_rewards[inds],)
        return experience_tuples

    def __len__(self):
        return self.size