import dataclasses

import numpy as np
import torch
from gymnasium.wrappers import PixelObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper
from torch import optim

from envs.babyai.gotoavoid import PickupAndAvoid
from envs.cartpole.cartpole import CartpoleDissimilar

from nets.CNN import DQNConvolutionalNetwork
from nets.QNetwork import QNetwork

from utils.buffer import ReplayBuffer
from utils.logger import Logger
from utils.torch import polyak_update, linearly_decaying_epsilon

import argparse


def resize_obs(observation, shape=(80,80)):
    """Updates the observations by resizing the observation to shape given by :attr:`shape`."""
    import cv2

    observation = cv2.resize(observation, shape, interpolation=cv2.INTER_AREA)
    return observation


@dataclasses.dataclass
class DQNConfig:
    learning_rate = 1e-3
    gradient_updates = 1
    batch_size = 256
    gamma = 0.95

    tau = 1
    target_net_update_freq = 10_000

    initial_epsilon = 1
    epsilon_decay_steps = 1.6e6
    final_epsilon = 0.03

    use_image_obs = True


@dataclasses.dataclass
class GotoAvoid(DQNConfig):
    pass


@dataclasses.dataclass
class CartpoleConfig(DQNConfig):
    gamma = 0.99
    epsilon_decay_steps = 1e6

    use_image_obs = False

class DQNAgent:
    def __init__(self, input_shape, env=None, device=None, config: DQNConfig = None):
        self.device = device

        self.learning_rate = config.learning_rate
        self.action_dim = 1
        self.gradient_updates = config.gradient_updates
        self.batch_size = config.batch_size

        self.gamma = config.gamma
        self.tau = config.tau
        self.target_net_update_freq = config.target_net_update_freq

        self.initial_epsilon = config.initial_epsilon
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self.final_epsilon = config.final_epsilon

        self.epsilon = self.initial_epsilon

        self.env = env
        self.n_actions = env.action_space.n

        self.num_timesteps = 0
        self.logger = Logger('./', prefix=config.__class__.__name__.lower())

        self.use_image_obs = config.use_image_obs

        if self.use_image_obs:
            self.obs_dim = input_shape
            self.q_net = DQNConvolutionalNetwork(input_shape, self.n_actions).to(self.device)
            self.target_q_net = DQNConvolutionalNetwork(input_shape, self.n_actions).to(self.device)
        else:
            self.obs_dim = self.env.observation_space.shape[0]
            self.q_net = QNetwork(self.obs_dim, self.n_actions).to(self.device)
            self.target_q_net = QNetwork(self.obs_dim, self.n_actions).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(self.obs_dim, self.action_dim, rew_dim=1, max_size=int(100_000), device=self.device)

    def process_obs(self, obs):
        if self.use_image_obs and 'image' in obs.keys():
            x = resize_obs(obs['image'])
            x = np.expand_dims(x, axis=0)
            x = np.transpose(x, (0,3,1,2))
            return x
        else:
            return obs

    def train(self):
        """
        This function is to run the training process and logging.
        """

        critic_loss = torch.tensor(0)

        for _ in range(self.gradient_updates):

            s_obs, s_actions, s_rewards, s_next_obs, s_dones, gammas = self.replay_buffer.sample(self.batch_size, to_tensor=True)

            with torch.no_grad():

                # double q-learning
                q_values = self.q_net(s_next_obs)
                max_acts = torch.argmax(q_values, dim=1)

                q_targets = self.target_q_net(s_next_obs)
                q_targets = q_targets.gather(1, max_acts.long().reshape(-1,1).expand(q_targets.size(0), 1))

                q_targets = (s_rewards + (1 - s_dones) * gammas * q_targets).detach()

                q_targets = q_targets.reshape(-1)

            q_value = self.q_net(s_obs)
            q_value = q_value.gather(1, s_actions.long().reshape(-1,1).expand(q_value.size(0), 1))
            q_value = q_value.reshape(-1)

            critic_loss = torch.nn.functional.mse_loss(q_value, q_targets)

            self.q_optim.zero_grad()
            critic_loss.backward()
            # th.nn.utils.clip_grad_norm_(self.psi_net.parameters(), 10.0)
            self.q_optim.step()

        if self.tau != 1.0 or self.num_timesteps % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps,1, self.final_epsilon)

        if self.num_timesteps % 100 == 0:
            self.logger.log({"losses/critic_loss": critic_loss.item(), 'timesteps': self.num_timesteps})
            self.logger.log({"metrics/epsilon": self.epsilon.item(), 'timesteps': self.num_timesteps})

            # accum_grads = 0
            # accum_weights = 0
            # data = 0.0
            # for params in self.q_net.parameters():
            #     accum_grads += torch.norm(params.grad)
            #     accum_weights += torch.norm(params.data)
            #     data += params.data.mean().item()
            # self.logger.log({"metrics/hl_gradients_model_norm": accum_grads.item(), 'timesteps': self.num_timesteps})
            # self.logger.log({"metrics/hl_weights_norm": accum_weights.item(), 'timesteps': self.num_timesteps})
            # self.logger.log({"metrics/hl_weights_mean": data, 'timesteps': self.num_timesteps})

    def act(self, tensor_obs):
        with torch.no_grad():
            if np.random.random() < self.epsilon:
                # only works for the discrete actions.
                return np.random.randint(self.n_actions)
            else:
                return torch.argmax(self.q_net(tensor_obs)).item()

    def learn(self, total_timesteps, total_episodes=None, reset_num_timesteps=True):
        episode_reward = 0.0

        num_episodes = 1

        self.num_timesteps += 1

        obs, _ = self.env.reset()
        obs = self.process_obs(obs)

        while True:

            if self.num_timesteps >= total_timesteps or num_episodes >= total_episodes:
                break

            tensor_obs = torch.tensor(obs).float().to(self.device)

            action = self.act(tensor_obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            next_obs = self.process_obs(next_obs)

            self.replay_buffer.add(obs, action, reward, next_obs, done, gamma=self.gamma)

            # Do not train until have all initial warmup.
            if self.replay_buffer.size >= self.batch_size:
                self.train()

            episode_reward += reward
            self.num_timesteps += 1

            if truncated or done:
                if num_episodes % 10 == 0:
                    print(f"Episode: {num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}")
                self.logger.log({f"metrics/episode_reward": episode_reward, 'episode': num_episodes})

                obs, _ = self.env.reset()
                obs = self.process_obs(obs)
                episode_reward = 0.0
                num_episodes += 1

            else:
                obs = next_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN agent experiment.')
    parser.add_argument('-env', type=str, choices=['gotoavoid', 'cartpole'], default='gotoavoid', help='Environment.')
    parser.add_argument('-device', type=str, default='cuda:0', help='Environment.')
    args = parser.parse_args()
    environment = args.env

    env = None
    input_shape = (3, 80, 80)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    max_total_timestps = 2e6
    max_episodes = 1e6
    config = None

    if environment == 'gotoavoid':
        env = RGBImgObsWrapper(PickupAndAvoid(12,[1,1,1,1],
                                              render_mode="rgb_array", max_steps=200, use_pick_action=False))

        config = GotoAvoid()
    elif environment == 'cartpole':
        # env = PixelObservationWrapper(CartpoleDissimilar(0.5, render_mode="rgb_array"), pixel_keys=('image',))
        env = CartpoleDissimilar(0.5, render_mode="rgb_array")
        max_total_timestps = 2e6
        max_episodes = 2000
        config = CartpoleConfig()

    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # state, _ = env.reset()
    # plt.imsave('cartpole.png', state['image'])
    # plt.imsave('cartpole_8080.png', resize_obs(state['image'], (80,80)))


    dqn = DQNAgent(input_shape, env, device=device, config=config)
    dqn.learn(max_total_timestps, total_episodes=max_episodes)

    # enable manual control for testing
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()