import gymnasium as gym
from  envs.babyai.gotoavoid import PickupAndAvoid
#env = gym.make("MiniGrid-Empty-5x5-v1", render_mode="human")
#env = gym.make("BabyAI-OneRoomS20-v0", render_mode="human")

import numpy as np

# First is 
vector_to_reward = [1,0,0,0]

env = PickupAndAvoid(12, vector_to_reward, render_mode='rgb_array')

# import metaworld
# import random

observation, info = env.reset(seed=42)

if __name__ == '__main__':
    total_reward = []
    total_terminated = []

    for _ in range(10_000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward.append(reward)
        total_terminated.append(terminated)
    
        # API 26
        if terminated or truncated:
            pass
            # API 21
            # observation = env.reset()
            # API 26
            #observation, info = env.reset()
        env.render()
    env.close()

    #print(f'observation {observation}')
    print(f'reward {np.sum(total_reward)}')
    print(f'terminated {np.sum(total_terminated)}')
