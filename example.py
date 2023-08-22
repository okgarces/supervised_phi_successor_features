import gymnasium as gym
from  envs.babyai.gotoavoid import PickupAndAvoid
#env = gym.make("MiniGrid-Empty-5x5-v1", render_mode="human")
#env = gym.make("BabyAI-OneRoomS20-v0", render_mode="human")

# First is 
vector_to_reward = [1,0,0,0]

env = PickupAndAvoid(10, vector_to_reward, render_mode='human')

import metaworld
import random

observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # API 26
    if terminated or truncated:
        # API 21
        observation = env.reset()
        # API 26
        #observation, info = env.reset()
    print(f'observation {observation}')
    print(f'reward {reward}')
    print(f'terminated {terminated}')
    env.render()
env.close()
