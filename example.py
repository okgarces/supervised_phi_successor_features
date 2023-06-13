import gymnasium as gym
env = gym.make("CartPole-v1", render_mode='human')

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    print(f'observation {observation}')
    print(f'reward {reward}')
    print(f'terminated {terminated}')
    env.render()
env.close()
