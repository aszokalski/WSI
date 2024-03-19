import gymnasium as gym
from wsilib.algorithms.rl.rl import Agent, HolePenaltyRewards

n = 8
is_slippery = False
map_name = f"{8}x{8}"
train_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
visible_env = gym.make(
    "FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode="human"
)

agent = Agent(
    train_env,
    n_episodes=10000,
    learning_rate=0.1,
    epsilon=1,
    epsilon_decay=0.1,
    discount_factor=0.99,
    rewards=HolePenaltyRewards,
)

agent.train()

agent.env = visible_env

agent.test(3)
