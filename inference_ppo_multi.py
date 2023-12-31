from custom_ppo_multi import Agent
from multi_pong import PongEnv, MaxEpisodeStepsWrapper
import gym
import torch
import numpy as np
import time
import os

seed = 1
num_envs = 1

env = PongEnv(width=20, height=20)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

agent = Agent(env)
agent.load_state_dict(torch.load('models/Pong__custom_ppo_multi__1__1703992910.pt'))
agent.eval()

env = PongEnv(width=20, height=20)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

for i in range(10):
    obs = env.reset()
    obs = np.array([obs[agent] for agent in env.agents])
    obs = torch.Tensor(obs)
    done = [False]
    while not any(done):
        action, logprob, _, value = agent.get_action_and_value(obs)
        action = action.cpu().numpy()
        action = {agent: action[i] for i, agent in enumerate(env.agents)}
        env.render()

        obs, rewards, done, info = env.step(action)
        print(obs, "\n", rewards, "\n")
        obs = np.array([obs[agent] for agent in env.agents])
        obs = torch.Tensor(obs)
        done = [done[agent] for agent in env.agents]
        time.sleep(0.1)
        if any(done):
            break