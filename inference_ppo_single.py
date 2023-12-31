from custom_ppo_single import Agent
from single_pong_multiple_balls import PongEnv, MaxEpisodeStepsWrapper
import gym
import torch
import numpy as np
import time
import os

seed = 1
num_envs = 1

def make_env(seed):
    def thunk():
        env = PongEnv(width=20, height=20)
        env = MaxEpisodeStepsWrapper(env, 2048)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv(
        [make_env(seed) for i in range(num_envs)]
    )
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
agent = Agent(envs)
agent.load_state_dict(torch.load('models/Pong__custom_ppo_multi__1__1703944331.pt'))
agent.eval()

env = PongEnv(width=20, height=20)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

for i in range(10):
    obs = torch.Tensor(np.expand_dims(env.reset(), axis=0))
    done = False
    while not done:
        os.system("clear")
        action, logprob, _, value = agent.get_action_and_value(obs)
        env.render()
        obs, rewards, done, info = env.step(action.numpy()[0])
        obs = torch.Tensor(np.expand_dims(obs, axis=0))
        time.sleep(0.1)
        if done:
            break