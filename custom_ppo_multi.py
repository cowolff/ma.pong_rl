import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from multi_pong import PongEnv, MaxEpisodeStepsWrapper
from progressbar import progressbar


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    # Set the variables with default values
    exp_name = os.path.basename(__file__).rstrip(".py")
    learning_rate = 2.5e-4
    seed = 1
    total_timesteps = 16000000
    torch_deterministic = True
    cuda = False
    wandb_project_name = "ppo-implementation-details"
    wandb_entity = None
    capture_video = False

    # Algorithm specific arguments
    num_envs = 2
    num_steps = 128
    anneal_lr = True
    gae = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None

    # Additional calculations based on set variables
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    run_name = f"Pong__{exp_name}__{seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(seed) for i in range(num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    env = PongEnv(width=20, height=20)
    env = MaxEpisodeStepsWrapper(env, 2048)

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    num_agents = len(env.agents)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_agents) + env.observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_agents) + env.action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_agents)).to(device)
    rewards = torch.zeros((num_steps, num_agents)).to(device)
    dones = torch.zeros((num_steps, num_agents)).to(device)
    values = torch.zeros((num_steps, num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = env.reset()
    next_obs = np.array([next_obs[agent] for agent in env.agents])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_agents).to(device)
    num_updates = total_timesteps // batch_size

    rewards_record = []
    lengths_record = []

    episode_rewards = np.array([])
    timesteps = 0

    for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done
            timesteps += 1

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            action = action.cpu().numpy()
            action = {agent: action[i] for i, agent in enumerate(env.agents)}

            next_obs, reward, done, info = env.step(action)
            

            reward = np.array([reward[agent] for agent in env.agents])
            episode_rewards = np.append(episode_rewards, reward)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = np.array([next_obs[agent] for agent in env.agents])
            done = np.array([done[agent] for agent in env.agents])
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if any(done):
                env.reset()
                current_reward = episode_rewards.sum()
                rewards_record.append(current_reward)
                lengths_record.append(timesteps)
                writer.add_scalar("charts/episodic_return", sum(rewards_record[-1 * min(100, len(rewards_record)):]) / min(100, len(rewards_record)), global_step)
                writer.add_scalar("charts/episodic_length", sum(lengths_record[-1 * min(100, len(lengths_record)):]) / min(100, len(lengths_record)), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print("Episode:", len(rewards_record), "Return:", current_reward, "Average Return:", sum(rewards_record[-1 * min(100, len(rewards_record)):]) / min(100, len(rewards_record)), "Episode Length:", timesteps)
                episode_rewards = np.array([])
                timesteps = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    torch.save(agent.state_dict(), f"models/{run_name}.pt")
    env.close()
    writer.close()