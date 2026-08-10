# /// script
# dependencies = [
#   "gymnasium>=1.3.0",
#   "fire>=0.7.1",
# ]
# ///

from collections import deque

import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

import fire
import gymnasium as gym
import numpy as np
from einops import rearrange

from light_loco_parkour import Agent, Actor, Critic, StateEncoder, Beta

# rollout collection

def collect_trajectory(
    agent,
    env,
    preprocess,
    steps_per_iter
):
    obs_list, action_list, reward_list, value_list, log_prob_list, terminated_list = ([] for _ in range(6))
    obs, episode_reward, episode_rewards = env.reset()[0], 0., []

    for _ in range(steps_per_iter):
        states = dict(obs = rearrange(preprocess(obs), 'd -> 1 1 d'))

        with torch.no_grad():
            (action, log_prob, _), (value, _) = agent(states, sample_action = True, return_log_prob = True)

        # beta action in [0, 1], mapped to cartpole action

        obs, reward, terminated, truncated, _ = env.step(action.item() > 0.5)
        done = terminated or truncated

        obs_list.append(obs)
        action_list.append(action.item())
        reward_list.append(reward)
        value_list.append(value.item())
        log_prob_list.append(log_prob.item())
        terminated_list.append(float(terminated))

        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            obs, episode_reward = env.reset()[0], 0.

    # bootstrap phantom step (done = 1) with predicted next value as reward

    with torch.no_grad():
        last_value, _ = agent.forward_critic(dict(obs = rearrange(preprocess(obs), 'd -> 1 1 d')))

    reward_list.append(last_value.item())
    value_list.append(last_value.item())
    terminated_list.append(1.)

    return dict(
        obs = obs_list,
        actions = action_list,
        rewards = reward_list,
        values = value_list,
        log_probs = log_prob_list,
        dones = terminated_list,
        episode_rewards = episode_rewards
    )

# generalized advantage estimation

def compute_gae(
    calc_gae_fn,
    rollout,
    *,
    gamma,
    gae_lambda
):
    rewards = tensor(rollout['rewards'])
    values = tensor(rollout['values'])
    masks = 1. - tensor(rollout['dones'])

    returns = calc_gae_fn(rewards, values, masks, gamma = gamma, lam = gae_lambda)[:-1]
    advantages = returns - values[:-1]

    return advantages, returns

# ppo update

def ppo_update(
    agent,
    optimizer,
    preprocess,
    rollout,
    advantages,
    returns,
    *,
    ppo_epochs,
    batch_size,
    entropy_weight,
    max_grad_norm
):
    obs = rearrange(preprocess(rollout['obs']), 't d -> t 1 d')
    actions = rearrange(tensor(rollout['actions'], dtype = torch.float32), 't -> t 1')
    old_log_probs = rearrange(tensor(rollout['log_probs'], dtype = torch.float32), 't -> t 1')
    advantages = rearrange(advantages, 't -> t 1')
    returns = rearrange(returns, 't -> t 1')

    dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    for _ in range(ppo_epochs):
        for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
            states = dict(obs = batch_obs)

            policy_loss = agent.actor_loss(
                states = states,
                actions = batch_actions,
                old_log_probs = batch_old_log_probs,
                advantages = batch_advantages,
                entropy_weight = entropy_weight
            )

            value_loss = agent.critic_loss(
                states = states,
                returns = batch_returns
            )

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

# main

def main(
    seed = 42,
    lr = 1.5e-4,
    gamma = 0.99,
    gae_lambda = 0.95,
    ppo_epochs = 4,
    steps_per_iter = 1024,
    batch_size = 64,
    entropy_weight = 0.005,
    max_grad_norm = 0.5,
    min_reward = 100.,
    max_iterations = 300,
    hl_gauss = True,
    symlog = False
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v1')
    env.reset(seed = seed)

    obs_dim = env.observation_space.shape[0]

    obs_scale = tensor([2.4, 3., 0.5, 4.])
    preprocess = lambda obs: tensor(np.asarray(obs), dtype = torch.float32) / obs_scale

    actor = Actor(
        64,
        state_encoder = StateEncoder(64, dim_state = obs_dim, num_stacked_frames = 1),
        num_actions = 1,
        action_distr = Beta(),
        depth = 2
    )

    # critic with hl gauss loss over value support, or plain mse regression

    critic = Critic(
        64,
        state_encoder = StateEncoder(64, dim_state = obs_dim, num_stacked_frames = 1),
        depth = 2,
        use_regression = not hl_gauss,
        min_value = -5.,
        max_value = 505.,
        num_bins = 256,
        use_symlog = symlog
    )

    agent = Agent(actor, critic)

    optimizer = AdamW(agent.parameters(), lr = lr)

    max_steps = env.spec.max_episode_steps
    episode_rewards = deque(maxlen = 10)

    for iteration in range(max_iterations):
        # collect trajectory

        rollout = collect_trajectory(agent, env, preprocess, steps_per_iter)

        # generalized advantage estimation

        advantages, returns = compute_gae(agent.calc_gae, rollout, gamma = gamma, gae_lambda = gae_lambda)

        # ppo update

        ppo_update(
            agent,
            optimizer,
            preprocess,
            rollout,
            advantages,
            returns,
            ppo_epochs = ppo_epochs,
            batch_size = batch_size,
            entropy_weight = entropy_weight,
            max_grad_norm = max_grad_norm
        )

        episode_rewards.extend(rollout['episode_rewards'])
        avg_reward = np.mean(episode_rewards)

        print(f'iteration {iteration:3d} | avg episode reward {avg_reward:6.2f}')

        if avg_reward > min_reward:
            print(f'balanced cartpole for {avg_reward:.2f} / {max_steps} steps at iteration {iteration}')
            break

if __name__ == '__main__':
    fire.Fire(main)
