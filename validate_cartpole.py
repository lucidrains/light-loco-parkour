# /// script
# dependencies = [
#   "env-ssl-wrapper>=0.1.0",
#   "gymnasium>=1.3.0",
#   "fire>=0.7.1",
# ]
# ///

# ppo sanity check for the beta mean-concentration action distribution.
# the env goes through env-ssl-wrapper's compose_env, so the identical training
# loop runs on any simulator env - gymnasium (default CartPole-v1), isaacgym
# (--env_id isaacgym, a dependency-free mock, or module:attr for a real env
# factory), etc.

from collections import deque

import torch
from torch import tensor, cat
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

import fire
import gymnasium as gym
import numpy as np
from einops import rearrange

from env_ssl_wrapper import compose_env

from light_loco_parkour import Agent, Actor, Critic, StateEncoder, Beta

# environment - one line normalizes any sim env to the same torch-native,
# vectorized interface

def build_env(
    env_id,
    num_envs = 1,
    device = 'cpu'
):
    if env_id == 'isaacgym':
        from env_ssl_wrapper.mocks import IsaacMockEnv
        env = IsaacMockEnv()
    elif ':' in env_id:
        import importlib
        module_path, attr = env_id.split(':', 1)
        factory = getattr(importlib.import_module(module_path), attr)
        env = factory() if callable(factory) else factory
    else:
        env = gym.make_vec(env_id, num_envs = num_envs) if num_envs > 1 else gym.make(env_id)

    return compose_env(
        env,
        ('tensor', dict(device = device)),   # obs / rewards / dones as torch
        'flatten_obs',                       # dict obs (isaacgym) -> flat vector
        'auto_batch',                        # single envs get a leading batch dim
        'done_tracker',                      # per-env episode lengths, active mask, all_done
        ('action_transform', dict(auto = True)) # beta actions in [0, 1] -> env action bounds
    )

def reset_env(env, seed = None):
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass

    return env.reset()

def build_preprocess(obs_scale):
    # divide by a per-dimension scale when it matches the obs dim (cartpole),
    # otherwise leave the obs untouched (any other env)

    if obs_scale is None:
        return lambda obs: obs

    def preprocess(obs):
        if obs.shape[-1] != obs_scale.numel():
            return obs
        return obs / obs_scale

    return preprocess

def to_env_action(action, env, num_envs):
    # beta actions live in [0, 1]; binary discrete envs map them to {0, 1},
    # continuous envs get them rescaled to their bounds by the action_transform wrapper

    if not hasattr(env.action_space, 'n'):
        return action

    assert env.action_space.n == 2, 'the beta action distribution is one dimensional, so only binary discrete action spaces are supported'

    action = (action > 0.5).long().reshape(-1)
    return action.item() if num_envs == 1 else action

# rollout collection

def collect_trajectory(
    agent,
    env,
    preprocess,
    steps_per_iter
):
    num_envs = env.num_envs

    obs, _ = reset_env(env)
    obs = preprocess(obs)

    episode_rewards = torch.zeros(num_envs)
    episode_reward_list = []

    obs_list, action_list, reward_list, value_list, log_prob_list, done_list, mask_list = ([] for _ in range(7))

    for _ in range(steps_per_iter):
        active = torch.from_numpy(env.active_mask).bool()

        # store the pre-step obs, so every transition stays aligned:
        # (s_t, a_t, log pi(a_t | s_t), V(s_t))

        obs_before = obs

        states = dict(obs = rearrange(obs_before, 'b d -> b 1 d'))

        with torch.no_grad():
            (action, log_prob, _), (value, _) = agent(states, sample_action = True, return_log_prob = True)

        action = rearrange(action, 'b 1 d -> b d')
        log_prob = rearrange(log_prob, 'b 1 -> b')
        value = rearrange(value, 'b 1 -> b')

        raw_action = action
        action = to_env_action(action, env, num_envs)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated | truncated

        # envs that finish are reset only once all are done, so their later steps
        # are garbage; only envs that were running before the step count

        reward = reward * active

        obs = preprocess(obs)

        obs_list.append(obs_before)
        action_list.append(raw_action)
        reward_list.append(reward)
        value_list.append(value)
        log_prob_list.append(log_prob)
        done_list.append(done)
        mask_list.append(active)

        episode_rewards += reward

        for env_ind in torch.where(done & active)[0].tolist():
            episode_reward_list.append(episode_rewards[env_ind].item())
            episode_rewards[env_ind] = 0.

        if env.all_done:
            obs, _ = reset_env(env)
            obs = preprocess(obs)

    # bootstrap the final obs with its predicted value, zeroed for envs already done

    with torch.no_grad():
        last_value, _ = agent.forward_critic(dict(obs = rearrange(obs, 'b d -> b 1 d')))

    bootstrap = rearrange(last_value, 'b 1 -> b') * (~done).float()

    value_list.append(bootstrap)

    return dict(
        obs = cat([rearrange(o, 'b d -> b 1 d') for o in obs_list]),
        actions = cat([rearrange(a, 'b d -> b 1 d') for a in action_list]),
        rewards = cat([r.unsqueeze(1) for r in reward_list], dim = 1),
        values = cat([v.unsqueeze(1) for v in value_list], dim = 1),
        log_probs = cat([lp.unsqueeze(1) for lp in log_prob_list], dim = 1),
        dones = cat([d.unsqueeze(1) for d in done_list], dim = 1),
        mask = cat([m.unsqueeze(1) for m in mask_list], dim = 1),
        episode_rewards = episode_reward_list
    )

# generalized advantage estimation

def compute_gae(
    calc_gae_fn,
    rollout,
    *,
    gamma,
    gae_lambda
):
    rewards = rollout['rewards']
    values = rollout['values']
    masks = 1. - rollout['dones'].float()

    returns = calc_gae_fn(rewards, values, masks, gamma = gamma, lam = gae_lambda)
    advantages = returns - values[..., :-1]

    return advantages, returns

# ppo update

def ppo_update(
    agent,
    optimizer,
    rollout,
    *,
    ppo_epochs,
    batch_size,
    entropy_weight,
    max_grad_norm
):
    obs = rearrange(rollout['obs'], 'b t d -> (b t) 1 d')
    actions = rearrange(rollout['actions'], 'b t d -> (b t) d')
    old_log_probs = rearrange(rollout['log_probs'], 'b t -> (b t) 1')
    advantages = rearrange(rollout['advantages'], 'b t -> (b t) 1')
    returns = rearrange(rollout['returns'], 'b t -> (b t) 1')
    mask = rearrange(rollout['mask'], 'b t -> (b t) 1')

    dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns, mask)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    for _ in range(ppo_epochs):
        for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_mask in dataloader:
            states = dict(obs = batch_obs)

            policy_loss = agent.actor_loss(
                states = states,
                actions = batch_actions,
                old_log_probs = batch_old_log_probs,
                advantages = batch_advantages,
                entropy_weight = entropy_weight,
                mask = batch_mask
            )

            value_loss = agent.critic_loss(
                states = states,
                returns = batch_returns,
                mask = batch_mask
            )

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

# deterministic evaluation - the mean episode reward of the current policy,
# collected without exploration noise, for a stable view of learning

def evaluate(
    agent,
    env,
    preprocess,
    num_episodes = 32,
    max_steps = 4096
):
    num_envs = env.num_envs

    obs, _ = reset_env(env)
    obs = preprocess(obs)

    episode_rewards = torch.zeros(num_envs)
    episode_reward_list = []

    for _ in range(max_steps):
        active = torch.from_numpy(env.active_mask).bool()

        states = dict(obs = rearrange(obs, 'b d -> b 1 d'))

        with torch.no_grad():
            (action, _), _ = agent(states, deterministic = True)

        action = to_env_action(rearrange(action, 'b 1 d -> b d'), env, num_envs)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated | truncated

        episode_rewards += reward * active

        for env_ind in torch.where(done & active)[0].tolist():
            episode_reward_list.append(episode_rewards[env_ind].item())
            episode_rewards[env_ind] = 0.

        if env.all_done:
            obs, _ = reset_env(env)
            obs = preprocess(obs)

        if len(episode_reward_list) >= num_episodes:
            break

    return np.mean(episode_reward_list) if episode_reward_list else 0.

# main

def main(
    env_id = 'CartPole-v1', # gymnasium env id, 'isaacgym' for the mock isaac env, or 'module:attr' for any env factory
    num_envs = 1, # parallel envs (gymnasium vectorization); use more for faster data collection
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
    eval_every = 0, # evaluate the deterministic policy every N iterations (0 to disable)
    hl_gauss = True,
    symlog = False,
    min_conc = 0.,
    obs_scale = (2.4, 3., 0.5, 4.)
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = build_env(env_id, num_envs = num_envs)

    obs, _ = reset_env(env, seed = seed)
    obs_dim = obs.shape[-1]

    # num_actions resolved from the env's action space (unbatched)

    if hasattr(env.action_space, 'n'):
        num_actions = 1
    else:
        num_actions = env.action_space.shape[0]

    preprocess = build_preprocess(tensor(obs_scale) if obs_scale is not None else None)

    actor = Actor(
        64,
        state_encoder = StateEncoder(64, dim_state = obs_dim, num_stacked_frames = 1),
        num_actions = num_actions,
        action_distr = Beta(min_conc = min_conc),
        depth = 2
    )

    # critic with hl gauss loss over a value support, or plain mse regression

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

    max_steps = getattr(getattr(env, 'spec', None), 'max_episode_steps', None) or getattr(env, 'max_steps', 500)
    episode_rewards = deque(maxlen = 10)

    print(f'training on {env_id} | {env.num_envs} parallel envs | {num_actions} action(s) | obs dim {obs_dim}')

    if eval_every > 0:
        print(f'evaluation | deterministic avg episode reward {evaluate(agent, env, preprocess):7.2f}')

    for iteration in range(max_iterations):
        # collect trajectory

        rollout = collect_trajectory(agent, env, preprocess, steps_per_iter)

        # generalized advantage estimation

        advantages, returns = compute_gae(agent.calc_gae, rollout, gamma = gamma, gae_lambda = gae_lambda)

        rollout.update(advantages = advantages, returns = returns)

        # ppo update

        ppo_update(
            agent,
            optimizer,
            rollout,
            ppo_epochs = ppo_epochs,
            batch_size = batch_size,
            entropy_weight = entropy_weight,
            max_grad_norm = max_grad_norm
        )

        episode_rewards.extend(rollout['episode_rewards'])
        avg_reward = np.mean(episode_rewards)

        print(f'iteration {iteration:3d} | avg episode reward {avg_reward:6.2f}')

        if eval_every > 0 and (iteration + 1) % eval_every == 0:
            print(f'evaluation | deterministic avg episode reward {evaluate(agent, env, preprocess):7.2f}')

        if avg_reward > min_reward:
            print(f'balanced cartpole for {avg_reward:.2f} / {max_steps} steps at iteration {iteration}')
            break

if __name__ == '__main__':
    fire.Fire(main)
