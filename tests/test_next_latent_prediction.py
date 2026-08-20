import torch
from torch import tensor
from torch.optim import Adam
from torch.nn import Linear

from light_loco_parkour import (
    Actor,
    Critic,
    Agent,
    StateEncoder,
    Gaussian,
    NextLatentPredictionWrapper
)

# helpers

def integrator_rollout(agent, batch, time, device = 'cpu'):
    # integrator task: pos_{t + 1} = pos_t + dt * action_t, tracking a per-episode target
    # dt is large so adjacent latents differ substantially, keeping the next latent
    # prediction nontrivial; reward is the gaussian kernel on the tracking error

    dt = 1.5

    actor, critic = agent.actor, agent.critic

    target = torch.rand(batch, 1, device = device) * 4. - 2.
    pos = torch.zeros(batch, 1, device = device)

    states = torch.zeros(batch, time, 2, device = device)
    actions = torch.zeros(batch, time, 1, device = device)
    old_log_probs = torch.zeros(batch, time, device = device)
    rewards = torch.zeros(batch, time, device = device)

    with torch.no_grad():
        for t in range(time):
            states[:, t] = torch.cat((pos, target), dim = -1)

            action, log_prob, _ = actor((states[:, :t + 1],), sample_action = True, return_log_prob = True)
            action = action[:, t]
            log_prob = log_prob[:, t]

            pos = pos + dt * action

            actions[:, t] = action
            old_log_probs[:, t] = log_prob
            rewards[:, t] = (-(pos - target).square()).exp().flatten()

        values, _ = critic((states,))

    returns = agent.calc_gae(rewards, values, torch.ones(batch, time))

    return states, actions, old_log_probs, rewards, returns, values

def rotation_dynamics_sequences(batch, time, dim, num_blocks, action_scale = 0.3):
    # a latent that rotates 90 degrees in each 2d block every step, plus an action
    # push - adjacent latents are genuinely different, so the prediction is nontrivial

    rotation = torch.zeros(dim, dim)

    for i in range(num_blocks):
        rotation[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[0., -1.], [1., 0.]])

    latents = torch.zeros(batch, time, dim)
    actions = torch.randn(batch, time, num_blocks * 2)

    latent = torch.randn(batch, dim)

    for t in range(time):
        latents[:, t] = latent
        latent = 0.9 * (rotation @ latent.unsqueeze(-1)).squeeze(-1) + action_scale * actions[:, t]

    return latents, actions

# the wrapper enacts only on (batch, time, ...) inputs with at least two time steps, and only during training

def test_next_latent_prediction_wrapper_guards():
    torch.manual_seed(42)

    wrapper = NextLatentPredictionWrapper(Linear(16, 8), dim = 16, dim_action = 8)

    # no time dimension / single time step - skipped

    out = wrapper(torch.randn(4, 16))
    assert out.shape == (4, 8)
    assert wrapper.next_latent_prediction_loss is None

    out = wrapper(torch.randn(4, 1, 16))
    assert wrapper.next_latent_prediction_loss is None

    # evaluation mode - skipped; training mode - enacted, off via the per-call kwarg

    wrapper.eval()
    out = wrapper(torch.randn(4, 8, 16))
    assert wrapper.next_latent_prediction_loss is None

    wrapper.train()
    out = wrapper(torch.randn(4, 8, 16))
    assert wrapper.next_latent_prediction_loss is not None

    out = wrapper(torch.randn(4, 8, 16), next_latent_prediction = False)
    assert wrapper.next_latent_prediction_loss is None

# the actor's head is wrapped by default; `next_latent_prediction = False` turns it off

def test_next_latent_prediction_actor_switch():
    torch.manual_seed(42)

    actor = Actor(64, state_encoder = StateEncoder(64, dim_state = 2), action_distr = Gaussian())
    off_actor = Actor(64, state_encoder = StateEncoder(64, dim_state = 2), action_distr = Gaussian(), next_latent_prediction = False)

    states = torch.randn(4, 3, 2)

    dist, _ = actor((states,), return_action_distr = True)
    assert dist.batch_shape == (4, 3, 21)
    assert actor.next_latent_prediction_loss is not None
    assert actor.next_latent_prediction_loss.ndim == 0

    off_actor((states,), return_action_distr = True)
    assert off_actor.next_latent_prediction_loss.item() == 0.

# the transition model learns a genuinely nontrivial next latent prediction. the
# dynamics are residual (nlp-style), so convergence is fast - the loop terminates
# early as soon as the loss hits the threshold, and must do so within 200 iterations

def test_next_latent_prediction_dynamics_learns_transition():
    torch.manual_seed(42)

    dim, num_blocks = 16, 8

    wrapper = NextLatentPredictionWrapper(Linear(dim, 8), dim = dim, dim_action = num_blocks * 2)

    latents, actions = rotation_dynamics_sequences(8, 8, dim, num_blocks)

    optimizer = Adam([p for name, p in wrapper.named_parameters() if not name.startswith('head')], lr = 1e-2)

    losses = []

    for _ in range(200):
        wrapper(latents, action = actions)

        loss = wrapper.next_latent_prediction_loss
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if losses[-1] < 0.05:
            break

    assert losses[0] > 0.5, f'initial next latent prediction loss should be nontrivial, got {losses[0]}'
    assert losses[-1] < 0.05, f'transition model did not converge: {losses[0]} -> {losses[-1]}'
    assert len(losses) < 200, f'transition model did not converge quickly, ran all {len(losses)} iterations'

# ppo with next latent prediction (spr / nlp) turned on - the policy converges on
# the integrator tracking task, and the next latent prediction loss stays well-behaved

def test_next_latent_prediction_converges():
    torch.manual_seed(42)

    dim = 64
    batch, time = 64, 16
    steps = 150

    actor = Actor(
        dim,
        state_encoder = StateEncoder(dim, dim_state = 2),
        num_actions = 1,
        action_distr = Gaussian(),
        next_latent_prediction = dict()
    )

    critic = Critic(dim, state_encoder = StateEncoder(dim, dim_state = 2))

    agent = Agent(actor, critic)
    optimizer = Adam(agent.parameters(), lr = 1e-3)

    states, actions, old_log_probs, rewards, returns, values = integrator_rollout(agent, batch, time)

    advantages = returns - values
    initial_reward = rewards.mean().item()

    spr_losses = []

    for _ in range(steps):
        states, actions, old_log_probs, rewards, returns, values = integrator_rollout(agent, batch, time)
        advantages = returns - values

        policy_loss = agent.actor_loss((states,), actions, old_log_probs, advantages, entropy_weight = 0.01)
        value_loss = agent.critic_loss((states,), returns)

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()

        spr_losses.append(actor.next_latent_prediction_loss.item())

    final_spr_loss = actor.next_latent_prediction_loss.item()
    final_reward = rewards.mean().item()

    # the next latent prediction loss must stay finite and small throughout training

    assert torch.isfinite(tensor(spr_losses)).all()
    assert final_spr_loss < 0.1, f'next latent prediction loss blew up: {final_spr_loss}'

    # the policy must have learned to track the commanded velocity

    assert final_reward > 0.7, f'policy did not converge on the tracking task: {initial_reward} -> {final_reward}'
    assert final_reward > initial_reward
