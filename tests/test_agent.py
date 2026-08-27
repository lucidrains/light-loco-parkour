import pytest
param = pytest.mark.parametrize

import torch
from torch import tensor
from torch.nn import Module

from torch_einops_utils import z_score

from x_mlps_pytorch import create_mlp
from light_loco_parkour.light_loco_parkour import (
    Actor,
    Critic,
    Agent,
    State,
    Gaussian,
    Beta,
    StateEncoder,
    RewardHyperParams,
    RewardShapingWrapper,
    StatefulReward,
    reward_linear_velocity_tracking,
    reward_angular_velocity_tracking,
    reward_velocity_slack,
    reward_heading_error,
    Discriminator,
    MotionPrior,
    exists
)

# helpers

@param('skill_groups', [
    1,
    tensor(1),
    tensor([1])
])
def test_agent(skill_groups):
    student_actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5, use_rnn = True))
    teacher_actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_skill_groups = 2, action_distr = Gaussian())

    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_skill_groups = 2)

    images = torch.randn(1, 3, 2, 2)
    proprio = torch.randn(1, 3, 5)

    decoder = create_mlp(16, dim_in = 512, dim_out = 17 * 11, depth = 2)
    privileged_target = torch.randn(1, 3, 17 * 11)

    pred_actions, next_hidden = student_actor((images, proprio))

    (pred_actions, next_hidden), aux_loss = student_actor((images, proprio), aux_decoder = decoder, aux_decoder_target = privileged_target)

    teacher_mean_actions, _ = teacher_actor((images, proprio), skill_groups = skill_groups, deterministic = True)

    values, _ = critic((images, proprio), skill_groups = skill_groups)

    assert pred_actions.shape == (1, 3, 21, 2)
    assert teacher_mean_actions.shape == (1, 3, 21)
    assert values.shape == (1, 3)

    (aux_loss.sum() + teacher_mean_actions.sum() + values.sum()).backward()

@param('distr_cls, param_type, pos_fn', [
    (Gaussian, 'log_std', 'exp'),
    (Gaussian, 'log_var', 'softplus'),
    (Beta, 'softplus', 'softplus'),
    (Beta, 'exp', 'exp'),
])
def test_action_distributions(distr_cls, param_type, pos_fn):
    if distr_cls == Gaussian:
        distr = Gaussian(param_type = param_type, pos_fn = pos_fn)
    else:
        distr = Beta(pos_fn = pos_fn)

    actor = Actor(
        512,
        state_encoder = StateEncoder(512, dim_state = 4 + 5, use_rnn = True),
        action_distr = distr
    )

    images = torch.randn(2, 3, 2, 2)
    proprio = torch.randn(2, 3, 5)

    # deterministic mode (returns mean directly for distillation / deployment)

    mean_actions, _ = actor((images, proprio), deterministic = True)
    assert mean_actions.shape == (2, 3, 21)

    # return distribution

    dist, _ = actor((images, proprio), return_action_distr = True)
    assert dist.batch_shape == (2, 3, 21)

    # sample action and get log_prob for ppo

    sampled_actions, log_prob, _ = actor((images, proprio), sample_action = True, return_log_prob = True)
    assert sampled_actions.shape == (2, 3, 21)
    assert log_prob.shape == (2, 3)

    (mean_actions.sum() + log_prob.sum()).backward()

@param('has_aux_decoder', [False, True])
def test_distillation_wrapper(has_aux_decoder):
    from light_loco_parkour import DistillationWrapper, Agent

    aux_decoder = create_mlp(10, dim_in = 512, depth = 2) if has_aux_decoder else None

    student_encoder = StateEncoder(512, dim_state = 4 + 5, use_rnn = True)
    teacher_encoder = StateEncoder(512, dim_state = 4 + 5 + 10)

    student_actor = Actor(512, state_encoder = student_encoder, aux_decoder = aux_decoder)
    teacher_actor = Actor(512, state_encoder = teacher_encoder, num_skill_groups = 2)
    critic = Critic(512, state_encoder = teacher_encoder, num_skill_groups = 2)

    distill_kwargs = dict(
        student_state_keys = ('images', 'proprio'),
        teacher_state_keys = ('images', 'proprio', 'privileged_info')
    )

    if has_aux_decoder:
        distill_kwargs.update(
            privileged_state_key = 'privileged_info',
            aux_loss_weight = 0.5
        )

    distill = DistillationWrapper(student_actor, teacher_actor, **distill_kwargs)

    states = dict(
        images = torch.randn(2, 3, 2, 2),
        proprio = torch.randn(2, 3, 5),
        privileged_info = torch.randn(2, 3, 10)
    )

    # reduced loss

    loss = distill(states, teacher_skill_groups = 1)
    assert loss.ndim == 0
    loss.backward()

    # unreduced loss

    unreduced_loss = distill(states, teacher_skill_groups = 1, return_unreduced = True)
    assert unreduced_loss.shape == (2, 3)

    # loss breakdown

    total_loss, (distill_loss, aux_loss) = distill(states, teacher_skill_groups = 1, return_loss_breakdown = True)
    assert total_loss.ndim == 0
    assert distill_loss.ndim == 0

    # soft weighting

    weights = torch.rand(2, 3)
    weighted_loss = distill(states, weights = weights, teacher_skill_groups = 1)
    assert weighted_loss.ndim == 0

    # variable length sequence support

    lens = tensor([2, 3])
    var_len_loss = distill(states, lens = lens, teacher_skill_groups = 1)
    assert var_len_loss.ndim == 0

def test_cascading_distillation():
    from light_loco_parkour import DistillationWrapper

    # 3 networks in a hierarchy with cascading privileged information:
    # top-most teacher (highest privilege: images, proprio, depth, scan) -> mid teacher (mid privilege: images, proprio, depth) -> final student (lowest privilege: images, proprio)

    top_teacher_encoder = StateEncoder(512, dim_state = 4 + 5 + 10 + 8)
    mid_teacher_encoder = StateEncoder(512, dim_state = 4 + 5 + 10)
    student_encoder = StateEncoder(512, dim_state = 4 + 5, use_rnn = True)

    top_teacher_actor = Actor(512, state_encoder = top_teacher_encoder, num_skill_groups = 3)
    mid_teacher_actor = Actor(512, state_encoder = mid_teacher_encoder, num_skill_groups = 2)
    student_actor = Actor(512, state_encoder = student_encoder)

    # first distillation stage: top teacher -> mid teacher

    distill_top_to_mid = DistillationWrapper(
        student = mid_teacher_actor,
        teacher = top_teacher_actor,
        student_state_keys = ('images', 'proprio', 'depth'),
        teacher_state_keys = ('images', 'proprio', 'depth', 'scan')
    )

    # second distillation stage: mid teacher -> student

    distill_mid_to_student = DistillationWrapper(
        student = student_actor,
        teacher = mid_teacher_actor,
        student_state_keys = ('images', 'proprio'),
        teacher_state_keys = ('images', 'proprio', 'depth')
    )

    states = dict(
        images = torch.randn(2, 3, 2, 2),
        proprio = torch.randn(2, 3, 5),
        depth = torch.randn(2, 3, 10),
        scan = torch.randn(2, 3, 8)
    )

    # stage 1 loss (top -> mid)

    loss_stage1 = distill_top_to_mid(
        states,
        student_skill_groups = 1,
        teacher_skill_groups = 2
    )

    assert loss_stage1.ndim == 0
    loss_stage1.backward()

    # stage 2 loss (mid -> student)

    loss_stage2 = distill_mid_to_student(
        states,
        teacher_skill_groups = 1
    )

    assert loss_stage2.ndim == 0
    loss_stage2.backward()

@param('use_dict', [True, False])
def test_film_proprioception_conditioning(use_dict):
    dim_proprio = 5
    dim_depth = 12

    if use_dict:
        cond_key = 'proprio'
        states = dict(
            depth = torch.randn(2, 3, 12),
            proprio = torch.randn(2, 3, 5)
        )
    else:
        cond_key = 1
        states = (
            torch.randn(2, 3, 12), # depth at index 0
            torch.randn(2, 3, 5)   # proprio at index 1
        )

    encoder = StateEncoder(
        512,
        dim_state = dim_depth,
        cond_key = cond_key,
        dim_cond = dim_proprio,
        use_rnn = True
    )

    actor = Actor(512, state_encoder = encoder)

    action_out, next_hidden = actor(states)

    assert action_out.shape == (2, 3, 21, 2)
    action_out.sum().backward()

@param('use_dict', [True, False])
def test_agent_routing(use_dict):
    from light_loco_parkour import Agent

    actor_encoder = StateEncoder(512, dim_state = 4 + 5)
    critic_encoder = StateEncoder(512, dim_state = 4 + 5 + 10)

    actor = Actor(512, state_encoder = actor_encoder)
    critic = Critic(512, state_encoder = critic_encoder)

    if use_dict:
        actor_state_keys = ('images', 'proprio')
        critic_state_keys = ('images', 'proprio', 'privileged_info')
        states = dict(
            images = torch.randn(2, 3, 2, 2),
            proprio = torch.randn(2, 3, 5),
            privileged_info = torch.randn(2, 3, 10),
            unused_sensor = torch.randn(2, 3, 16)
        )
    else:
        actor_state_keys = (0, 1)
        critic_state_keys = (0, 1, 2)
        states = (
            torch.randn(2, 3, 2, 2),
            torch.randn(2, 3, 5),
            torch.randn(2, 3, 10),
            torch.randn(2, 3, 16)
        )

    agent = Agent(
        actor,
        critic,
        actor_state_keys = actor_state_keys,
        critic_state_keys = critic_state_keys
    )

    (actions, _), (values, _) = agent(states)

    assert actions.shape == (2, 3, 21, 2)
    assert values.shape == (2, 3)

    (actions.sum() + values.sum()).backward()

def test_ppo_learning():
    actor = Actor(
        512,
        state_encoder = StateEncoder(512, dim_state = 4 + 5),
        action_distr = Gaussian()
    )

    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5))

    agent = Agent(actor, critic)

    images = torch.randn(2, 8, 2, 2)
    proprio = torch.randn(2, 8, 5)
    actions = torch.randn(2, 8, 21)
    old_log_probs = torch.randn(2, 8)
    advantages = torch.randn(2, 8)
    returns = torch.rand(2, 8)

    policy_loss = agent.actor_loss((images, proprio), actions, old_log_probs, advantages)
    value_loss = agent.critic_loss((images, proprio), returns)
    loss = policy_loss + value_loss

    loss.backward()

    assert all(exists(p.grad) for p in actor.parameters())
    assert all(exists(p.grad) for p in critic.parameters())

def mock_state(batch = 4, d = 21, k = 8, l = 6):
    return State(
        linear_velocity = torch.randn(batch, 3),
        angular_velocity = torch.randn(batch, 3),
        projected_gravity = torch.randn(batch, 3),
        commanded_velocity = torch.randn(batch, 3),
        joint_limit_flags = torch.rand(batch, d) > 0.9,
        contact_forces = torch.rand(batch, l) * 2.,
        foot_contact = torch.rand(batch, 2) > 0.5,
        foot_heights = torch.rand(batch, 2),
        foot_ray_hit_heights = torch.rand(batch, 2, k) * 0.3,
        foot_acceleration = torch.rand(batch, 2, 3) * 40.,
        heading_error = torch.rand(batch),
        action_rate = torch.randn(batch, d)
    )

def test_reward_shaping():
    wrapper = RewardShapingWrapper(reward_hparams = RewardHyperParams())

    state = mock_state()
    state.linear_velocity.requires_grad_(True)

    reward = wrapper(state)
    assert reward.shape == (4,)
    reward.sum().backward()
    assert exists(state.linear_velocity.grad)

    # the equation 5 decay filter carries over between steps, and resets
    assert not torch.allclose(wrapper(state), reward)

    wrapper.reset_()
    assert torch.allclose(wrapper(state), reward)

def test_reward_weights_and_registration():
    state = mock_state()
    state.linear_velocity[:, :2] = state.commanded_velocity[:, :2]
    state.foot_acceleration.zero_()

    # per-reward weight overrides, keyed by function name
    wrapper = RewardShapingWrapper(
        reward_fns = ((reward_linear_velocity_tracking, 2.0),),
        reward_weights = {'reward_linear_velocity_tracking': 3.0},
        reward_hparams = RewardHyperParams()
    )
    assert torch.allclose(wrapper(state), torch.full((4,), 3.0))

    # custom reward functions can be registered at runtime
    def reward_base_height(state, hparams):
        return state.heading_error

    wrapper = RewardShapingWrapper(reward_fns = (), reward_hparams = RewardHyperParams())
    wrapper.add_reward_function_(reward_base_height, 5.)
    assert torch.allclose(wrapper(state), state.heading_error * 5.)

    # ... and stateful ones, with reset
    class AccumulatingReward(StatefulReward):
        def __init__(self):
            super().__init__()
            self.accum = None

        def reset_(self):
            self.accum = None

        def forward(self, state, hparams):
            new = state.heading_error
            self.accum = new if not exists(self.accum) else self.accum + new
            return self.accum

    wrapper = RewardShapingWrapper(
        reward_fns = (),
        stateful_reward_fns = ((AccumulatingReward(), 1.0),),
        reward_hparams = RewardHyperParams()
    )

    reward = wrapper(state)
    assert torch.allclose(wrapper(state), reward * 2.)

    wrapper.reset_()
    assert torch.allclose(wrapper(state), reward)

def test_agent_combined_forward_routes_skill_groups_to_critic():
    # the combined agent forward must pass the skill group conditioning to the critic, matching
    # a standalone critic call with the same group, and differ across groups

    actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_skill_groups = 3, action_distr = Gaussian())
    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_skill_groups = 3)

    agent = Agent(actor, critic)

    states = (torch.randn(2, 3, 4), torch.randn(2, 3, 5))

    (actions, log_probs, _), (values, _) = agent(states, skill_groups = 2, sample_action = True, return_log_prob = True)

    assert actions.shape == (2, 3, 21)
    assert log_probs.shape == (2, 3)
    assert values.shape == (2, 3)

    standalone_values, _ = agent.forward_critic(states, skill_groups = 2)
    other_group_values, _ = agent.forward_critic(states, skill_groups = 0)

    assert torch.allclose(values, standalone_values)
    assert not torch.allclose(values, other_group_values)

def test_recurrent_rollout_hidden_state_threading():
    # step-by-step inference that threads the hidden state must match a single full-sequence
    # forward pass, so the recurrent student can be deployed one frame at a time

    actor = Actor(
        512,
        state_encoder = StateEncoder(512, dim_state = 4 + 5, use_rnn = True, num_stacked_frames = 1),
        action_distr = Gaussian()
    )

    images = torch.randn(1, 6, 2, 2)
    proprio = torch.randn(1, 6, 5)

    with torch.no_grad():
        full_actions, _ = actor((images, proprio), deterministic = True)

        step_actions = []
        hidden = None

        for t in range(proprio.shape[1]):
            action, hidden = actor((images[:, t:t + 1], proprio[:, t:t + 1]), time_hiddens = hidden, deterministic = True)
            step_actions.append(action)

    assert torch.allclose(torch.cat(step_actions, dim = 1), full_actions, atol = 1e-4)

def test_velocity_slack_reverse_commands():
    # paper eq. 3 - 1[v^c / v in [slack_low, slack_high]] is sign-symmetric: reversing at half the
    # commanded reverse speed earns the bonus, while opposing the command never does

    hparams = RewardHyperParams()

    def reward(cmd, cur):
        state = State(
            linear_velocity = torch.tensor([[cur, 0., 0.]]),
            angular_velocity = torch.zeros(1, 3),
            projected_gravity = torch.zeros(1, 3),
            commanded_velocity = torch.tensor([[cmd, 0., 0.]]),
            joint_limit_flags = torch.zeros(1, 21),
            contact_forces = torch.zeros(1, 6),
            foot_contact = torch.zeros(1, 2),
            foot_heights = torch.zeros(1, 2),
            foot_ray_hit_heights = torch.zeros(1, 2, 8),
            foot_acceleration = torch.zeros(1, 2, 3),
            heading_error = torch.zeros(1),
            action_rate = torch.zeros(1, 21)
        )
        return reward_velocity_slack(state, hparams).item()

    assert reward(1., 0.5) == 1.    # forward command, half speed
    assert reward(-1., -0.5) == 1.  # reverse command, half speed
    assert reward(-1., -2.0) == 0.  # reversing too fast
    assert reward(-1., 0.5) == 0.   # moving against the command
    assert reward(0., 0.0) == 0.    # zero command - the ratio is undefined
    assert reward(0.05, 0.03) == 0. # below the commanded-speed floor

    # forward-only mode restores the previous behavior

    hparams.slack_allow_reverse = False
    assert reward(-1., -0.5) == 0.

def test_stateful_state_not_shared():
    wrapper_a = RewardShapingWrapper(reward_hparams = RewardHyperParams())
    wrapper_b = RewardShapingWrapper(reward_hparams = RewardHyperParams())

    state = mock_state()
    state.foot_acceleration.zero_()
    state.foot_acceleration[..., 0] = 40. # above the threshold, so the filter accumulates

    reward_a1 = wrapper_a(state)
    reward_a2 = wrapper_a(state)
    reward_b1 = wrapper_b(state) # fresh wrapper - must start from a zeroed filter

    assert torch.allclose(reward_b1, reward_a1)
    assert not torch.allclose(reward_b1, reward_a2)

class FixedLogit(Module):
    # a discriminator that scores by its input mean, so the reward is closed-form

    def forward(self, states):
        return states.mean(dim = -1)

def test_motion_prior():
    real = torch.randn(4, 8, 64)
    fake = torch.randn(4, 8, 64)

    prior = MotionPrior(Discriminator(512, dim_in = 64))

    # logits from the discriminator; the equation 12 reward is positive and grows with realness

    assert prior(real).shape == (4, 8)
    assert (prior.reward(fake) > 0.).all()

    # the discriminator is trained to tell real from fake, and receives gradients

    loss = prior.discriminator_loss(real, fake)
    loss.backward()

    assert all(exists(p.grad) for p in prior.discriminator.parameters())

    # with a fixed discriminator, the softplus reward grows monotonically with the logit

    fixed = MotionPrior(FixedLogit())

    realistic = torch.full((1, 8, 64), 5.)
    unreal = torch.full((1, 8, 64), -5.)

    assert (fixed.reward(realistic) > fixed.reward(unreal)).all()

    # the gradient penalty is configurable - one-centered (wgan-gp) or off

    for kwargs in (dict(grad_penalty_center = 1.), dict(use_grad_penalty = False)):
        configured = MotionPrior(Discriminator(512, dim_in = 64), **kwargs)
        assert configured.discriminator_loss(real, fake).ndim == 0

    # the loss breakdown sums to the total, with no penalty when disabled

    loss, (bce_loss, grad_penalty) = prior.discriminator_loss(real, fake, return_loss_breakdown = True)
    assert torch.allclose(loss, bce_loss + grad_penalty)

    no_gp = MotionPrior(Discriminator(512, dim_in = 64), use_grad_penalty = False)
    loss, (bce_loss, grad_penalty) = no_gp.discriminator_loss(real, fake, return_loss_breakdown = True)
    assert torch.allclose(loss, bce_loss) and grad_penalty == 0.

# grouped weighted advantages (Mysore et al., ICLR 2022) - one critic head per
# reward group, each group's advantage z-scored across the batch and weighted

def test_grouped_reward_wrapper():
    state = mock_state()

    # a flat config coerces to a single group of weight 1., the flat reward exactly

    wrapper = RewardShapingWrapper(reward_hparams = RewardHyperParams())

    assert wrapper.group_names == ('total',)
    assert torch.allclose(wrapper.group_weights, tensor([1.]))

    reward = wrapper(state)
    assert reward.shape == (4,)

    # grouped config - one scalar per reward group, the group weight not baked in

    groups = (
        ('tracking', 2.0, (
            (reward_linear_velocity_tracking, 2.0),
            (reward_angular_velocity_tracking, 2.0),
        )),
        ('safety', 0.5, (
            (reward_velocity_slack, 1.5),
            (reward_heading_error, -1.0),
        )),
    )

    wrapper = RewardShapingWrapper(reward_fns = groups, stateful_reward_fns = (), reward_hparams = RewardHyperParams())

    assert wrapper.group_names == ('tracking', 'safety')
    assert torch.allclose(wrapper.group_weights, tensor([2.0, 0.5]))

    grouped_rewards = wrapper(state)
    assert grouped_rewards.shape == (4, 2)

    # per-group sums match computing the group terms by hand

    tracking_manual = reward_linear_velocity_tracking(state, wrapper.reward_hparams) * 2.0 + reward_angular_velocity_tracking(state, wrapper.reward_hparams) * 2.0
    assert torch.allclose(grouped_rewards[:, 0], tracking_manual)

    safety_manual = reward_velocity_slack(state, wrapper.reward_hparams) * 1.5 + reward_heading_error(state, wrapper.reward_hparams) * -1.0
    assert torch.allclose(grouped_rewards[:, 1], safety_manual)

    # per-reward weight overrides still apply within groups

    wrapper.reward_weights['reward_heading_error'] = -3.0
    assert torch.allclose(wrapper(state)[:, 1], safety_manual - 2.0 * state.heading_error)

    # runtime registration into an explicit group

    def reward_base_height(state, hparams):
        return state.heading_error

    wrapper.add_reward_function_(reward_base_height, 5., group_name = 'safety')
    assert torch.allclose(wrapper(state)[:, 1], safety_manual - 2.0 * state.heading_error + state.heading_error * 5.)

    # adding a function without a group name is an error with multiple groups

    with pytest.raises(AssertionError):
        wrapper.add_reward_function_(reward_base_height, 1.)

def test_grouped_critic():
    num_groups = 3

    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_value_heads = num_groups)

    images = torch.randn(2, 8, 2, 2)
    proprio = torch.randn(2, 8, 5)

    values, _ = critic((images, proprio))
    assert values.shape == (2, 8, num_groups)

    # per-group value targets - each head trains on its own group's returns

    returns = torch.rand(2, 8, num_groups)
    value_loss, _ = critic((images, proprio), target = returns)
    assert value_loss.ndim == 0
    value_loss.backward()

    assert all(exists(p.grad) for p in critic.parameters())

    # a single head still returns flat values

    flat_critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5))
    flat_values, _ = flat_critic((images, proprio))
    assert flat_values.shape == (2, 8)

def test_grouped_calc_gae():
    # per-group gae with trailing reward-group dims, independent per group

    rewards = torch.randn(2, 16, 3)
    values = torch.randn(2, 16, 3)
    masks = torch.ones(2, 16)

    returns = Agent.calc_gae(rewards, values, masks, gamma = 0.99, lam = 0.95)
    assert returns.shape == (2, 16, 3)

    # each group matches an independently computed flat gae

    for group in range(3):
        flat_returns = Agent.calc_gae(rewards[..., group], values[..., group], masks, gamma = 0.99, lam = 0.95)
        assert torch.allclose(returns[..., group], flat_returns)

    # bootstrapped values with an extra step

    bootstrap_returns = Agent.calc_gae(rewards, torch.randn(2, 17, 3), masks)
    assert bootstrap_returns.shape == (2, 16, 3)

def test_grouped_weighted_advantages():
    actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), action_distr = Gaussian())
    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_value_heads = 2)

    agent = Agent(actor, critic, group_weights = (0.5, 2.0))

    values = torch.randn(2, 8, 2)
    returns = torch.randn(2, 8, 2)

    advantages = agent.calc_advantages(values, returns)

    # per-group gae, each z-scored across the batch, weighted and summed

    expected = (
        z_score(returns[..., 0] - values[..., 0]) * 0.5 +
        z_score(returns[..., 1] - values[..., 1]) * 2.0
    )
    assert torch.allclose(advantages, expected)

    # each group is z-scored on its own scale - rescaling one group changes no other

    rescaled_returns = returns.clone()
    rescaled_returns[..., 1] *= 100.
    rescaled_values = values.clone()
    rescaled_values[..., 1] *= 100.
    assert torch.allclose(agent.calc_advantages(rescaled_values, rescaled_returns), advantages, atol = 1e-3)

    # masked batches

    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0]]).bool()
    masked_advantages = agent.calc_advantages(values, returns, mask = mask)
    assert masked_advantages.shape == (2, 8)

    # a single group of weight 1. reduces to the standard z-scored advantage

    flat_agent = Agent(actor, Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5)))
    assert torch.allclose(flat_agent.group_weights, tensor([1.]))

    flat_values = torch.randn(2, 8)
    flat_returns = torch.randn(2, 8)
    flat_advantages = flat_agent.calc_advantages(flat_values, flat_returns)
    assert torch.allclose(flat_advantages, z_score(flat_returns - flat_values))

def test_grouped_agent_learning():
    num_groups = 2

    actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), action_distr = Gaussian())
    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_value_heads = num_groups)

    agent = Agent(actor, critic, group_weights = (2.0, 0.5))

    images = torch.randn(2, 8, 2, 2)
    proprio = torch.randn(2, 8, 5)
    actions = torch.randn(2, 8, 21)
    old_log_probs = torch.randn(2, 8)
    advantages = torch.randn(2, 8)
    returns = torch.rand(2, 8, num_groups)

    policy_loss = agent.actor_loss((images, proprio), actions, old_log_probs, advantages)
    value_loss = agent.critic_loss((images, proprio), returns)
    loss = policy_loss + value_loss

    loss.backward()

    assert all(exists(p.grad) for p in actor.parameters())
    assert all(exists(p.grad) for p in critic.parameters())

def test_grouped_gae_to_advantages_flow():
    # mirrors the training loop: per-group rewards -> per-group gae -> grouped weighted advantages

    num_groups = 2

    actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), action_distr = Gaussian())
    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_value_heads = num_groups)

    agent = Agent(actor, critic, group_weights = (1.5, 1.0))

    rewards = torch.randn(2, 12, num_groups)
    values = torch.randn(2, 13, num_groups) # includes the bootstrap step
    masks = torch.ones(2, 12)
    rollout_mask = torch.ones(2, 12).bool()

    returns = agent.calc_gae(rewards, values, masks)
    assert returns.shape == (2, 12, num_groups)

    advantages = agent.calc_advantages(values[:, :-1], returns, mask = rollout_mask)
    assert advantages.shape == (2, 12)
