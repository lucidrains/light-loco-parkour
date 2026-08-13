import pytest
param = pytest.mark.parametrize

import torch
from torch import tensor
from torch.nn import Module

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

    images = torch.randn(2, 3, 2, 2)
    proprio = torch.randn(2, 3, 5)
    actions = torch.randn(2, 3, 21)
    old_log_probs = torch.randn(2, 3)
    advantages = torch.randn(2, 3)
    returns = torch.rand(2, 3)

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
