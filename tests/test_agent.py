import pytest
param = pytest.mark.parametrize

import torch
from torch import tensor

from x_mlps_pytorch import create_mlp
from light_loco_parkour.light_loco_parkour import Actor, Critic, StateEncoder, Gaussian, Beta

@param('skill_groups', [
    1,
    tensor(1),
    tensor([1])
])
def test_agent(skill_groups):
    student_actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5))
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
        state_encoder = StateEncoder(512, dim_state = 4 + 5),
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
