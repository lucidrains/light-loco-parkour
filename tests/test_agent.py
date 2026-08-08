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

@param('has_aux_decoder', [False, True])
def test_distillation_wrapper(has_aux_decoder):
    from light_loco_parkour import DistillationWrapper, Agent

    aux_decoder = create_mlp(10, dim_in = 512, depth = 2) if has_aux_decoder else None

    student_encoder = StateEncoder(512, dim_state = 4 + 5)
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

    if has_aux_decoder:
        assert unreduced_loss.shape == (2, 3)
    else:
        assert unreduced_loss.shape == (2, 3, 21)

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
    student_encoder = StateEncoder(512, dim_state = 4 + 5)

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
