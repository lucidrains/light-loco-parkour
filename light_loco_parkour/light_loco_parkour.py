from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
from beartype import beartype
from contextlib import nullcontext

import torch
from torch import nn, cat, stack, einsum, is_tensor, tensor, diff, Tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm
import torch.nn.functional as F

import einx
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from torch_einops_utils import (
    maybe,
    cast_tensor,
    pad_left_ndim_to,
    pad_left_at_dim,
    pad_right_at_dim,
    pad_right_ndim_to,
    lens_to_mask,
    masked_mean,
    pack_with_inverse,
    safe_cat,
    z_score
)
from torch_einops_utils.shape import shape, is_shape

from x_mlps_pytorch import create_mlp, create_filmable_mlp

from hl_gauss_pytorch import HLGaussLayer

from assoc_scan import AssocScan

from torch.distributions import Distribution, Normal, Beta as _Beta

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def xnor(x, y):
    return x == y

def cast_tuple(v, length = 1):
    return v if isinstance(v, tuple) else (v,) * length

def pluck(d, keys):
    return [d[k] for k in keys]

def cast_to_tensor(t, device = None):
    t = maybe(cast_tensor)(t, device = device)

    if exists(t) and not is_tensor(t):
        t = tensor(t, device = device)

    return t

def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

# action distributions

class Gaussian(Module):
    def __init__(
        self,
        param_type = 'log_std',
        pos_fn = 'exp',
        min_std = 1e-5,
        raw_bounds = (-20, 2)
    ):
        super().__init__()
        assert param_type in ('log_std', 'log_var')
        assert pos_fn in ('exp', 'softplus')

        self.param_type = param_type
        self.pos_fn = pos_fn
        self.min_std = min_std
        self.raw_bounds = raw_bounds

    def mean(self, params):
        return params[..., 0]

    def log_prob(
        self,
        params_or_dist,
        action,
        sum_action_dim = True
    ):
        dist = params_or_dist if isinstance(params_or_dist, Distribution) else self(params_or_dist)
        log_prob = dist.log_prob(action)
        return log_prob.sum(dim = -1) if sum_action_dim else log_prob

    def forward(self, params):
        pos_fn = self.pos_fn
        mean, raw = params.unbind(dim = -1)

        # maybe clamp raw parameters to bounds

        if exists(self.raw_bounds):
            min_val, max_val = self.raw_bounds
            raw = raw.clamp(min = min_val, max = max_val)

        # positive activation function

        if pos_fn == 'exp':
            pos = raw.exp()
        elif pos_fn == 'softplus':
            pos = F.softplus(raw)

        # derive standard deviation

        if self.param_type == 'log_std':
            std = pos.clamp(min = self.min_std)
        elif self.param_type == 'log_var':
            std = pos.clamp(min = self.min_std ** 2).sqrt()

        return Normal(mean, std)

# beta distribution policy - unimodal mean-concentration reparameterization

class Beta(Module):
    def __init__(
        self,
        pos_fn = 'softplus',
        init_conc = 10.,
        eps = 1e-5
    ):
        super().__init__()
        assert pos_fn in ('exp', 'softplus')
        self.pos_fn = pos_fn
        self.init_conc = init_conc
        self.eps = eps

    def concentration(
        self,
        raw_conc
    ):
        raw_conc = raw_conc + self.init_conc

        if self.pos_fn == 'softplus':
            return F.softplus(raw_conc) + 2.
        elif self.pos_fn == 'exp':
            return raw_conc.exp() + 2.

    def mean(
        self,
        params
    ):
        raw_mean, _ = params.unbind(dim = -1)
        return raw_mean.sigmoid().clamp(min = self.eps, max = 1. - self.eps)

    def log_prob(
        self,
        params_or_dist,
        action,
        sum_action_dim = True,
        eps = None
    ):
        eps = default(eps, self.eps)
        action = action.clamp(min = eps, max = 1. - eps)
        dist = params_or_dist if isinstance(params_or_dist, Distribution) else self(params_or_dist)
        log_prob = dist.log_prob(action)
        return log_prob.sum(dim = -1) if sum_action_dim else log_prob

    def forward(self, params):
        raw_mean, raw_conc = params.unbind(dim = -1)

        # mean mapped to (0, 1), clamped so the concentration parameters stay positive

        mean = raw_mean.sigmoid().clamp(min = self.eps, max = 1. - self.eps)

        # positive concentration

        conc = self.concentration(raw_conc)

        # convert to beta distribution with exact mean = mean

        alpha = mean * conc
        beta = (1. - mean) * conc

        return _Beta(alpha, beta)

# one hot helper module

class OneHot(Module):
    def __init__(
        self,
        num_classes = 1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim_cond = 0 if num_classes <= 1 else num_classes
        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def forward(
        self,
        category: int | Tensor | None,
        batch_size: int | None = None,
        time_steps: int | None = None
    ):
        num_classes = self.num_classes

        if self.dim_cond == 0 or not exists(category):
            return None

        # handle 0d (int / scalar tensor) - expand across batch and time

        if isinstance(category, int) or (is_tensor(category) and category.ndim == 0):
            assert exists(batch_size) and exists(time_steps), 'batch_size and time_steps must be provided for scalar category'
            category = torch.full((batch_size, time_steps), category, dtype = torch.long, device = self.device)

        # handle 1d (batch,) - expand across time

        elif is_tensor(category) and is_shape(category, 'b'):
            assert exists(time_steps), 'time_steps must be provided for 1d category tensor (batch)'
            category = repeat(category, 'b -> b t', t = time_steps)

        return F.one_hot(category, num_classes).float()

# state encoder

class StateEncoder(Module):
    def __init__(
        self,
        dim,
        *,
        dim_state,
        cond_key: int | str | None = None,
        dim_cond: int | None = None,
        num_stacked_frames = 5,
        tbptt_timesteps = 10,
        use_rnn = False
    ):
        super().__init__()

        is_filmable = exists(cond_key) or exists(dim_cond)
        assert not is_filmable or (exists(cond_key) and exists(dim_cond)), 'both cond_key and dim_cond must be provided if film conditioning is enabled'

        self.num_stacked_frames = num_stacked_frames
        self.use_rnn = use_rnn

        self.cond_key = cond_key
        self.dim_cond = dim_cond
        self.is_filmable = is_filmable

        dim_state_total = dim_state * num_stacked_frames

        create_mlp_fn = create_filmable_mlp if is_filmable else create_mlp

        mlp_kwargs = dict()
        if is_filmable:
            mlp_kwargs.update(cond_dim = dim_cond * num_stacked_frames)

        self.mlp_encoder = create_mlp_fn(dim, depth = 2, dim_in = dim_state_total, **mlp_kwargs)
        self.rnn = nn.GRU(dim, dim, batch_first = True) if use_rnn else None

        self.tbptt_timesteps = tbptt_timesteps
        self.has_tbptt = tbptt_timesteps > 0 and use_rnn

    def forward(
        self,
        states: Sequence[Tensor] | dict[str, Tensor] | Tensor, # [b t ...]
        time_hiddens = None
    ):
        frames, tbptt_timesteps = self.num_stacked_frames, self.tbptt_timesteps

        def stack_frames(state_inputs):
            packed, _ = pack_with_inverse(state_inputs, 'b t *')
            padded = pad_left_at_dim(packed, frames - 1, dim = 1)
            stacked = padded.unfold(1, frames, 1)
            return rearrange(stacked, 'b t ... -> b t (...)')

        # maybe film conditioning with proprioception

        mlp_kwargs = dict()

        if self.is_filmable:
            cond_key = self.cond_key
            if isinstance(cond_key, int) and isinstance(states, Sequence):
                states = list(states)
                cond_state = states.pop(cond_key)
            elif isinstance(cond_key, str) and isinstance(states, dict):
                states = dict(states)
                cond_state = states.pop(cond_key)
            else:
                raise ValueError(f'states must be a Sequence when cond_key is int, or a dict when cond_key is str (got {type(states)})')

            mlp_kwargs.update(cond = stack_frames(cond_state))

        # dict to list of tensors

        if isinstance(states, dict):
            states = list(states.values())

        # stack the states and encode them

        stacked_states = stack_frames(states)

        encoded_state = self.mlp_encoder(stacked_states, **mlp_kwargs)

        if not self.use_rnn:
            return encoded_state, None

        # maybe truncated backprop through time

        if not self.has_tbptt:
            encoded_states = [encoded_state]
        else:
            encoded_states = encoded_state.split(tbptt_timesteps, dim = 1)

        # temporal

        time_encoded_states = None

        for encoded_states_timechunk in encoded_states:
            chunked_time_encoded_states, time_hiddens = self.rnn(encoded_states_timechunk, time_hiddens)

            time_hiddens = time_hiddens.detach()

            time_encoded_states = safe_cat((time_encoded_states, chunked_time_encoded_states), dim = 1)

        # set the next time hiddens

        next_time_hiddens = time_hiddens

        return time_encoded_states, next_time_hiddens

# actor

class Actor(Module):
    def __init__(
        self,
        dim,
        *,
        state_encoder: StateEncoder,
        num_skill_groups = 1,
        depth = 4,
        num_actions = 21,
        distr: Module | None = None,
        action_distr: Module | None = None,
        distr_dim_out = 2,
        aux_decoder: Module | None = None
    ):
        super().__init__()
        self.state_encoder = state_encoder

        self.skill_cond = OneHot(num_skill_groups)

        self.backbone = create_mlp(dim, dim_in = self.skill_cond.dim_cond + dim, depth = depth)

        action_distr = default(action_distr, distr)
        self.action_distr = action_distr
        self.distr_dim_out = distr_dim_out

        self.aux_decoder = aux_decoder

        self.to_actions = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, num_actions * distr_dim_out),
            Rearrange('... (d distr_params) -> ... d distr_params', distr_params = distr_dim_out)
        )

    def forward(
        self,
        states: Sequence[Tensor],
        time_hiddens = None,
        skill_groups = None,
        deterministic = False,
        return_action_distr = False,
        sample_action = False,
        return_log_prob = False,
        aux_decoder: Module | None = None,
        aux_decoder_target: Tensor | None = None
    ):
        aux_decoder = default(aux_decoder, self.aux_decoder)

        time_encoded_states, next_time_hiddens = self.state_encoder(states, time_hiddens)

        batch, time, _ = shape(time_encoded_states, 'b t ...')

        maybe_one_hot = self.skill_cond(
            skill_groups,
            batch_size = batch,
            time_steps = time
        )

        time_encoded_states = safe_cat((time_encoded_states, maybe_one_hot), dim = -1)

        # backbone

        embed = self.backbone(time_encoded_states)

        # predict actions

        params = self.to_actions(embed)

        action_distr = self.action_distr

        # handle action outputs (deterministic, sampling with optional log_prob, or distribution)

        if deterministic:
            action_out = action_distr.mean(params) if exists(action_distr) else params[..., 0]
            output = (action_out, next_time_hiddens)

        elif sample_action or return_log_prob:
            assert exists(action_distr), 'action_distr must be passed to Actor to sample actions'
            dist = action_distr(params)
            action = dist.sample()

            if return_log_prob:
                log_prob = action_distr.log_prob(dist, action)
                output = (action, log_prob, next_time_hiddens)
            else:
                output = (action, next_time_hiddens)

        elif return_action_distr:
            assert exists(action_distr), 'action_distr must be passed to Actor to return action distribution'
            dist = action_distr(params)
            output = (dist, next_time_hiddens)

        else:
            action_out = action_distr(params) if exists(action_distr) else params
            output = (action_out, next_time_hiddens)

        # maybe aux decoder for student

        assert xnor(exists(aux_decoder), exists(aux_decoder_target))

        if not exists(aux_decoder):
            return output

        aux_decoder_pred = aux_decoder(embed)

        aux_loss = F.mse_loss(aux_decoder_pred, aux_decoder_target, reduction = 'none')

        return output, aux_loss

# critic

class Critic(Module):
    def __init__(
        self,
        dim,
        *,
        num_skill_groups = 1,
        state_encoder: StateEncoder,
        depth = 4,
        use_regression = True,
        min_value = -1.,
        max_value = 1.,
        num_bins = 2,
        use_symlog = False
    ):
        super().__init__()
        self.state_encoder = state_encoder

        self.skill_cond = OneHot(num_skill_groups)

        self.backbone = create_mlp(dim, dim_in = self.skill_cond.dim_cond + dim, depth = depth)

        # mse regression or hl gauss value prediction

        hl_gauss_loss = dict(
            min_value = min_value,
            max_value = max_value,
            num_bins = num_bins,
            use_symlog = use_symlog
        )

        self.use_regression = use_regression

        self.hl_gauss_layer = HLGaussLayer(
            dim,
            use_regression = use_regression,
            hl_gauss_loss = hl_gauss_loss
        )

    def forward(
        self,
        states: Sequence[Tensor],
        time_hiddens = None,
        skill_groups = None, # int | Tensor
        target = None, # value targets, if passed a value loss (mse or hl gauss) is returned in place of the values
        mask = None,
        return_logits = False
    ):
        time_encoded_states, next_time_hiddens = self.state_encoder(states, time_hiddens)

        batch, time, _ = shape(time_encoded_states, 'b t ...')

        maybe_one_hot = self.skill_cond(
            skill_groups,
            batch_size = batch,
            time_steps = time
        )

        time_encoded_states = safe_cat((time_encoded_states, maybe_one_hot), dim = -1)

        # backbone

        embed = self.backbone(time_encoded_states)

        # value prediction (mse regression) or hl gauss classification over value bins

        values = self.hl_gauss_layer(embed, target, mask = mask, return_logits = return_logits)

        return values, next_time_hiddens

# agent - handles both inference and ppo learning for asymmetric actor / critic

class Agent(Module):
    def __init__(
        self,
        actor: Module,
        critic: Module,
        *,
        clip = 0.2,
        entropy_weight = 0.01,
        norm_advantages = True,
        actor_state_keys: tuple[str, ...] | None = None,
        critic_state_keys: tuple[str, ...] | None = None
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic

        # ppo hyperparameters

        self.clip = clip
        self.entropy_weight = entropy_weight
        self.norm_advantages = norm_advantages

        self.actor_state_keys = actor_state_keys
        self.critic_state_keys = critic_state_keys

    def route_states(
        self,
        states: dict[str, Tensor] | Sequence[Tensor] | Tensor
    ):
        actor_states = pluck(states, self.actor_state_keys) if exists(self.actor_state_keys) else states
        critic_states = pluck(states, self.critic_state_keys) if exists(self.critic_state_keys) else states
        return actor_states, critic_states

    # gae with assoc scan

    @staticmethod
    def calc_gae(
        rewards,
        values,
        masks,
        gamma = 0.99,
        lam = 0.95,
        use_accelerated = None
    ):
        n_steps = rewards.shape[-1]
        n_values = values.shape[-1]

        # values may be aligned with rewards (bootstrapping zero at the end),
        # or hold one extra step carrying the bootstrap value of the next state

        assert n_values in (n_steps, n_steps + 1)

        if n_values == n_steps:
            values = pad_right_at_dim(values, 1, value = 0.)

        values, values_next = values[..., :-1], values[..., 1:]

        delta = rewards + gamma * values_next * masks - values
        gates = gamma * lam * masks

        scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

        gae = scan(gates, delta)

        returns = gae + values

        return returns

    # ppo losses

    def actor_loss(
        self,
        states: dict[str, Tensor] | Sequence[Tensor] | Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        *,
        mask: Tensor | None = None,
        lens: Tensor | None = None,
        skill_groups = None,
        time_hiddens = None,
        clip: float | None = None,
        entropy_weight: float | None = None
    ):
        actor_states, _ = self.route_states(states)
        mask = lens_to_mask(lens) if exists(lens) and not exists(mask) else mask

        clip = default(clip, self.clip)
        entropy_weight = default(entropy_weight, self.entropy_weight)

        # sample actions from current policy and derive log probs

        dist, _ = self.actor(actor_states, time_hiddens = time_hiddens, skill_groups = skill_groups, return_action_distr = True)

        log_probs = self.actor.action_distr.log_prob(dist, actions)
        entropy = reduce(dist.entropy(), '... d -> ...', 'sum')

        # maybe normalize advantages (masked z-score)

        if self.norm_advantages:
            advantages = z_score(advantages, mask = mask)

        # clipped surrogate objective

        ratios = (log_probs - old_log_probs).exp()
        clipped_ratios = ratios.clamp(1. - clip, 1. + clip)

        policy_loss = -torch.minimum(ratios * advantages, clipped_ratios * advantages)

        # entropy bonus

        policy_loss = policy_loss - entropy * entropy_weight

        return masked_mean(policy_loss, mask)

    def critic_loss(
        self,
        states: dict[str, Tensor] | Sequence[Tensor] | Tensor,
        returns: Tensor, # value targets for the critic
        *,
        mask: Tensor | None = None,
        lens: Tensor | None = None,
        skill_groups = None,
        time_hiddens = None
    ):
        _, critic_states = self.route_states(states)
        mask = lens_to_mask(lens) if exists(lens) and not exists(mask) else mask

        value_loss, _ = self.critic(critic_states, time_hiddens = time_hiddens, skill_groups = skill_groups, target = returns, mask = mask)

        return value_loss

    # inference

    def forward_actor(
        self,
        states: dict[str, Tensor] | Sequence[Tensor] | Tensor,
        **kwargs
    ):
        actor_states, _ = self.route_states(states)
        return self.actor(actor_states, **kwargs)

    def forward_critic(
        self,
        states: dict[str, Tensor] | Sequence[Tensor] | Tensor,
        **kwargs
    ):
        _, critic_states = self.route_states(states)
        return self.critic(critic_states, **kwargs)

    def forward(
        self,
        states: dict[str, Tensor] | Sequence[Tensor] | Tensor,
        **kwargs
    ):
        actor_out = self.forward_actor(states, **kwargs)
        critic_out = self.forward_critic(states)
        return actor_out, critic_out

# distillation wrapper

class DistillationWrapper(Module):
    @beartype
    def __init__(
        self,
        student: Module,
        teacher: Module,
        *,
        student_state_keys: tuple[str, ...],
        teacher_state_keys: tuple[str, ...],
        aux_decoder: Module | None = None,
        privileged_state_key: str | tuple[str, ...] | None = None,
        aux_loss_weight = 1.0,
        detach_teacher = True
    ):

        super().__init__()
        self.student = student
        self.teacher = teacher

        self.student_state_keys = student_state_keys
        self.teacher_state_keys = teacher_state_keys

        self.aux_decoder = aux_decoder
        self.privileged_state_key = privileged_state_key
        self.aux_loss_weight = aux_loss_weight

        self.teacher_context = torch.no_grad if detach_teacher else nullcontext

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        states: dict[str, Tensor],
        weights: Tensor | None = None,
        lens: Tensor | None = None,
        mask: Tensor | None = None,
        teacher_skill_groups = None,
        student_skill_groups = None,
        student_time_hiddens = None,
        teacher_time_hiddens = None,
        aux_decoder: Module | None = None,
        aux_decoder_target: Tensor | None = None,
        return_unreduced = False,
        return_loss_breakdown = False
    ):

        student_states = pluck(states, self.student_state_keys)
        teacher_states = pluck(states, self.teacher_state_keys)

        # resolve mask from lens if provided

        mask = lens_to_mask(lens) if exists(lens) and not exists(mask) else mask

        # resolve aux decoder and target

        aux_decoder = default(aux_decoder, self.aux_decoder)
        aux_decoder = default(aux_decoder, self.student.aux_decoder)

        if not exists(aux_decoder_target) and exists(self.privileged_state_key):
            privileged_keys = cast_tuple(self.privileged_state_key)
            aux_decoder_target = safe_cat(pluck(states, privileged_keys), dim = -1)

        # forward student

        student_kwargs = dict(
            time_hiddens = student_time_hiddens,
            skill_groups = student_skill_groups,
            deterministic = True
        )

        has_aux = exists(aux_decoder) and exists(aux_decoder_target)

        if has_aux:
            student_kwargs.update(
                aux_decoder = aux_decoder,
                aux_decoder_target = aux_decoder_target
            )

        student_out = self.student(student_states, **student_kwargs)

        aux_loss = self.zero
        if has_aux:
            student_out, aux_loss = student_out

        student_mean, _ = student_out

        # forward teacher

        with self.teacher_context():
            teacher_mean, _ = self.teacher(
                teacher_states,
                time_hiddens = teacher_time_hiddens,
                skill_groups = teacher_skill_groups,
                deterministic = True
            )

        # compute distillation mse loss

        distill_loss = F.mse_loss(student_mean, teacher_mean, reduction = 'none')
        distill_loss = reduce(distill_loss, '... d -> ...', 'mean')

        # maybe combine with aux loss

        if has_aux:
            aux_loss = reduce(aux_loss, '... d -> ...', 'mean') * self.aux_loss_weight

        loss = distill_loss + aux_loss

        # maybe weight loss

        if exists(weights):
            loss = loss * pad_right_ndim_to(weights, loss.ndim)
            distill_loss = distill_loss * pad_right_ndim_to(weights, distill_loss.ndim)
            if has_aux:
                aux_loss = aux_loss * pad_right_ndim_to(weights, aux_loss.ndim)

        # maybe mask unreduced loss

        if exists(mask) and return_unreduced:
            loss = einx.where('b t, b t ..., -> b t ...', mask, loss, 0.)
            distill_loss = einx.where('b t, b t ..., -> b t ...', mask, distill_loss, 0.)
            if has_aux:
                aux_loss = einx.where('b t, b t ..., -> b t ...', mask, aux_loss, 0.)

        # return

        if not return_unreduced:
            loss = masked_mean(loss, mask)
            distill_loss = masked_mean(distill_loss, mask)
            if has_aux:
                aux_loss = masked_mean(aux_loss, mask)

        if not return_loss_breakdown:
            return loss

        loss_breakdown = (distill_loss, aux_loss)

        return loss, loss_breakdown

# reward shaping (section iv-c, table 1)
# each reward function returns one scalar per sample; RewardShapingWrapper sums them with their weights

@dataclass
class State:
    linear_velocity: Tensor
    angular_velocity: Tensor
    projected_gravity: Tensor
    commanded_velocity: Tensor
    joint_limit_flags: Tensor
    contact_forces: Tensor
    foot_contact: Tensor
    foot_heights: Tensor
    foot_ray_hit_heights: Tensor
    foot_acceleration: Tensor
    heading_error: Tensor
    action_rate: Tensor

@dataclass
class RewardHyperParams:
    sigma_linear: float = 0.25
    sigma_angular: float = 0.25
    slack_low: float = 0.3
    slack_high: float = 1.5
    footstep_threshold: float = 0.1
    contact_force_threshold: float = 1.
    foot_accel_threshold: float = 30.
    foot_accel_tau: float = 0.06
    dt: float = 0.02

def gaussian_kernel(errors, sigma):
    # equation 1 - exp(-||errors||^2 / sigma^2), peaks at 1 when the error is zero
    return (-errors.square().sum(dim = -1) / (sigma ** 2)).exp()

def reward_linear_velocity_tracking(state: State, hparams: RewardHyperParams):
    # equation 1 - track the commanded planar velocity (x, y)
    error = state.linear_velocity[..., :2] - state.commanded_velocity[..., :2]
    return gaussian_kernel(error, hparams.sigma_linear)

def reward_angular_velocity_tracking(state: State, hparams: RewardHyperParams):
    # equation 1 - exponential kernel on yaw rate error
    error = state.angular_velocity[..., 2:3] - state.commanded_velocity[..., 2:3]
    return gaussian_kernel(error, hparams.sigma_angular)

def reward_upright_orientation(state: State, hparams: RewardHyperParams):
    # equation 2 - stay upright: tilt is how far the projected gravity vector leans from the vertical
    tilt_error = state.projected_gravity[..., :2].norm(dim = -1)
    return (-2 * tilt_error.square()).exp() + 0.1 * (-tilt_error).exp()

def reward_velocity_slack(state: State, hparams: RewardHyperParams):
    # equation 3 - reward forward speed within [slack_low, slack_high] of the commanded speed
    forward_speed = state.linear_velocity[..., 0]
    commanded_speed = state.commanded_velocity[..., 0]

    slow_enough = forward_speed >= hparams.slack_low * commanded_speed
    fast_enough = forward_speed <= hparams.slack_high * commanded_speed
    is_commanded = commanded_speed > 0.

    return (slow_enough & fast_enough & is_commanded).float()

def reward_undesired_contact(state: State, hparams: RewardHyperParams):
    # table 1 - penalize contact on any link other than the feet
    contact = (state.contact_forces > hparams.contact_force_threshold).float()
    return reduce(contact, 'b ... -> b', 'sum')

def reward_joint_limit(state: State, hparams: RewardHyperParams):
    # table 1 - penalize joints that are near a limit and moving toward it
    return reduce(state.joint_limit_flags.float(), 'b ... -> b', 'sum')

def reward_illegal_footstep(state: State, hparams: RewardHyperParams):
    # equation 4 - penalize steps where any ray under a contacting foot hits ground too far below it
    foot_ray_depth = state.foot_heights[..., None] - state.foot_ray_hit_heights
    illegal_fraction = (foot_ray_depth > hparams.footstep_threshold).float().mean(dim = -1)
    return reduce(illegal_fraction * state.foot_contact, 'b ... -> b', 'sum')

def reward_heading_error(state: State, hparams: RewardHyperParams):
    # table 1 - penalize deviation from the commanded heading
    return state.heading_error.abs()

def reward_opposite_direction(state: State, hparams: RewardHyperParams):
    # table 1 - penalize moving against the commanded direction
    current = state.linear_velocity[..., :2]
    commanded_direction = l2norm(state.commanded_velocity[..., :2])
    return F.relu(-(current * commanded_direction).sum(dim = -1))

def reward_action_rate(state: State, hparams: RewardHyperParams):
    # table 1 - penalize abrupt changes in actions
    return reduce(state.action_rate.square(), 'b ... -> b', 'sum')

DEFAULT_REWARD_FNS = (
    (reward_linear_velocity_tracking, 2.0),
    (reward_angular_velocity_tracking, 2.0),
    (reward_upright_orientation, 1.0),
    (reward_velocity_slack, 1.5),
    (reward_undesired_contact, -2.0),
    (reward_joint_limit, -10.0),
    (reward_illegal_footstep, -1.0),
    (reward_heading_error, -1.0),
    (reward_opposite_direction, -1.0),
    (reward_action_rate, -0.1)
)

class StatefulReward(Module):
    # reward function with state across calls
    def reset_(self):
        raise NotImplementedError

    def forward(self, state, hparams):
        raise NotImplementedError

class FootAccelerationPenalty(StatefulReward):
    # equation 5 - penalize foot acceleration above the threshold, smoothed over time
    # so a single spike is remembered for a few steps and sustained jerk keeps growing
    def __init__(self):
        super().__init__()
        self.prev_excess_accel = None

    def reset_(self):
        self.prev_excess_accel = None

    def forward(self, state, hparams):
        alpha = math.exp(-hparams.dt / hparams.foot_accel_tau)

        excess_accel = F.relu(state.foot_acceleration.norm(dim = -1) - hparams.foot_accel_threshold)
        excess_accel = reduce(excess_accel, 'b ... -> b', 'sum')

        # lazy init of the running filter; reset it if the batch size or device changes
        if self.prev_excess_accel is None or self.prev_excess_accel.shape != excess_accel.shape or self.prev_excess_accel.device != excess_accel.device:
            self.prev_excess_accel = excess_accel.new_zeros(excess_accel.shape)

        filtered = alpha * self.prev_excess_accel + excess_accel
        self.prev_excess_accel = filtered.detach()

        return filtered

def default_stateful_reward_fns():
    # instantiate fresh per wrapper, so state is not shared
    return ((FootAccelerationPenalty(), 0.01),)

class RewardShapingWrapper(Module):
    def __init__(
        self,
        reward_fns = DEFAULT_REWARD_FNS,
        stateful_reward_fns = None,
        reward_weights: dict | None = None, # dict of fn name -> weight overrides
        reward_hparams: RewardHyperParams | None = None
    ):
        super().__init__()
        stateful_reward_fns = default(stateful_reward_fns, default_stateful_reward_fns())

        self.reward_fns = {reward_fn.__name__: (reward_fn, weight) for reward_fn, weight in reward_fns}
        self.stateful_reward_fns = {
            reward_fn.__class__.__name__: (reward_fn, weight)
            for reward_fn, weight in stateful_reward_fns
        }
        self.reward_weights = default(reward_weights, dict())
        self.reward_hparams = reward_hparams
        self.reset_()

    def add_reward_function_(
        self,
        reward_fn,
        weight
    ):
        reward_fn_name = reward_fn.__name__
        assert reward_fn_name not in self.reward_fns, 'reward function names must be unique'
        self.reward_fns[reward_fn_name] = (reward_fn, weight)

    def delete_reward_function_(
        self,
        reward_fn_name
    ):
        assert reward_fn_name in self.reward_fns
        del self.reward_fns[reward_fn_name]

    def add_stateful_reward_function_(
        self,
        reward_fn,
        weight
    ):
        reward_fn_name = reward_fn.__class__.__name__
        assert reward_fn_name not in self.stateful_reward_fns, 'stateful reward function names must be unique'
        self.stateful_reward_fns[reward_fn_name] = (reward_fn, weight)

    def delete_stateful_reward_function_(
        self,
        reward_fn_name
    ):
        assert reward_fn_name in self.stateful_reward_fns
        del self.stateful_reward_fns[reward_fn_name]

    def reset_(self):
        for reward_fn, _ in self.stateful_reward_fns.values():
            reward_fn.reset_()

    def forward(
        self,
        state: State,
        reward_hparams: RewardHyperParams | None = None
    ):
        # section iv-c1 - weighted sum of reward terms

        reward_hparams = default(reward_hparams, self.reward_hparams)
        assert exists(reward_hparams), 'reward hyperparameters must be provided'

        total_reward = state.linear_velocity.new_zeros(state.linear_velocity.shape[0])

        for reward_fns in (self.reward_fns, self.stateful_reward_fns):
            for reward_fn_name, (reward_fn, weight) in reward_fns.items():
                weight = self.reward_weights.get(reward_fn_name, weight)
                total_reward = total_reward + reward_fn(state, reward_hparams) * weight

        return total_reward

# motion prior (equation 12)

def gradient_penalty(inputs, output, weight = 10., center = 0.):
    # center = 0. is zero-centered (modern gan baseline), center = 1. is one-centered (wgan-gp)

    gradients, = torch.autograd.grad(
        outputs = output,
        inputs = inputs,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True
    )

    gradients = rearrange(gradients, 'b ... -> b (...)')

    return weight * ((gradients.norm(dim = 1) - center) ** 2).mean()

class Discriminator(Module):
    # equation 12 - the amp discriminator D_phi

    def __init__(
        self,
        dim,
        *,
        dim_in,
        depth = 4
    ):
        super().__init__()
        self.net = create_mlp(dim, dim_in = dim_in, dim_out = 1, depth = depth)

    def forward(self, states):
        logits = self.net(states)
        return rearrange(logits, '... 1 -> ...')

class MotionPrior(Module):
    def __init__(
        self,
        discriminator: Discriminator,
        *,
        grad_penalty_weight = 10.,
        grad_penalty_center = 0.,
        use_grad_penalty = True
    ):
        super().__init__()
        self.discriminator = discriminator
        self.grad_penalty_weight = grad_penalty_weight
        self.grad_penalty_center = grad_penalty_center
        self.use_grad_penalty = use_grad_penalty

        self.register_buffer('zero', tensor(0.), persistent = False)

    def reward(self, states):
        # equation 12 - r_amp = -log(1 - D) = softplus(logit)
        logits = self.discriminator(states)
        return F.softplus(logits)

    def discriminator_loss(self, real, fake, return_loss_breakdown = False):
        # section v-c2 - bce between real references and policy rollouts

        # labels

        labels = cat((torch.ones_like(real[..., 0]), torch.zeros_like(fake[..., 0])))

        real_and_fake = cat((real, fake))

        if self.use_grad_penalty:
            real_and_fake = real_and_fake.requires_grad_(True)

        logits = self.discriminator(real_and_fake)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # gradient penalty

        grad_penalty = self.zero

        if self.use_grad_penalty:
            grad_penalty = gradient_penalty(
                real_and_fake,
                logits,
                weight = self.grad_penalty_weight,
                center = self.grad_penalty_center
            )

        loss = bce_loss + grad_penalty

        if not return_loss_breakdown:
            return loss

        return loss, (bce_loss, grad_penalty)

    def forward(self, states):
        return self.discriminator(states)

# phase conditional motion prior (section v-c2, equation 12)
# the amp prior switches between the locomotion and skill discriminators
# once the robot passes the prior transition position ahead of the obstacle, driving the handoff

class PhaseConditionalMotionPrior(Module):
    def __init__(
        self,
        motion_priors: MotionPrior | Sequence[MotionPrior],
        *,
        prior_transition_positions: Tensor | float | Sequence[float] | None = None,
        smooth_handoff = False,
        handoff_temperature = 1.
    ):
        super().__init__()

        self.smooth_handoff = smooth_handoff
        self.handoff_temperature = handoff_temperature

        prior_transition_positions = cast_to_tensor(prior_transition_positions)
        prior_transition_positions = maybe(pad_left_ndim_to)(prior_transition_positions, 1)

        is_single = isinstance(motion_priors, MotionPrior)

        # a single prior is replicated (shared weights) across the phases

        if is_single:
            assert exists(prior_transition_positions), 'a single motion prior cannot tell us how many phases there are; provide prior_transition_positions (the number of phases is its length plus one)'

            motion_priors = [motion_priors] * (len(prior_transition_positions) + 1)

        self.motion_priors = ModuleList(motion_priors)
        self.num_phases = len(self.motion_priors)

        if exists(prior_transition_positions):
            assert len(prior_transition_positions) == self.num_phases - 1, f'need one phase per motion prior: {len(prior_transition_positions)} transition positions divide the track into {len(prior_transition_positions) + 1} phases, but {self.num_phases} priors were given'

        self.register_buffer('prior_transition_positions', prior_transition_positions, persistent = False)

    def resolve_transition_positions(self, prior_transition_positions = None, device = None):
        # the per-call transition positions, defaulting to the ones given at construction

        prior_transition_positions = default(prior_transition_positions, self.prior_transition_positions)
        assert exists(prior_transition_positions), 'pass prior_transition_positions to place the phase switches (or construct the prior with them)'

        return cast_to_tensor(prior_transition_positions, device = device)

    def resolve_phases(self, positions = None, prior_transition_positions = None, phases = None):
        # explicit phases take priority over automatic resolution

        if exists(phases):
            return phases

        assert exists(positions), 'pass positions to resolve the phase automatically, or pass phases explicitly'

        # a flat position trace (no batch dim) is treated as one trajectory and broadcast over the batch

        if positions.ndim == 1:
            positions = rearrange(positions, 'n -> 1 n')

        # the phase is the number of transition positions passed

        prior_transition_positions = self.resolve_transition_positions(prior_transition_positions, device = positions.device)

        passed = einx.greater('b ... , n -> b ... n', positions, prior_transition_positions)
        return reduce(passed, 'b ... n -> b ...', 'sum')

    def phase_weights(self, positions = None, prior_transition_positions = None, phases = None):
        # hard selection - one hot of the resolved phase

        if not self.smooth_handoff or exists(phases):
            phases = self.resolve_phases(positions, prior_transition_positions, phases)
            return F.one_hot(phases.long(), self.num_phases).float()

        # smooth handoff - cumulative sigmoid switches across the transition positions, so the
        # phase weights shift gradually and always sum to one

        assert exists(positions), 'the smooth handoff needs positions to know how close the robot is to each transition point'

        # a flat position trace (no batch dim) is treated as one trajectory and broadcast over the batch

        if positions.ndim == 1:
            positions = rearrange(positions, 'n -> 1 n')

        prior_transition_positions = self.resolve_transition_positions(prior_transition_positions, device = positions.device)

        switches = (einx.subtract('b ... , n -> b ... n', positions, prior_transition_positions) * self.handoff_temperature).sigmoid()

        phase_switch_shape = (*positions.shape, 1)

        ones = positions.new_ones(phase_switch_shape)
        zeros = positions.new_zeros(phase_switch_shape)

        # weight_i = switch_i - switch_{i + 1}, so the weights always sum to one

        switches = cat((ones, switches, zeros), dim = -1)

        return -diff(switches, dim = -1)

    def auto_positions(self, num_steps, prior_transition_positions = None):
        # when positions are not given, assume the time steps traverse the track uniformly,
        # from the start of the track to one trigger-spacing past the last transition, so
        # every phase is visited

        triggers = default(prior_transition_positions, self.prior_transition_positions)
        assert exists(triggers), 'automatic positions need prior_transition_positions to span the track'

        triggers = cast_to_tensor(triggers)

        first, *rest = triggers
        last = first if not rest else rest[-1]

        # span the track from its start to one trigger-spacing past the last transition

        lo = torch.minimum(first, triggers.new_tensor(0.))

        if not rest:
            hi = 2 * last
        else:
            spacing = (last - first) / len(rest)
            hi = last + spacing

        return torch.linspace(lo, hi, num_steps, device = triggers.device, dtype = triggers.dtype)

    def reward(self, states, positions = None, prior_transition_positions = None, phases = None):
        # equation 12 - r_amp = -log(1 - D_phi) from the active phase, or a smooth handoff across them

        if isinstance(phases, int):
            return self.motion_priors[phases].reward(states)

        # when neither positions nor phases are given, the time steps stand in for the
        # robot's traversal of the track, with a uniform linspace covering every phase

        if not exists(positions) and not exists(phases):
            positions = self.auto_positions(states.shape[-2], prior_transition_positions)

        weights = self.phase_weights(positions, prior_transition_positions, phases)

        phase_rewards = stack([prior.reward(states) for prior in self.motion_priors])
        phase_rewards = rearrange(phase_rewards, 'n b ... -> b ... n')

        weights = pad_right_ndim_to(weights, phase_rewards.ndim)

        # a batch of one (a single flat position trace) is broadcast over the whole batch

        weights = weights.broadcast_to(phase_rewards.shape)

        return einsum('b ... n, b ... n -> b ...', phase_rewards, weights)

    def discriminator_loss(self, real_by_phase, fake_by_phase, return_loss_breakdown = False):
        # each phase discriminator trains against its own reference distribution

        if isinstance(real_by_phase, dict):
            real_by_phase = list(real_by_phase.values())
            fake_by_phase = list(fake_by_phase.values())

        assert len(real_by_phase) == len(fake_by_phase) == self.num_phases, f'one real and one fake transition tensor per phase: expected {self.num_phases} of each, got {len(real_by_phase)} real and {len(fake_by_phase)} fake'

        losses = [prior.discriminator_loss(real, fake) for prior, real, fake in zip(self.motion_priors, real_by_phase, fake_by_phase)]

        loss = sum(losses)

        if not return_loss_breakdown:
            return loss

        return loss, tuple(losses)

    def forward(self, states):
        # logits from every phase discriminator

        return stack([prior(states) for prior in self.motion_priors])

# training wrapper

class LightLocoParkour(Module):
    def __init__(
        self,
        agent: Agent
    ):
        super().__init__()
        self.agent = agent
