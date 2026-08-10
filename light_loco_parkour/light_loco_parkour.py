from __future__ import annotations
from typing import Sequence
from beartype import beartype
from contextlib import nullcontext

import torch
from torch import nn, cat, stack, is_tensor, tensor, Tensor
from torch.nn import Module, Linear, RMSNorm
import torch.nn.functional as F

import einx
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from torch_einops_utils import (
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
        return raw_mean.sigmoid()

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

        # mean mapped to (0, 1)

        mean = raw_mean.sigmoid()

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
        assert values.shape[-1] == rewards.shape[-1]

        use_accelerated = default(use_accelerated, rewards.is_cuda)

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

        # maybe combine with aux loss

        if has_aux:
            aux_loss = reduce(aux_loss, '... d -> ...', 'mean') * self.aux_loss_weight
            distill_loss = reduce(distill_loss, '... d -> ...', 'mean')

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

# training wrapper

class LightLocoParkour(Module):
    def __init__(
        self,
        agent: Agent
    ):
        super().__init__()
        self.agent = agent
