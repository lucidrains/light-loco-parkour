from __future__ import annotations
from typing import Sequence

import torch
from torch import nn, cat, stack, is_tensor, Tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch_einops_utils import (
    pad_left_at_dim,
    tree_map_tensor,
    pack_with_inverse,
    safe_cat
)

from x_mlps_pytorch import create_mlp

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def xnor(x, y):
    return x == y

# state encoder

class StateEncoder(Module):
    def __init__(
        self,
        dim,
        *,
        dim_state,
        num_stacked_frames = 5,
        tbptt_timesteps = 10
    ):
        super().__init__()

        self.num_stacked_frames = num_stacked_frames

        dim_state_total = dim_state * num_stacked_frames

        self.mlp_encoder = create_mlp(dim, dim_in = dim_state_total, depth = 2)

        self.rnn = nn.GRU(dim, dim, batch_first = True)

        self.tbptt_timesteps = tbptt_timesteps
        self.has_tbptt = tbptt_timesteps > 0

    def forward(
        self,
        states: Sequence[Tensor], # [b t ...]
        time_hiddens = None
    ):
        frames, tbptt_timesteps = self.num_stacked_frames, self.tbptt_timesteps

        # they simply pack all the states and use an MLP - they tried a conv encoder for the 2d depth map but saw no benefits

        states, _ = pack_with_inverse(states, 'b t *')

        # pad for the unfolding to get the stacked frames - first timestep will have 4 padded frames, second have 3, so on

        padded_states = pad_left_at_dim(states, frames - 1, dim = 1)

        stacked_states = padded_states.unfold(1, frames, 1)

        stacked_states = rearrange(stacked_states, 'b t ... -> b t (...)')

        # encode the states

        encoded_state = self.mlp_encoder(stacked_states)

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
        depth = 4,
        num_actions = 21,
    ):
        super().__init__()
        self.state_encoder = state_encoder

        self.backbone = create_mlp(dim, depth = 4)

        self.to_actions = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, num_actions * 2),
            Rearrange('... (d distr_params) -> ... d distr_params', distr_params = 2)
        )

    def forward(
        self,
        states: Sequence[Tensor],
        time_hiddens = None
    ):

        time_encoded_states, next_time_hiddens = self.state_encoder(states, time_hiddens)

        # backbone

        embed = self.backbone(time_encoded_states)

        # prediction actions

        action_dist = self.to_actions(embed)

        return action_dist, next_time_hiddens

# critic

class Critic(Module):
    def __init__(
        self,
        dim,
        *,
        num_skill_groups = 1,
        state_encoder: StateEncoder,
        depth = 4,
    ):
        super().__init__()
        self.state_encoder = state_encoder

        self.num_skill_groups = num_skill_groups
        dim_cond = 0 if num_skill_groups <= 1 else num_skill_groups

        self.backbone = create_mlp(dim, dim_in = dim_cond + dim, depth = depth)

        self.to_value_pred = Linear(dim, 1, bias = False)

    def forward(
        self,
        states: Sequence[Tensor],
        time_hiddens = None,
        skill_groups = None # int[b]
    ):
        num_skill_groups = self.num_skill_groups

        time_encoded_states, next_time_hiddens = self.state_encoder(states, time_hiddens)

        # take care of the skill group conditioning

        assert xnor(exists(skill_groups), num_skill_groups > 1)

        if num_skill_groups > 1:
            batch, time = time_encoded_states.shape[:2]

            if skill_groups.ndim == 0:
                skill_groups = repeat(skill_groups, ' -> b', b = batch)

            num_skill_groups = F.one_hot(skill_groups, num_skill_groups)
            num_skill_groups = repeat(num_skill_groups, 'b g -> b t g', t = time)
            time_encoded_states = cat((time_encoded_states, num_skill_groups), dim = -1)

        # backbone

        embed = self.backbone(time_encoded_states)

        # predict values

        values = self.to_value_pred(embed)

        values = rearrange(values, '... 1 -> ... ')

        return values, next_time_hiddens

# classes

class Agent(Module):
    def __init__(
        self,
        student_actor: Module,
        teacher_actor: Module,
        critic: Module
    ):
        super().__init__()

        self.student_actor = student_actor
        self.teacher_actor = teacher_actor
        self.critic = critic

# training wrapper

class LightLocoParkour(Module):
    def __init__(
        self,
        agent: Agent
    ):
        super().__init__()
        self.agent = agent
