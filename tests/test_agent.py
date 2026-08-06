import torch
from x_mlps_pytorch import create_mlp
from light_loco_parkour.light_loco_parkour import Actor, Critic, StateEncoder

def test_agent():
    actor = Actor(512, state_encoder = StateEncoder(512, dim_state = 4 + 5))

    critic = Critic(512, state_encoder = StateEncoder(512, dim_state = 4 + 5), num_skill_groups = 2)

    images = torch.randn(1, 3, 2, 2)
    proprio = torch.randn(1, 3, 5)

    decoder = create_mlp(16, dim_in = 512, dim_out = 17 * 11, depth = 2)
    privileged_target = torch.randn(1, 3, 17 * 11)

    pred_actions, next_hidden = actor((images, proprio))

    (pred_actions, next_hidden), aux_loss = actor((images, proprio), aux_decoder = decoder, aux_decoder_target = privileged_target)

    skill_groups = torch.randint(0, 2, (1,))
    values, _ = critic((images, proprio), skill_groups = skill_groups)

    assert pred_actions.shape == (1, 3, 21, 2)
    assert values.shape == (1, 3)

    aux_loss.backward()
