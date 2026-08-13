# full pipeline e2e - exercises all modules: teacher / student actors, asymmetric critic,
# distillation, reward shaping, amp motion prior, and ppo

import torch
from torch import cat, tensor
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from einops import repeat, rearrange

from x_mlps_pytorch import create_mlp

from validate_transition import (
    MockTransitionEnv,
    sample_reference_transitions,
    sample_naive_transitions,
    split_fakes_by_phase,
    create_prior,
    train_discriminator,
    phase_margins
)

from light_loco_parkour import (
    Actor,
    Critic,
    Agent,
    StateEncoder,
    DistillationWrapper,
    LightLocoParkour,
    Gaussian,
    State,
    RewardHyperParams,
    RewardShapingWrapper,
    FootAccelerationPenalty,
    Discriminator,
    MotionPrior,
    PhaseConditionalMotionPrior,
    gradient_penalty,
    exists
)

# constants

BATCH = 4
TIME = 8
DIM = 512

DIM_PROPRIO = 5
DIM_DEPTH = 64
DIM_SCAN = 64

NUM_SKILL_GROUPS = 3
NUM_ACTIONS = 21

# helpers

def mock_state(batch = BATCH, dim_joints = NUM_ACTIONS, num_rays = 8, num_links = 6):
    # realistic magnitudes, so the rewards stay in a sane range (table 1)
    return State(
        linear_velocity = torch.randn(batch, 3) * 0.5,
        angular_velocity = torch.randn(batch, 3) * 0.5,
        projected_gravity = torch.randn(batch, 3) * 0.05,
        commanded_velocity = torch.randn(batch, 3) * 0.5,
        joint_limit_flags = torch.rand(batch, dim_joints) > 0.99,
        contact_forces = torch.rand(batch, num_links) * 0.5,
        foot_contact = torch.rand(batch, 2) > 0.5,
        foot_heights = torch.rand(batch, 2),
        foot_ray_hit_heights = torch.rand(batch, 2, num_rays) * 0.3,
        foot_acceleration = torch.rand(batch, 2, 3) * 10.,
        heading_error = torch.rand(batch) * 0.1,
        action_rate = torch.randn(batch, dim_joints) * 0.1
    )

def test_full_pipeline_e2e():
    # 1. teacher (privileged height scan) and student (onboard depth, recurrent)

    teacher_actor = Actor(
        DIM,
        state_encoder = StateEncoder(DIM, dim_state = DIM_PROPRIO + DIM_SCAN),
        num_skill_groups = NUM_SKILL_GROUPS,
        action_distr = Gaussian()
    )

    student_actor = Actor(
        DIM,
        state_encoder = StateEncoder(DIM, dim_state = DIM_PROPRIO + DIM_DEPTH, use_rnn = True),
        action_distr = Gaussian()
    )

    # 2. asymmetric critic on privileged observations

    critic = Critic(
        DIM,
        state_encoder = StateEncoder(DIM, dim_state = DIM_PROPRIO + DIM_SCAN),
        num_skill_groups = NUM_SKILL_GROUPS
    )

    agent = Agent(
        student_actor,
        critic,
        actor_state_keys = ('proprio', 'depth'),
        critic_state_keys = ('proprio', 'scan')
    )

    trainer = LightLocoParkour(agent)

    # 3. distill the height-scan teacher onto the depth student (section vi)

    aux_decoder = create_mlp(DIM_SCAN, dim_in = DIM, depth = 2)

    distill = DistillationWrapper(
        student = student_actor,
        teacher = teacher_actor,
        student_state_keys = ('proprio', 'depth'),
        teacher_state_keys = ('proprio', 'scan'),
        aux_decoder = aux_decoder,
        privileged_state_key = 'scan',
        aux_loss_weight = 0.1
    )

    # 4. reward shaping (section iv-c1) and motion prior (section v-c2)

    reward_shaping = RewardShapingWrapper(
        reward_hparams = RewardHyperParams(),
        stateful_reward_fns = ((FootAccelerationPenalty(), 0.01),)
    )

    motion_prior = MotionPrior(Discriminator(DIM, dim_in = DIM_PROPRIO + DIM_DEPTH))

    # the zero-centered gradient penalty (r3gan), configurable to one-centered (wgan-gp) via grad_penalty_center

    demo_inputs = torch.randn(4, 8, DIM_PROPRIO + DIM_DEPTH, requires_grad = True)
    demo_logits = motion_prior.discriminator(demo_inputs)
    assert gradient_penalty(demo_inputs, demo_logits, weight = 1., center = 0.) > 0.

    # 5. optimizers - the teacher is frozen, distilled onto the student

    policy_parameters = [*student_actor.parameters(), *critic.parameters(), *aux_decoder.parameters()]
    discriminator_parameters = list(motion_prior.discriminator.parameters())

    policy_optimizer = AdamW(policy_parameters, lr = 1e-4)
    discriminator_optimizer = AdamW(discriminator_parameters, lr = 1e-4)

    # 6. training loop

    for _ in range(3):
        # mock onboard observations

        proprio = torch.randn(BATCH, TIME, DIM_PROPRIO)
        depth = torch.randn(BATCH, TIME, DIM_DEPTH)
        scan = torch.randn(BATCH, TIME, DIM_SCAN)

        states = dict(
            proprio = proprio,
            depth = depth,
            scan = scan
        )

        # sample actions from the student, estimate values from the critic
        # under no_grad, so the old log probs and value targets are detached

        with torch.no_grad():
            actions, old_log_probs, _ = trainer.agent.forward_actor(states, sample_action = True, return_log_prob = True)
            values, _ = trainer.agent.forward_critic(states, skill_groups = 1)

        # reward shaping (table 1) and generalized advantage estimation

        rewards = repeat(reward_shaping(mock_state()), 'b -> b t', t = TIME)
        returns = trainer.agent.calc_gae(rewards, values, torch.ones(BATCH, TIME))
        advantages = returns - values

        # ppo objectives

        policy_loss = trainer.agent.actor_loss(states, actions, old_log_probs, advantages, skill_groups = 1)
        value_loss = trainer.agent.critic_loss(states, returns, skill_groups = 1)

        # distillation, with reconstruction of the privileged height scan

        distill_loss = distill(states, teacher_skill_groups = 1)

        # adversarial motion prior (section v-c2) - real transitions from the learned behaviors,
        # fake transitions from the student policy, both in the same deployable observation space

        real_proprio = torch.randn(BATCH, TIME, DIM_PROPRIO)
        real_depth = torch.randn(BATCH, TIME, DIM_DEPTH)

        real_transitions = cat((real_proprio, real_depth), dim = -1)
        fake_transitions = cat((proprio, depth), dim = -1)

        amp_reward = motion_prior.reward(fake_transitions).detach()
        discriminator_loss = motion_prior.discriminator_loss(real_transitions, fake_transitions)

        # policy side update

        policy_optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss + distill_loss - 0.01 * amp_reward.mean()).backward()
        clip_grad_norm_(policy_parameters, 1.)
        policy_optimizer.step()

        # discriminator side update

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        clip_grad_norm_(discriminator_parameters, 1.)
        discriminator_optimizer.step()

        assert torch.isfinite(policy_loss)
        assert torch.isfinite(value_loss)
        assert torch.isfinite(distill_loss)
        assert torch.isfinite(discriminator_loss)
        assert amp_reward.mean() > 0.

# phase conditional motion prior - one discriminator per phase, handed off at the transition positions

def test_phase_conditional_motion_prior():
    def prior():
        return MotionPrior(Discriminator(DIM, dim_in = 64))

    run, leap = prior(), prior()
    conditional = PhaseConditionalMotionPrior([run, leap], prior_transition_positions = (0.5,))

    states = torch.randn(BATCH, TIME, 64)
    positions = tensor([[0.1, 0.3, 0.49, 0.55, 0.7, 0.8, 0.9, 1.0]] * BATCH)

    # automatic phase resolution matches explicit phases, and routes the reward accordingly

    phases = (positions >= 0.5).long()
    assert torch.equal(conditional.resolve_phases(positions = positions), phases)
    assert torch.allclose(conditional.reward(states, positions = positions), conditional.reward(states, phases = phases))

    # a flat position trace (no batch dim) is broadcast over the whole batch

    flat = positions[0]
    assert torch.equal(conditional.resolve_phases(positions = flat), rearrange(phases[0], 'n -> 1 n'))
    assert torch.allclose(conditional.reward(states, positions = flat), conditional.reward(states, positions = positions))

    # without positions or phases, the time steps stand in for the traversal, with a
    # uniform linspace from the start of the track to one spacing past the last trigger

    auto = conditional.auto_positions(TIME)
    expected = torch.linspace(0., 1., TIME)
    assert torch.allclose(auto, expected)
    assert torch.allclose(conditional.reward(states), conditional.reward(states, positions = auto))

    # an int phase routes straight to that discriminator

    assert torch.allclose(conditional.reward(states, phases = 1), leap.reward(states))

    # phases chain across multiple transition positions

    three = PhaseConditionalMotionPrior([prior(), prior(), prior()], prior_transition_positions = (0.5, 1.0))
    assert torch.equal(three.resolve_phases(positions = tensor([[0.1, 0.6, 1.2]])), tensor([[0, 1, 2]]))

    # the smooth handoff weights sum to one and shift gradually across the transition

    smooth = PhaseConditionalMotionPrior([run, leap], prior_transition_positions = (0.5,), smooth_handoff = True, handoff_temperature = 2.)
    weights = smooth.phase_weights(positions = positions)
    assert torch.allclose(weights.sum(dim = -1), torch.ones_like(weights[..., 0]))
    assert (weights[..., 1][:, 0] < 0.5).all() and (weights[..., 1][:, -1] > 0.5).all()

    # a single prior is replicated with shared weights across the phases

    shared = PhaseConditionalMotionPrior(prior(), prior_transition_positions = (0.5,))
    assert shared.num_phases == 2 and shared.motion_priors[0] is shared.motion_priors[1]
    assert torch.allclose(shared.reward(states, positions = positions), shared.motion_priors[0].reward(states))

    # each phase discriminator trains against its own references

    loss = conditional.discriminator_loss(
        [torch.randn(BATCH, TIME, 64), torch.randn(BATCH, TIME, 64)],
        [torch.randn(BATCH, TIME, 64), torch.randn(BATCH, TIME, 64)]
    )
    loss.backward()

    for prior in conditional.motion_priors:
        assert all(exists(p.grad) for p in prior.discriminator.parameters())

# mock transition world - each phase's amp reward must prefer its own reference pattern,
# while a single unconditional prior cannot serve every phase

def test_phase_conditional_amp_reward():
    torch.manual_seed(42)

    env = MockTransitionEnv()
    env.reset(seed = 42)

    obs_dim = env.observation_space.shape[0]

    # naive policy transitions provide the fakes

    fakes = sample_naive_transitions(env, 2048)

    real_by_phase = [sample_reference_transitions(env, k, 256) for k in range(env.num_phases)]

    # every phase's reward must prefer its own reference pattern

    conditional_prior = create_prior(64, obs_dim, env)

    train_discriminator(
        conditional_prior,
        real_by_phase,
        split_fakes_by_phase(conditional_prior, fakes, fakes[..., 0]),
        AdamW(conditional_prior.parameters(), lr = 2e-3),
        steps = 150
    )

    margins = phase_margins(conditional_prior, env, conditional = True)

    assert min(margins) > 0.5, f'each phase must prefer its own pattern, got margins {margins}'

    # a single unconditional prior cannot serve every phase

    unconditional_prior = MotionPrior(Discriminator(64, dim_in = 2 * obs_dim), grad_penalty_weight = 1.)

    train_discriminator(
        unconditional_prior,
        torch.cat(real_by_phase),
        fakes,
        AdamW(unconditional_prior.parameters(), lr = 2e-3),
        steps = 150
    )

    unconditional_margins = phase_margins(unconditional_prior, env, conditional = False)

    assert min(unconditional_margins) <= 0.5, f'unconditional prior must underserve some phase, got margins {unconditional_margins}'
