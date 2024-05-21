import sys
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint
import flax
from flax.training import orbax_utils
from flax import struct, core
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
# from wrappers import FlattenObservation, LogWrapper, TransformObservation, TransformReward, BraxGymnaxWrapper
from environments.gymnax_wrappers import LogWrapperWithDemos, FlattenObservation
import functools
from gymnax.environments import spaces
from environments.minimal_env import GoalCycle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from celluloid import Camera
from environments.grid_viz import GridVisualizer
import wandb
from networks import ActorCriticRNN, ScannedRNN, StatePredictorSmall
from s5 import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel
from scipy.interpolate import make_interp_spline, BSpline


d_model = 256
ssm_size = 256
C_init = "lecun_normal"
discretization="zoh"
dt_min=0.001
dt_max=0.1
n_layers = 4
conj_sym=True
clip_eigs=False
bidirectional=False

blocks = 1
block_size = int(ssm_size / blocks)

Lambda, _, B, V, B_orig = make_DPLR_HiPPO(ssm_size)

block_size = block_size // 2
ssm_size = ssm_size // 2

Lambda = Lambda[:block_size]
V = V[:, :block_size]

Vinv = V.conj().T

ssm_init_fn = init_S5SSM(H=d_model,
                            P=ssm_size,
                            Lambda_re_init=Lambda.real,
                            Lambda_im_init=Lambda.imag,
                            V=V,
                            Vinv=Vinv,
                            C_init=C_init,
                            discretization=discretization,
                            dt_min=dt_min,
                            dt_max=dt_max,
                            conj_sym=conj_sym,
                            clip_eigs=clip_eigs,
                            bidirectional=bidirectional)


class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dir, dones, trial, reward = x
        embedding = nn.Conv(32, kernel_size=(1, 1), strides=1, padding=0,
                            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = nn.Conv(64, kernel_size=(1, 1), strides=1, padding=0,
                            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.leaky_relu(embedding)
        embedding = nn.Conv(64, kernel_size=(1, 1), strides=1, padding=0,
                            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.leaky_relu(embedding)

        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)

        embedding = jnp.concatenate([embedding, dir, trial, reward], -1)
        embedding = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.tanh(embedding)

        # hidden, embedding = self.s5(hidden, embedding, dones)
        hidden, embedding = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=d_model,
            n_layers=n_layers,
            activation="half_glu1",
        )(hidden, embedding, dones)

        actor_mean = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), embedding

def eval_and_return(rng, reset_rng, demo_actions, prev_best):
    last_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, 0)

    init_state = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)

    action_sequences = jnp.zeros((config["NUM_ENVS"], 4, env_params.max_episode_steps))

    # COLLECT TRAJECTORIES
    def _env_step(runner_state, unused):
        prev_env_state, last_obs, last_done, prev_hstate, hstate_1, rng, save_return, returns, demo_returns, demo_scores, saved_scores, action_sequences = runner_state
        rng, _rng = jax.random.split(rng)

        # SELECT ACTION
        ac_in = (last_obs['agent_0']['image'][np.newaxis, :],
                 last_obs['agent_0']['agent_dir'][np.newaxis, :],
                 last_done[np.newaxis, :],
                 last_obs['agent_0']['trial'][np.newaxis, :],
                 last_obs['agent_0']['reward'][np.newaxis, :])

        in_1 = (last_obs['agent_1']['image'][np.newaxis, :],
                last_obs['agent_1']['agent_dir'][np.newaxis, :],
                last_done[np.newaxis, :],
                last_obs['agent_1']['trial'][np.newaxis, :],
                last_obs['agent_1']['reward'][np.newaxis, :])

        hstate, pi, _, _ = network.apply(params, prev_hstate, ac_in)
        hstate_1, pi_1, _, _ = network.apply(demo_params, hstate_1, in_1)

        rng, _rng = jax.random.split(_rng)
        action = pi.sample(seed=_rng).squeeze(0)
        # action = jax.lax.select(prev_env_state.time[0] == 0, jnp.ones_like(action) * 2, action)
        rng, _rng = jax.random.split(rng)
        # action_1 = pi_1.sample(seed=_rng).squeeze(0)
        action_1 = demo_actions.at[:, prev_env_state.time[0]].get()
        actions = jnp.array([action, action_1]).transpose().astype(int)
        action_sequences = action_sequences.at[:, prev_env_state.trial[0], prev_env_state.time[0]].set(action)

        # STEP ENV
        rng_step = jax.random.split(rng, config["NUM_ENVS"])
        penalty = jnp.repeat(1.0, config["NUM_ENVS"])
        prob_obs = jax.lax.select(config["FIRST"], 0.0, 1.0)
        prob_obs = jnp.repeat(prob_obs, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))(
            rng_step, prev_env_state, actions, penalty, prob_obs
        )

        returns += reward[:, 0]
        demo_returns += reward[:, 1]
        save_return += reward[:, 0]
        save_return *= (env_state.time != 0)
        saved_scores = saved_scores.at[:, env_state.trial[0]].set(save_return)
        demo_scores = demo_scores.at[:, env_state.trial[0]].set(demo_returns)
        # returns *= (env_state.time[0] != 0)
        # demo_returns *= (env_state.time[0] != 0)

        transition = (env_state, action, reward, hstate)
        runner_state = (env_state, obsv, done, hstate, hstate_1, rng, save_return, returns, demo_returns, demo_scores, saved_scores, action_sequences)
        return runner_state, transition

    runner_state = (env_state, last_obs, jnp.zeros((config["NUM_ENVS"],)).astype(bool), init_state, init_state, rng,
                    jnp.zeros((config["NUM_ENVS"],)), jnp.zeros((config["NUM_ENVS"],)), jnp.zeros((config["NUM_ENVS"],)),
                    jnp.zeros((config["NUM_ENVS"], 4)), jnp.zeros((config["NUM_ENVS"], 4)), action_sequences)
    runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, env_params.max_episode_steps*4)
    saved_scores = runner_state[-2]
    action_sequences = runner_state[-1]
    demo_scores = runner_state[-3]
    best = jnp.argmax(saved_scores, axis=1)
    scores = saved_scores
    idxs = jnp.arange(config["NUM_ENVS"])
    new_demo_actions = action_sequences.at[idxs, best, :].get()
    # new_demo_actions = action_sequences.at[idxs, -1, :].get()
    new_best = jnp.max(saved_scores, axis=1)
    improvement = (new_best >= prev_best)
    demo_actions = improvement[:, jnp.newaxis] * new_demo_actions + ~improvement[:, jnp.newaxis] * demo_actions
    prev_best = improvement * new_best + ~improvement * prev_best

    return runner_state, (scores, prev_best, demo_actions, demo_scores)


config = {
    "NUM_ENVS": 10,
    "NUM_DEMOS": 1,
    "NUM_STEPS": 120,
    "FIRST": True
}

rng = jax.random.PRNGKey(5)
env = GoalCycle()
env_params = env.params

init_state = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
init_hstate_1 = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
network = ActorCriticS5(env.action_space('agent_0').n, config=config)
checkpoint_loc = 'tmp/orbax/social/4_trial_ext_perf'
# checkpoint_loc = 'tmp/orbax/social/in_context_scratch_2'
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
params = orbax_checkpointer.restore(checkpoint_loc)['params']['ac_params']
# checkpoint_loc = 'tmp/orbax/experts/in_context_demo'
demo_params = orbax_checkpointer.restore(checkpoint_loc)['params']['ac_params']
rng, _rng = jax.random.split(rng)
demo_actions = jax.random.randint(_rng, (config["NUM_ENVS"], env_params.max_episode_steps), 0, 3)
prev_best = jnp.zeros(config["NUM_ENVS"])

reset_rng = jax.random.split(rng, config["NUM_ENVS"])
avg_scores = []
total_scores = []
stds = []
for i in range(7):
    rng, _rng = jax.random.split(rng)
    eval_jit = jax.jit(eval_and_return)
    # print(demo_actions[0])
    rngs = jax.random.split(rng, 2)
    runner_state, out = eval_jit(_rng, reset_rng, demo_actions, prev_best)
    scores = out[0]
    prev_best = out[1]
    demo_actions = out[2]
    print(demo_actions.shape)
    demo_scores = out[3]
    # print(demo_actions.shape)
    # print(demo_actions[0][:5])
    # print(prev_best.mean())
    # print("demo:", jnp.mean(jnp.sum(demo_scores, axis=1)))
    avg_scores.append(jnp.mean(scores, axis=0))
    stds.append(jnp.std(scores, axis=0))
    # print("social_scores:", jnp.mean(jnp.sum(scores, axis=1)))
    demo_score = runner_state[8]
    social_score = runner_state[7]
    print("demo:", jnp.mean(demo_score))
    print("social:", jnp.mean(social_score))
    total_scores.append(jnp.mean(social_score))
    config["FIRST"] = False

members = avg_scores
n = len(members)
colors = cm.viridis([i / n for i in range(n)])  # Use the viridis colormap

fig = plt.figure()
ax = plt.subplot(111)

for i, (m, color) in enumerate(zip(members, colors)):
    # best = jnp.argmax(m[...,-1])
    # best_vals.append(m[best][...,-1])
    # m = m[best]
    error = stds[i] / 10
    if i == 0 or i == 6:
        ax.plot(jnp.arange(i * 4, i * 4 + 4), m, label=f"Generation {i}", color=color, alpha=1)
    else:
        ax.plot(jnp.arange(i * 4, i * 4 + 4), m, color=color, alpha=1)
    ax.fill_between(jnp.arange(i * 4, i * 4 + 4), m - error, m + error, alpha=0.1, color=color)


rl2_10 = jnp.load("rl2_10_long.npy")
rl2_28 = jnp.load("rl2_28.npy")
std_28 = jnp.load("rl2_28_std.npy")
rl2_4 = jnp.load("rl2_4_long.npy")
std_4 = jnp.load("rl2_4_std.npy")
oracle = jnp.load("oracle.npy")
# # rl2_10_std = jnp.load("rl2_10_test_std.npy")

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

rl2_4_smooth = smooth(rl2_4[:28], .7)
rl2_28_smooth = smooth(rl2_28[:28], .7)

# ax.scatter(np.arange(28), rl2_10[:28], label="RL2 10 Trials")
# rl2_4 = [np.max(rl2_4[:28])] * 28
# rl2_20 = [np.max(rl2_20[:28])] * 28
oracle = [np.mean(oracle[:4])] * 28
# oracle = [4.81] * 28
ax.plot(rl2_4_smooth, linestyle='dashed', label="RL\u00b2 4 Trials", alpha=0.5)
error = std_4 / 10
ax.fill_between(jnp.arange(28), np.array(rl2_4_smooth) - error, np.array(rl2_4_smooth) + error, alpha=0.05)
ax.plot(rl2_28_smooth, linestyle='dashed', label="RL\u00b2 28 Trials", alpha=0.5)
error = std_28 / 10
ax.fill_between(jnp.arange(28), np.array(rl2_28_smooth) - error, np.array(rl2_28_smooth) + error, alpha=0.05)
ax.plot(oracle[:28], linestyle='dashed', label="Noisy Oracle", color='black')
ax.set_ylim(ymin=0)

# plt.plot(total_scores)
fig.legend(bbox_to_anchor=(0.9, 0.35))
# fig.legend(loc="lower right")
plt.ylabel("Return")
plt.xlabel("Trial")
plt.title("In-Context Accumulation: Goal Sequence")
# plt.show()
plt.savefig("test_new", bbox_inches='tight')

# for i, (m, color) in enumerate(zip(members, colors)):
#     # best = jnp.argmax(m[...,-1])
#     # best_vals.append(m[best][...,-1])
#     # m = m[best]
#     error = stds[i] / np.sqrt(10)
#     ax.plot(jnp.arange(i * 4, i * 4 + 4), m, label=f"Generation {i}", color=color, alpha=1)
#     ax.fill_between(jnp.arange(i * 4, i * 4 + 4), m - error, m + error, alpha=0.1, color=color)
#
# ax.plot(rl2_4, linestyle='dashed', label="RL\u00b2 4 Trials", alpha=0.5)
# ax.fill_between(jnp.arange(28), np.array(rl2_4) - rl2_4_error, np.array(rl2_4) + rl2_4_error, alpha=0.05)
# ax.plot(rl2_28, linestyle='dashed', label="RL\u00b2 28 Trials", alpha=0.5)
# ax.fill_between(jnp.arange(28), np.array(rl2_28) - rl2_28_error, np.array(rl2_28) + rl2_28_error, alpha=0.05)
# ax.plot(oracle, linestyle='dashed', label="Noisy Oracle", color='black')