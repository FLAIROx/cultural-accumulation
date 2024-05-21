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
# from environments.minimal_env_pretrain import GoalCycle
from environments.minimal_env import GoalCycle
import matplotlib.pyplot as plt
from celluloid import Camera
from environments.grid_viz import GridVisualizer
import wandb
from networks import ActorCriticRNN, ScannedRNN, StatePredictorSmall
from s5 import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel


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
        # obs, dir, dones, reward = x
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
        # embedding = jnp.concatenate([embedding, dir, reward], -1)
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

        if self.config["CONTINUOUS"]:
            actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), embedding


@struct.dataclass
class AgentParams:
    ac_params: core.FrozenDict
    # aux_params: core.FrozenDict


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def make_train(config):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = GoalCycle()
    env_params = env.params
    env = LogWrapperWithDemos(env)

    config["CONTINUOUS"] = False

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCriticS5(env.action_space('agent_0').n, config=config)
        demo_net_1 = ActorCriticS5(env.action_space('agent_1').n, config=config)
        # demo_net_1 = ActorCriticS5Demos(env.action_space('agent_1').n, config=config)
        # demo_net_2 = ActorCriticS5Demos(env.action_space('agent_2').n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space('agent_0').shape)),
                  jnp.zeros((1, config["NUM_ENVS"], 4)), jnp.zeros((1, config["NUM_ENVS"])),
                  jnp.zeros((1, config["NUM_ENVS"], 10)), jnp.zeros((1, config["NUM_ENVS"], 3)))
        # init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space('agent_0').shape)),
        #           jnp.zeros((1, config["NUM_ENVS"], 4)), jnp.zeros((1, config["NUM_ENVS"])), 
        #           jnp.zeros((1, config["NUM_ENVS"], 3)))
        init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
        network_params = network.init(_rng, init_hstate, init_x)
        # network_params = learner_state
        rng, _rng = jax.random.split(rng)
        if config['FIRST']:
            demo_params = [network.init(_rng, init_hstate, init_x)]
        else:
            demo_params = demo_params_ckpt
        # model = StatePredictorSmall()

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        _, _, _, init_embedding = network.apply(network_params, init_hstate, init_x)
        init_action = jnp.zeros((1, config["NUM_ENVS"], env.action_space('agent_0').n))
        init_s = jnp.concatenate((init_embedding, init_action), axis=2)

        train_state = TrainState.create(
            apply_fn=None,
            params=AgentParams(
                network_params,
                # model.init(_rng, init_s)
            ),
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstates, rng, count = runner_state
                count += 1
                rng, _rng = jax.random.split(rng)

                hstate, hstate_1 = hstates

                # SELECT ACTION
                ac_in = (last_obs['agent_0']['image'][np.newaxis, :],
                         last_obs['agent_0']['agent_dir'][np.newaxis, :],
                         last_done[np.newaxis, :],
                         last_obs['agent_0']['trial'][np.newaxis, :],
                         last_obs['agent_0']['reward'][np.newaxis, :])

                in_1 = (last_obs['agent_1']['image'][np.newaxis, :],
                        last_obs['agent_1']['agent_dir'][np.newaxis, :],
                        last_done[np.newaxis, :],
                        last_obs['agent_1']['trial_4'][np.newaxis, :],
                        last_obs['agent_1']['reward'][np.newaxis, :])

                # in_2 = (last_obs['agent_2']['image'][np.newaxis, :],
                #         last_obs['agent_2']['agent_dir'][np.newaxis, :],
                #         last_done[np.newaxis, :],
                #         last_obs['agent_2']['trial'][np.newaxis, :],
                #         last_obs['agent_2']['reward'][np.newaxis, :])

                hstate, pi, value, embedding = network.apply(train_state.params.ac_params, hstate, ac_in)
                hstate_1, pi_1, _, _ = demo_net_1.apply(demo_params[0], hstate_1, in_1)
                # hstate_2, pi_2, _, _ = demo_net_2.apply(demo_params[1], hstate_2, in_2)

                action = pi.sample(seed=_rng)
                rng, _rng = jax.random.split(rng)
                action_1 = pi_1.sample(seed=_rng).squeeze(0)
                # rng, _rng = jax.random.split(rng)
                # action_2 = pi_2.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)
                # actions = jnp.array([action, action_1, action_2]).transpose()
                actions = jnp.array([action, action_1]).transpose()

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # penalty = 0.5
                # frac = (train_state.step // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                #         ) / (config["NUM_UPDATES"] * 2 // 5)
                frac = (train_state.step // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                        ) / (config["NUM_UPDATES"])
                prob_obs = 1 - frac
                # prob_obs = jax.lax.select((frac > 0.25), 0.75, 1.0)
                # prob_obs = 1.0
                # penalty = jax.lax.select((frac < 0.05), 0.0, 0.25)
                # penalty = jax.lax.select(0.25 < frac, 1.0, penalty)
                penalty = 1.0
                # penalty = jax.lax.select(config['FIRST'], penalty, 1.5)
                # penalty = 1.5
                # prob_obs = 0.0
                prob_obs = jax.lax.select(config['FIRST'], 0.0, prob_obs)
                penalty = jnp.repeat(penalty, config["NUM_ENVS"])
                prob_obs = jnp.repeat(prob_obs, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))(
                    rng_step, env_state, actions, penalty, prob_obs
                )
                transition = Transition(done, actions, value, reward, log_prob, last_obs, info)
                hstates = jnp.array([hstate, hstate_1])
                runner_state = (train_state, env_state, obsv, done, hstates, rng, count)

                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstates, rng, count = runner_state
            ac_in = (last_obs['agent_0']['image'][np.newaxis, :],
                     last_obs['agent_0']['agent_dir'][np.newaxis, :],
                     last_done[np.newaxis, :],
                     last_obs['agent_0']['trial'][np.newaxis, :],
                     last_obs['agent_0']['reward'][np.newaxis, :])
            hstate = hstates[0]
            _, _, last_val, _ = network.apply(train_state.params.ac_params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward[:, 0]
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch,
                                             reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        ac_in = (traj_batch.obs['agent_0']['image'],
                                 traj_batch.obs['agent_0']['agent_dir'].reshape(
                                     traj_batch.obs['agent_0']['agent_dir'].shape[0],
                                     traj_batch.obs['agent_0']['agent_dir'].shape[1], -1),
                                 traj_batch.done,
                                 traj_batch.obs['agent_0']['trial'].reshape(
                                     traj_batch.obs['agent_0']['trial'].shape[0],
                                     traj_batch.obs['agent_0']['trial'].shape[1], -1),
                                 traj_batch.obs['agent_0']['reward'].reshape(
                                     traj_batch.obs['agent_0']['reward'].shape[0],
                                     traj_batch.obs['agent_0']['reward'].shape[1], -1)
                                 )
                        _, pi, value, embedding = network.apply(params.ac_params, init_hstate, ac_in)
                        action = traj_batch.action[:, :, 0]
                        action_onehot = jax.nn.one_hot(action, 3)

                        log_prob = pi.log_prob(action)

                        # state_pred = model.apply(
                        #     params.aux_params, jnp.concatenate((embedding.squeeze(), action_onehot), 2))

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"],
                                                                                                config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # CALCULATE AUXILIARY LOSS
                        # state_target = jnp.roll(traj_batch.obs['agent_0']['image'], -1, axis=0)
                        # state_losses = jnp.abs(state_target[:-1] - state_pred[:-1])
                        # aux_loss = state_losses.mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy # + \
                                     # aux_loss * config["AUX_COEF"]
                        return total_loss, (value_loss, loss_actor, entropy) # , aux_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(jnp.reshape(
                        x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])
                    ), 1, 0),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            init_hstate = initial_hstate[0]  # TBH
            init_hstate = [i for i in init_hstate]
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]

            metric = traj_batch.info
            rng = update_state[-1]

            if config["DEBUG"]:
                def callback(metric, loss_info):
                    print(metric["returned_episode_returns"][-1, :].mean())
                    wandb.log({'Mean Return': metric["returned_episode_returns"][-1, :].mean()})
                    wandb.log({'Mean Demo Return': metric["returned_episode_returns_demo"][-1, :].mean()})

                jax.debug.callback(callback, traj_batch.info, loss_info)

            runner_state = (train_state, env_state, last_obs, last_done, hstates, rng, count)
            return runner_state, (metric, loss_info)

        rng, _rng = jax.random.split(rng)
        init_hstates = jnp.array([init_hstate for i in range(1 + config['NUM_DEMOS'])])
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstates,
                        _rng, 0)
        runner_state, (metric, loss_info) = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return runner_state, metric, loss_info

    return train


if __name__ == "__main__":
    # with jax.disable_jit():
    config = {
        "LR": 1e-5,
        "NUM_ENVS": 128,
        "NUM_STEPS": 32,
        "TOTAL_TIMESTEPS": 1.5e7,
        "UPDATE_EPOCHS": 8,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "AUX_COEF": 0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "GoalCycle",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "NUM_DEMOS": 1,
        "LOG": "metric",
        "FIRST": False
    }

    wandb.init(
        project='culture',
        entity='flair',
        tags=['social_training', 'sequence']
    )
    wandb.config = config

    for i in range(1):

        stage_str = 'stage_' + str(i + 1)
        not_first_run = (i != 0)

        # INIT DEMONSTRATOR
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        if i == 0:
            checkpoint_loc_1 = 'tmp/orbax/experts/in_context_4_trials'
            checkpoint_loc_2 = checkpoint_loc_1
        else:
            checkpoint_loc_1 = 'tmp/orbax/experts/in_context_4_trials'
            # checkpoint_loc_2 = 'tmp/orbax/social/accumulation/sequence/stage_' + str(i) + '_more_goals'
        # checkpoint_loc_1 = 'tmp/orbax/experts/standard_seq_latest'
        demo_state_1 = orbax_checkpointer.restore(checkpoint_loc_1)['params']['ac_params']
        # demo_state_2 = orbax_checkpointer.restore(checkpoint_loc_2)['params']['ac_params']
        demo_params_ckpt = [demo_state_1, demo_state_1]
        # learner_loc = 'tmp/orbax/social/pretrained_social_4'
        # learner_loc = 'tmp/orbax/social/in_context_scratch_2'
        learner_loc = 'tmp/orbax/social/rl2_10_trials'
        learner_state = orbax_checkpointer.restore(learner_loc)['params']['ac_params']

        gpus = jax.devices('gpu')

        rng = jax.random.PRNGKey(42)
        # rng = jax.random.PRNGKey(90)
        rngs = jax.random.split(rng, 2)
        # train_jit = jax.vmap(jax.jit(make_train(config), device=gpus[4]))
        train_jit = jax.pmap(make_train(config), devices=gpus[:2])
        # out, metric, loss_info = train_jit(rngs[2:])
        out, metric, loss_info = train_jit(rngs)
        train_state = out[0]

        results = jnp.array([metric["returned_episode_returns"][i].mean(-1).reshape(-1) for i in range(2)])
        best_idx = jnp.argmax(jnp.mean(results[:, -1000:], axis=1))
        results = metric["returned_episode_returns"][best_idx].mean(-1).reshape(-1)
        # results_file = "accumulation_stage_" + str(i + 1) + '_sequence_demo'
        # results_path = "results/accumulation/" + results_file + ".npy"
        # jnp.save(results_path, results)

        model_config = {'name': 'AC-GRU'}
        # save social checkpoint
        best_model = jax.tree_util.tree_map(lambda x: x[best_idx], train_state)
        save_args = orbax_utils.save_args_from_target(best_model)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_loc = 'tmp/orbax/social/rl2_28_trial_no_expert_short'
        # save_loc = 'tmp/orbax/experts/pretrained_expert_2'
        orbax_checkpointer.save(save_loc, best_model, save_args=save_args)

        config["FIRST"] = False
        # config["TOTAL_TIMESTEPS"] = 1.5e8