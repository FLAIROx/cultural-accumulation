import sys
import os
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
from gymnax_wrappers import LogWrapper, FlattenObservation
import functools
from gymnax.environments import spaces
from goal_cycle import GoalCycle
import matplotlib.pyplot as plt
from celluloid import Camera
from grid_viz import GridVisualizer


class ScannedRNN(nn.Module):

    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        # c, h = carry
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(ins.shape[0], ins.shape[1]), rnn_state)
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size)


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dir, dones = x
        embedding = nn.Conv(32, kernel_size=(3, 3), strides=3,
                            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = nn.Conv(64, kernel_size=(3, 3), strides=1,
                            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.leaky_relu(embedding)
        embedding = nn.Conv(64, kernel_size=(3, 3), strides=1,
                            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.leaky_relu(embedding)

        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)

        embedding = jnp.concatenate([embedding, dir], -1)
        embedding = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.tanh(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

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


class StatePredictor(nn.Module):

    @nn.compact
    def __call__(self, x):
        embedding = nn.Dense(576, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        embedding = nn.tanh(embedding)
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], 3, 3, -1)

        embedding = nn.ConvTranspose(64, kernel_size=(3, 3), strides=[1, 1],
                                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.leaky_relu(embedding)
        embedding = nn.ConvTranspose(32, kernel_size=(3, 3), strides=[1, 1],
                                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.leaky_relu(embedding)
        state_pred = nn.ConvTranspose(3, kernel_size=(3, 3), strides=[3, 3], padding=1,
                                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding)

        return state_pred


@struct.dataclass
class AgentParams:
    ac_params: core.FrozenDict
    aux_params: core.FrozenDict


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
    env = LogWrapper(env)

    config["CONTINUOUS"] = False

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        if config["CONTINUOUS"]:
            network = ActorCriticRNN(env.action_space('agent_0').shape[0], config=config)
        else:
            network = ActorCriticRNN(env.action_space('agent_0').n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space('agent_0').shape)),
                  jnp.zeros((1, config["NUM_ENVS"], 4)), jnp.zeros((1, config["NUM_ENVS"])))
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
        # network_params = network.init(_rng, init_hstate, init_x)
        # network_params = learner_state['params']['ac_params']
        network_params = demo_state['params']['ac_params']
        model = StatePredictor()
        if config['NUM_DEMOS'] != 0:
            demo_nets = {}
            demo_params = {}
            for i in range(config['NUM_DEMOS']):
                demo_nets['agent_'+str(i+1)] = ActorCriticRNN(env.action_space('agent_0').n, config=config)
                demo_params['agent_'+str(i+1)] = demo_states[i]['params']['ac_params']
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        _, _, _, init_embedding = network.apply(network_params, init_hstate, init_x)
        init_action = jnp.zeros((1, config["NUM_ENVS"], 3))  # *env.action_space(env_params).shape))
        init_s = jnp.concatenate((init_embedding, init_action), axis=2)
        train_state = TrainState.create(
            apply_fn=None,
            params=AgentParams(
                network_params,
                # model.init(_rng, init_s)
                demo_state['params']['aux_params']
                # learner_state['params']['aux_params']
            ),
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
        init_hstate_d = jnp.array(
            [ScannedRNN.initialize_carry(config["NUM_ENVS"], 256) for i in range(config['NUM_DEMOS'])])

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, hstate_d, rng, count = runner_state
                count += 1
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs['agent_0']['image'][np.newaxis, :],
                         last_obs['agent_0']['agent_dir'][np.newaxis, :],
                         last_done[np.newaxis, :])

                hstate, pi, value, embedding = network.apply(train_state.params.ac_params, hstate, ac_in)
                hs = jnp.repeat(jnp.zeros_like(hstate[jnp.newaxis, :, :]), config['NUM_DEMOS'], axis=0)
                values = jnp.repeat(jnp.zeros_like(value), config['NUM_DEMOS'], axis=0)
                keys = list(last_obs.keys())[1:]

                def _get_demo_inputs(idx, carry):
                    hs, values, actions, hstate_d = carry
                    key = 'agent_1'
                    # key = keys.at[idx].get()
                    demonstrator = demo_nets[key]
                    ac_in_d = (last_obs[key]['image'][np.newaxis, :],
                             last_obs[key]['agent_dir'][np.newaxis, :],
                             last_done[np.newaxis, :])
                    h = hstate_d.at[idx].get()
                    hstate_d, pi_d, value_d, embedding_d = demonstrator.apply(demo_params[key], h, ac_in_d)
                    hs = hs.at[idx].set(h)
                    values = values.at[idx].set(value_d.squeeze(0))
                    action_d = pi_d.sample(seed=_rng).squeeze(0)
                    actions = actions.at[idx].set(action_d)
                    hstate_d = hstate_d[jnp.newaxis, :, :]
                    return (hs, values, actions, hstate_d)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                actions = jnp.repeat(jnp.zeros_like(action), config['NUM_DEMOS'], axis=0)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)
                # hs, values, actions, hstate_d = jax.lax.fori_loop(0, config['NUM_DEMOS'], _get_demo_inputs,
                #                                              (hs, values, actions, hstate_d))

                # STEP ENV
                # actions = jnp.insert(actions, 0, action, axis=0).transpose()
                actions = jnp.array([action]).transpose()
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # penalty = 0.5 + ((count > (config["TOTAL_TIMESTEPS"] // 5)) * 0.5)
                penalty = 1.5
                penalty = jnp.repeat(penalty, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(
                    rng_step, env_state, actions, penalty
                )
                transition = Transition(done, actions, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, done, hstate, hstate_d, rng, count)

                # rendering
                # img = viz.render(env_params, env_state.env_state, 0)
                # imshow_obj = ax.imshow(img, interpolation='bilinear')
                # imshow_obj.set_data(img)
                # ax.text(0.5, 1.1, 'Reward: ' + str(reward[0]))
                # fig.canvas.draw()
                # camera.snap()

                return runner_state, transition

            initial_hstate = runner_state[-4]
            # viz = GridVisualizer()
            # fig, ax = plt.subplots()
            # camera = Camera(fig)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            # animation = camera.animate()
            # animation.save('animations/first_full_exp.gif')

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, hstate_d, rng, count = runner_state
            ac_in = (last_obs['agent_0']['image'][np.newaxis, :],
                     last_obs['agent_0']['agent_dir'][np.newaxis, :],
                     last_done[np.newaxis, :])
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
                                 traj_batch.done)
                        _, pi, value, embedding = network.apply(params.ac_params, init_hstate[0], ac_in)
                        action = traj_batch.action[:, :, 0]
                        action_onehot = jax.nn.one_hot(action, 3)
                        log_prob = pi.log_prob(traj_batch.action[:, :, 0])

                        state_pred = model.apply(
                            params.aux_params, jnp.concatenate((embedding, action_onehot), 2))

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
                        state_target = traj_batch.obs['agent_0']['image']

                        # def callback(state_pred, state_target):
                        #     plt.imshow(state_pred[100][5])
                        #     plt.savefig('aux_task/pred_recon_5')
                        #     plt.imshow(state_target[100][5])
                        #     plt.savefig('aux_task/target_recon_5')
                        #
                        # jax.debug.callback(callback, state_pred, state_target)

                        state_losses = jnp.abs(state_target[:-1] - state_pred[:-1])
                        aux_loss = state_losses.mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy + \
                            aux_loss * config["AUX_COEF"]
                        return total_loss, (value_loss, loss_actor, entropy, aux_loss)

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

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            # commented out line below as this is an eval script
            # train_state = update_state[0]

            metric = traj_batch.info
            rng = update_state[-1]

            if config["DEBUG"]:
                if config["LOG"] == "aux":
                    def callback(loss_info):
                        print(loss_info[1][-1].mean())

                    jax.debug.callback(callback, loss_info)

                else:
                    def callback(metric):
                        print(metric["returned_episode_returns"][-1, :].mean())

                    jax.debug.callback(callback, traj_batch.info)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, hstate_d, rng, count)
            return runner_state, (metric, loss_info)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate,
                        init_hstate_d, _rng, 0)
        runner_state, (metric, loss_info) = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return runner_state, metric, loss_info

    return train


if __name__ == "__main__":
    # with jax.disable_jit():
    config = {
        "LR": 1e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1.25e8,
        # "TOTAL_TIMESTEPS": 1e3,
        "UPDATE_EPOCHS": 8,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "AUX_COEF": 3,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "GoalCycle",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "NUM_DEMOS": 0,
        "LOG": "metric"
    }

    # INIT DEMONSTRATOR
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # learner_state = orbax_checkpointer.restore('tmp/orbax/social/recon_social_learner')#['model']
    demo_state = orbax_checkpointer.restore('tmp/orbax/experts/stage_5/curriculum_1e8_0')#['model']
    demo_states = [demo_state]

    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    out, metric, loss_info = train_jit(rng)
    train_state = out[0]
    jnp.save("results.npy", metric["returned_episode_returns"].mean(-1).reshape(-1))