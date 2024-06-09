import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from gymnax.environments import environment, spaces

from typing import NamedTuple, Optional, Tuple, Union


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservation(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        # assert isinstance(self._env.observation_space(params), spaces.Box), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self, key: chex.PRNGKey, seq: chex.Array
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, seq)
        # obs, state = self._env.reset(key)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvStateWithDemosComp:
    env_state: environment.EnvState
    episode_returns_1: float
    episode_returns_2: float
    episode_returns_demo: float
    episode_lengths: int
    returned_episode_returns_1: float
    returned_episode_returns_2: float
    returned_episode_returns_demo: float
    returned_episode_lengths: int


@struct.dataclass
class LogEnvStateWithDemos:
    env_state: environment.EnvState
    episode_returns: float
    episode_returns_demo: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_returns_demo: float
    returned_episode_lengths: int


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapperWithDemosComp(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvStateWithDemos(env_state, 0, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            # action: Union[int, float],
            action: chex.Array,
            penalty: float,
            prob_obs: float,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict, chex.Array]:
        obs, env_state, reward, done, info, abs_rewards = self._env.step(key, state.env_state, action, penalty, prob_obs)
        # print(reward.squeeze())
        new_episode_return_1 = state.episode_returns + abs_rewards[0].squeeze()
        new_episode_return_2 = state.episode_returns + abs_rewards[1].squeeze()
        mean_demo_return = jnp.mean(reward[2:])
        new_episode_return_demo = state.episode_returns_demo + mean_demo_return.squeeze()
        new_episode_length = state.episode_lengths + 1
        state = LogEnvStateWithDemos(
            env_state=env_state,
            episode_returns_1=new_episode_return_1 * (1 - done),
            episode_returns_2=new_episode_return_2 * (1 - done),
            episode_returns_demo=new_episode_return_demo * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns_1=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_returns_2=state.returned_episode_returns_2 * (1 - done) + new_episode_return_2 * done,
            returned_episode_returns_demo=state.returned_episode_returns_demo * (1 - done) + new_episode_return_demo * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
        )
        info["returned_episode_returns_1"] = state.returned_episode_returns_1
        info["returned_episode_returns_2"] = state.returned_episode_returns_2
        info["returned_episode_returns_demo"] = state.returned_episode_returns_demo
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info #, other_actions


class LogWrapperWithDemos(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self, key: chex.PRNGKey, seq: chex.Array
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, seq) 
        # obs, env_state = self._env.reset(key)
        state = LogEnvStateWithDemos(env_state, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            # action: Union[int, float],
            action: chex.Array,
            penalty: float,
            prob_obs: float,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict, chex.Array]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, penalty, prob_obs)
        # print(reward.squeeze())
        new_episode_return = state.episode_returns + reward[0].squeeze()
        # mean_demo_return = jnp.mean(reward[1:])
        # new_episode_return_demo = state.episode_returns_demo + mean_demo_return.squeeze()
        new_episode_return_demo = state.episode_returns_demo + reward[1].squeeze()
        new_episode_length = state.episode_lengths + 1
        state = LogEnvStateWithDemos(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_returns_demo=new_episode_return_demo * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_returns_demo=state.returned_episode_returns_demo * (1 - done) + new_episode_return_demo * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_returns_demo"] = state.returned_episode_returns_demo
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info #, other_actions


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            # action: Union[int, float],
            action: chex.Array,
            penalty: float,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, penalty)
        # print(reward.squeeze())
        new_episode_return = state.episode_returns + reward.squeeze()
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info
    
class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info
    
class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info
