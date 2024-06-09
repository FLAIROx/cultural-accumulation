import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Tuple, Any
from collections import OrderedDict
import chex
from functools import partial
from flax import struct
from enum import IntEnum
from multi_agent_env import MultiAgentEnv
from gymnax.environments.spaces import Box, Discrete
import numpy as np
from common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    DIR_TO_VEC,
    make_map,
    OBJECT_ARRAY,
    GOAL_ARRAY)


INDEX_TO_COLOR = [k for k, _ in COLOR_TO_INDEX.items()]


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


@struct.dataclass
class EnvState:
    agents_pos: chex.Array
    agents_dir: chex.Array
    agents_dir_idx: chex.Array
    goals_pos: chex.Array
    wall_map: chex.Array
    map: chex.Array
    goals_map: chex.Array
    agents_map: chex.Array
    time: int
    terminal: bool
    scores: chex.Array
    sequence: chex.Array
    hit_target: chex.Array
    dropped: bool
    trial: int
    prob_obs: float
    reset_key: chex.PRNGKey
    reward: chex.Array
    scrambled: bool


@struct.dataclass
class EnvParams:
    num_agents: int = 2
    height: int = 7
    width: int = 7
    n_walls: int = 0
    n_goals: int = 3
    agent_view_size: int = 5
    replace_wall_pos: bool = False
    see_through_walls: bool = True
    see_agent: bool = True
    normalize_obs: bool = True
    sample_n_walls: bool = False
    max_episode_steps: int = 50


class GoalSequence(MultiAgentEnv):
    def __init__(self):
        super().__init__(num_agents=self.params.num_agents)
        self.agents = [f"agent_{i}" for i in range(self.params.num_agents)]
        self.agent_range = jnp.arange(self.params.num_agents)

        self.obs_shape = (self.params.agent_view_size, self.params.agent_view_size, 5)
        self.action_set = jnp.array([
            Actions.left,
            Actions.right,
            Actions.forward
        ])

        self.observation_spaces = {i: Box(0, 1, self.obs_shape) for i in self.agents}
        self.action_spaces = {i: Discrete(len(self.action_set)) for i in self.agents}

    @property
    def params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            actions: dict,
            penalty: float,
            prob_obs: float
    ) -> Tuple[Dict, EnvState, float, bool, Dict]:
        """Perform single timestep state transition."""
        a = jnp.array([self.action_set[action] for action in actions])
        prob_obs = state.prob_obs
        state, rewards = self.step_agents(key, state, a, penalty, prob_obs)
        # Check game condition & no. steps for termination condition
        time = state.time + 1
        state = state.replace(time=time)
        new_trial = (time > self.params.max_episode_steps)
        trial = state.trial + new_trial
        done = self.is_terminal(state)
        state = state.replace(terminal=done)
        _, new_state = self.reset_env(state.reset_key, trial=trial, sequence_inp=state.sequence)
        reset_cond = jnp.logical_and(new_trial, jnp.logical_not(state.terminal))

        def reset_fn(state, new_state):
            return new_state

        def cont_fn(state, new_state):
            return state

        state = lax.cond(reset_cond, reset_fn, cont_fn, state, new_state)

        info = {"returned_episode": done}

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            rewards,
            done,
            info,
        )

    def reset_env(self, key: chex.PRNGKey, trial=0, sequence_inp=jnp.zeros(100)) -> Tuple[Dict, Any]:
        """Reset environment state by resampling contents of maze_map
        - initial agent position
        - goal position
        - wall positions
        """
        params = self.params
        h = params.height
        w = params.width
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        # Reset wall map, with shape H x W, and value of 1 at (i,j) iff there is a wall at (i,j)
        reset_key = key
        key, subkey = jax.random.split(key)
        wall_idx = jax.random.choice(
            subkey, all_pos,
            shape=(params.n_walls,),
            replace=params.replace_wall_pos)

        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent positions + dirs
        key, subkey = jax.random.split(key)
        # Don't spawn demo on social pos
        occupied_mask = occupied_mask.at[21].set(1)
        occupied_mask = occupied_mask.at[22].set(1)
        agents_idx = jax.random.choice(subkey, all_pos, shape=(self.num_agents,),
                                       p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32))
        agents_pos = jnp.transpose(jnp.array([agents_idx % w, agents_idx // w], dtype=jnp.uint32))
        # Replace with fixed start position
        agents_pos = agents_pos.at[0].set([0, 3])
        agents_pos = agents_pos.at[1].set([1, 3])

        agents_mask = jnp.zeros_like(all_pos)
        agents_mask = agents_mask.at[agents_idx].set(1)
        agents_map = agents_mask.reshape(h, w).astype(jnp.bool_)

        key, subkey = jax.random.split(key)
        agents_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.uint8),
                                           shape=(self.num_agents,))
        # Replace with fixed initial orientations
        agents_dir_idx = agents_dir_idx.at[0].set(0)
        agents_dir_idx = agents_dir_idx.at[1].set(0)
        agents_dir = DIR_TO_VEC.at[agents_dir_idx].get()

        # Reset goal positions
        key, subkey = jax.random.split(key)
        goals_idx = jax.random.choice(subkey, all_pos, shape=(params.n_goals,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32))
        goals_pos = jnp.transpose(jnp.array([goals_idx % w, goals_idx // w], dtype=jnp.uint32))

        goals_mask = jnp.zeros_like(all_pos)
        goals_mask = goals_mask.at[goals_idx].set(1)
        goals_map = goals_mask.reshape(h, w).astype(jnp.bool_)

        key, subkey = jax.random.split(key)
        sequence = jax.random.randint(subkey, (100,), 0, params.n_goals)
        sequence = lax.select((trial == 0), sequence.astype(int), sequence_inp.astype(int))
        scores = jnp.zeros(params.num_agents)

        hit_target = jnp.zeros(params.num_agents).astype(jnp.bool_)

        prob_obs = 1 - (trial / 3)

        map = make_map(
            params,
            wall_map,
            goals_pos,
            agents_pos,
            agents_dir_idx,
            pad_obs=True)

        state = EnvState(
            agents_pos=agents_pos,
            agents_dir=agents_dir,
            agents_dir_idx=agents_dir_idx,
            goals_pos=goals_pos,
            wall_map=wall_map.astype(jnp.bool_),
            map=map,
            agents_map=agents_map.astype(jnp.bool_),
            goals_map=goals_map.astype(jnp.bool_),
            time=0,
            terminal=False,
            sequence=sequence,
            scores=scores,
            hit_target=hit_target,
            dropped=False,
            trial=trial,
            prob_obs=prob_obs,
            reset_key=reset_key,
            reward=jnp.zeros(params.num_agents),
            scrambled=False
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: EnvState) -> Dict:

        def _observation(aidx: int, state: EnvState) -> Dict:
            """Return limited grid view ahead of agent."""
            obs = jnp.zeros(self.obs_shape, dtype=jnp.uint8)

            obs_fwd_bound1 = state.agents_pos[aidx]
            obs_fwd_bound2 = state.agents_pos[aidx] + state.agents_dir[aidx] * (self.obs_shape[0] - 1)

            side_offset = self.obs_shape[0] // 2
            obs_side_bound1 = state.agents_pos[aidx] + (state.agents_dir[aidx] == 0) * side_offset
            obs_side_bound2 = state.agents_pos[aidx] - (state.agents_dir[aidx] == 0) * side_offset

            all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])

            # Clip obs to grid bounds appropriately
            padding = obs.shape[0] - 1
            pad = jnp.array([padding])
            obs_bounds_min = np.min(all_bounds, 0) + padding
            obs_range_x = jnp.arange(obs.shape[0]) + obs_bounds_min[1]
            obs_range_y = jnp.arange(obs.shape[0]) + obs_bounds_min[0]

            meshgrid = jnp.meshgrid(obs_range_y, obs_range_x)
            coord_y = meshgrid[1].flatten()
            coord_x = meshgrid[0].flatten()

            # Remove other agents from demonstrator obs
            def true_fn():
                # Only observe object component
                obs_map = state.map[:, :, 0]
                # Convert indices to objects
                obs_map = jax.tree_map(lambda idx: OBJECT_ARRAY[idx], obs_map)
                demo_obj = lax.select(state.dropped, jnp.zeros_like(OBJECT_ARRAY[OBJECT_TO_INDEX['agent']]),
                                      OBJECT_ARRAY[OBJECT_TO_INDEX['agent']])
                obs_map = obs_map.at[
                          pad + state.agents_pos[1, 1], pad + state.agents_pos[1, 0], :].set(
                    demo_obj
                )
                # Observe different goal colours
                obs_map = obs_map.at[
                          pad + state.goals_pos[0, 1], pad + state.goals_pos[0, 0], :].set(
                    GOAL_ARRAY[0]
                )
                obs_map = obs_map.at[
                          pad + state.goals_pos[1, 1], pad + state.goals_pos[1, 0], :].set(
                    GOAL_ARRAY[1]
                )
                obs_map = obs_map.at[
                          pad + state.goals_pos[2, 1], pad + state.goals_pos[2, 0], :].set(
                    GOAL_ARRAY[2]
                )
                obs_map = obs_map.at[
                        pad + state.agents_pos[0, 1], pad + state.agents_pos[0, 0], :].set(
                    OBJECT_ARRAY[OBJECT_TO_INDEX['agent']]
                )
                return obs_map

            def false_fn():
                # Only observe color component and remove other agents
                agents_pos = jnp.array([state.agents_pos.at[aidx].get()])
                agents_dir_idx = jnp.array([state.agents_dir_idx.at[aidx].get()])
                map = make_map(
                    self.params,
                    state.wall_map,
                    state.goals_pos,
                    agents_pos,
                    agents_dir_idx,
                    pad_obs=True)
                # Only observe object component
                obs_map = map[:, :, 0]
                # Convert indices to objects
                obs_map = jax.tree_map(lambda idx: OBJECT_ARRAY[idx], obs_map)
                # Observe different goal colours
                obs_map = obs_map.at[
                          pad + state.goals_pos[:, 1], pad + state.goals_pos[:, 0], :].set(
                    GOAL_ARRAY[0]
                )
                cur_score = state.scores.at[aidx].get()
                cur_score = lax.select(state.scrambled, cur_score + 10, cur_score)
                target_idx = state.sequence.at[cur_score.astype(int) + 1].get()
                obs_map = obs_map.at[
                          pad + state.goals_pos[target_idx, 1], pad + state.goals_pos[target_idx, 0], :].set(
                    GOAL_ARRAY[1]
                )
                obs_map = obs_map.at[
                        pad + state.agents_pos[1, 1], pad + state.agents_pos[1, 0], :].set(
                    OBJECT_ARRAY[OBJECT_TO_INDEX['agent']]
                )
                return obs_map

            is_social_learner = (aidx == 0)
            obs_map = lax.cond(is_social_learner, true_fn, false_fn)

            obs = obs_map.at[
                  coord_y, coord_x, :].get().reshape(obs.shape[0], obs.shape[1], 5)

            obs = (state.agents_dir_idx[aidx] == 0) * jnp.rot90(obs, 1) + \
                  (state.agents_dir_idx[aidx] == 1) * jnp.rot90(obs, 2) + \
                  (state.agents_dir_idx[aidx] == 2) * jnp.rot90(obs, 3) + \
                  (state.agents_dir_idx[aidx] == 3) * jnp.rot90(obs, 4)

            agent_dir = state.agents_dir_idx[aidx]
            agent_dir = jax.nn.one_hot(agent_dir, 4)
            image = obs.astype(jnp.uint8)

            trial = jax.nn.one_hot(state.trial, num_classes=4)
            reward = state.reward.at[aidx].get() + 1
            reward = jax.nn.one_hot(reward, num_classes=3)

            obs_dict = dict(
                image=image,
                agent_dir=agent_dir,
                trial=trial,
                reward=reward
            )

            return OrderedDict(obs_dict)

        obs = [_observation(aidx, state) for aidx in range(self.num_agents)]
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def step_agents(self, key: chex.PRNGKey, state: EnvState, actions: chex.Array,
                    penalty: float, prob_obs: float) -> Tuple[EnvState, chex.Array]:
        params = self.params

        def _step(aidx, carry):
            actions, rewards, state = carry
            action = actions[aidx]
            new_key, subkey = jax.random.split(key)
            fix_a_cond = jnp.logical_and((aidx == 0), state.time == 0)
            action = lax.select(fix_a_cond, 2, action)
            noop_cond = jnp.logical_and((aidx == 1), state.time == 0)
            action = lax.select(noop_cond, -1, action)
            fwd = (action == Actions.forward)
            # Update agent position (forward action)
            fwd_pos = jnp.minimum(
                jnp.maximum(state.agents_pos[aidx] + (action == Actions.forward) * state.agents_dir[aidx], 0),
                jnp.array((params.width - 1, params.height - 1), dtype=jnp.uint32))

            # Can't go past wall or goal
            fwd_pos_has_wall = state.wall_map.at[fwd_pos[1], fwd_pos[0]].get()
            fwd_pos_has_goal = state.goals_map.at[fwd_pos[1], fwd_pos[0]].get()

            cur_score = state.scores.at[aidx].get()
            target_idx = state.sequence.at[cur_score.astype(int) + 1].get()
            target_pos = state.goals_pos[target_idx, :]
            # Check if next goal index was hit
            hit_target = jnp.logical_and((fwd_pos[1] == target_pos[1]), (fwd_pos[0] == target_pos[0]))
            # Check if an incorrect goal was hit
            moved_to_goal = jnp.logical_and(fwd_pos_has_goal, fwd)
            hit_wrong = jnp.logical_and(moved_to_goal, jnp.logical_not(hit_target))

            fwd_pos_blocked = fwd_pos_has_wall

            agent_pos_prev = jnp.array(state.agents_pos[aidx])
            agent_pos = (fwd_pos_blocked * state.agents_pos[aidx] + (~fwd_pos_blocked) * fwd_pos).astype(jnp.uint32)
            agents_pos = state.agents_pos
            agents_pos = agents_pos.at[aidx].set(agent_pos)

            # Update agent direction (left_turn or right_turn action)
            agent_dir_offset = \
                0 \
                + (action == Actions.left) * (-1) \
                + (action == Actions.right) * 1

            agent_dir_idx = (state.agents_dir_idx[aidx] + agent_dir_offset) % 4
            agents_dir_idx = state.agents_dir_idx
            agents_dir_idx = agents_dir_idx.at[aidx].set(agent_dir_idx)
            agent_dir = DIR_TO_VEC[agents_dir_idx[aidx]]
            agents_dir = state.agents_dir
            agents_dir = agents_dir.at[aidx].set(agent_dir)

            empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
            color_list = jnp.array([COLOR_TO_INDEX['yellow'], COLOR_TO_INDEX['green'], COLOR_TO_INDEX['blue']])
            replace_idx = jnp.argwhere(jnp.array(
                    [jnp.logical_and(state.goals_pos[i, 1] == agent_pos_prev[1],
                                     state.goals_pos[i, 0] == agent_pos_prev[0]) for i in range(params.n_goals)]),
                    size=1)[0][0]
            goal = jnp.array([OBJECT_TO_INDEX['goal'], color_list[replace_idx], 0], dtype=jnp.uint8)
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], 0], dtype=jnp.uint8)

            # Make sure map shows goal after agent has crossed it
            prev_pos_has_goal = state.goals_map.at[agent_pos_prev[1], agent_pos_prev[0]].get()
            other_agents = jnp.delete(agents_pos, aidx, axis=0, assume_unique_indices=True)
            intersections = jnp.equal(agent_pos_prev, other_agents)
            prev_pos_has_other_agent = jnp.max(jnp.prod(intersections, axis=1))
            reset_obj = lax.select(prev_pos_has_goal, goal, empty)
            reset_obj = lax.select(prev_pos_has_other_agent, agent, reset_obj)

            # Update map
            padding = self.obs_shape[0] - 1
            map = state.map
            map = map.at[padding + agent_pos_prev[1], padding + agent_pos_prev[0], :].set(reset_obj)
            map = map.at[padding + agent_pos[1], padding + agent_pos[0], :].set(agent)
            agents_map = state.agents_map
            agents_map = agents_map.at[agent_pos_prev[1], agent_pos_prev[0]].set(prev_pos_has_other_agent)
            agents_map = agents_map.at[agent_pos[1], agent_pos[0]].set(1)

            reward = hit_target
            reward -= penalty * hit_wrong
            rewards = rewards.at[aidx].set(reward)

            score = cur_score + reward
            scores = state.scores.at[aidx].set(score)

            hit_target_update = state.hit_target.at[aidx].set(reward)

            new_key, subkey = jax.random.split(new_key)
            draw = jax.random.uniform(subkey)
            drop_demos = (draw > prob_obs)
            dropped = lax.select((aidx == 0), drop_demos, state.dropped)

            new_key, subkey = jax.random.split(new_key)
            draw = jax.random.uniform(subkey)
            scramble_target = jnp.logical_and(hit_target, (draw > 0.8))
            scramble_target = lax.select(hit_target, scramble_target, state.scrambled)
            scrambled = lax.select((aidx == 1), scramble_target, state.scrambled)

            return (actions, rewards, state.replace(
                agents_pos=agents_pos,
                agents_dir_idx=agents_dir_idx,
                agents_dir=agents_dir,
                agents_map=agents_map,
                map=map,
                scores=scores,
                hit_target=hit_target_update,
                dropped=dropped,
                reward=state.reward.at[aidx].set(reward),
                scrambled=scrambled))

        rewards = jnp.zeros(self.num_agents)
        actions, rewards, state = jax.lax.fori_loop(0, self.num_agents, _step, (actions, rewards, state))

        return (state, rewards)

    def is_terminal(self, state: EnvState) -> bool:
        """Check whether state is terminal."""
        terminate = (state.trial > 3)
        return jnp.logical_or(terminate, state.terminal)

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Goal Cycle"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def observation_space(self, agent: str):
        """ Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """ Action space for a given agent."""
        return self.action_spaces[agent]