import numpy as np
import jax.numpy as jnp

OBJECT_TO_INDEX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

COLORS = {
    'black': np.array([0, 0, 0]),
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'worst': np.array([74, 65, 42])
}

COLOR_ARRAY = jnp.array([
    jnp.array([0, 0, 0]),
    jnp.array([255, 0, 0]),
    jnp.array([0, 255, 0]),
    jnp.array([0, 0, 255]),
    jnp.array([112, 39, 195]),
    jnp.array([255, 255, 0]),
    jnp.array([100, 100, 100]),
    jnp.array([74, 65, 42])
])

OBJECT_ARRAY = jnp.array([
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([1, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 1, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 1])
])

OBJECT_ARRAY_HETEROGENEOUS_AGENTS = jnp.array([
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([1, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 1, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 1, 0, 0])
])

OBJECT_ARRAY_MORE_GOALS = jnp.array([
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([1, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 1, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 0, 1])
])

GOAL_ARRAY = jnp.array([
    jnp.array([0, 1, 0, 0, 0]),
    jnp.array([0, 0, 1, 0, 0]),
    jnp.array([0, 0, 0, 1, 0])
])


AGENT_ARRAY = jnp.array([
    jnp.array([0, 0, 0, 0, 1, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 1, 0]),
    jnp.array([0, 0, 0, 0, 0, 0, 1])
])


COLOR_TO_INDEX = {
    'black': 0,
    'red': 1,
    'green': 2,
    'blue': 3,
    'purple': 4,
    'yellow': 5,
    'grey': 6,
    'worst': 7
}

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    # Pointing right (positive X)
    (1, 0),  # right
    (0, 1),  # down
    (-1, 0),  # left
    (0, -1),  # up
], dtype=jnp.int8)


def make_map(
        params,
        wall_map,
        goals_pos,
        agents_pos,
        agents_dir_idx,
        pad_obs=False):
    empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
    wall = jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['worst'], 0], dtype=jnp.uint8)
    map = jnp.array(jnp.expand_dims(wall_map, -1), dtype=jnp.uint8)
    map = jnp.where(map > 0, wall, empty)

    if len(agents_dir_idx) == 1:
        agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], agents_dir_idx[0]],
                          dtype=jnp.uint8)
        agent_x, agent_y = agents_pos[0]
        map = map.at[agent_y, agent_x, :].set(agent)
    else:
        agents = jnp.stack(
            [jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], 0]) for i in
             range(params.num_agents)], dtype=jnp.uint8).at[:agents_pos.shape[0]].get()

        agents_x = agents_pos.at[:, 0].get()
        agents_y = agents_pos.at[:, 1].get()
        map = map.at[agents_y, agents_x, :].set(agents)

    if goals_pos != None:
        if len(goals_pos.shape) == 1:
            goal = jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['yellow'], 0], dtype=jnp.uint8)
            goal_x, goal_y = goals_pos
            map = map.at[goal_y, goal_x, :].set(goal)
        else:
            color_list = [COLOR_TO_INDEX['yellow'], COLOR_TO_INDEX['green'], COLOR_TO_INDEX['blue'], 
                          COLOR_TO_INDEX['purple'], COLOR_TO_INDEX['yellow'], COLOR_TO_INDEX['green']]
            goals = jnp.stack(
                [jnp.array([OBJECT_TO_INDEX['goal'], color_list[i], 0]) for i in
                 range(params.n_goals)], dtype=jnp.uint8).at[:goals_pos.shape[0]].get()
            goals_x = goals_pos.at[:, 0].get()
            goals_y = goals_pos.at[:, 1].get()
            map = map.at[goals_y, goals_x, :].set(goals)

    # Add observation padding
    if pad_obs:
        padding = params.agent_view_size - 1
    else:
        padding = 1

    map_padded = jnp.tile(wall.reshape((1, 1, *empty.shape)),
                               (map.shape[0] + 2 * padding, map.shape[1] + 2 * padding, 1))
    map_padded = map_padded.at[padding:-padding, padding:-padding, :].set(map)

    # Add surrounding walls
    wall_start = padding - 1  # start index for walls
    wall_end_y = map_padded.shape[0] - wall_start - 1
    wall_end_x = map_padded.shape[1] - wall_start - 1
    map_padded = map_padded.at[wall_start, wall_start:wall_end_x + 1, :].set(wall)  # top
    map_padded = map_padded.at[wall_end_y, wall_start:wall_end_x + 1, :].set(wall)  # bottom
    map_padded = map_padded.at[wall_start:wall_end_y + 1, wall_start, :].set(wall)  # left
    map_padded = map_padded.at[wall_start:wall_end_y + 1, wall_end_x, :].set(wall)  # right

    return map_padded