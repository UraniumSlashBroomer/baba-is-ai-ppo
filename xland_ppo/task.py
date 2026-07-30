from __future__ import annotations

from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np

import xminigrid
from xminigrid.core.constants import Colors, TILES_REGISTRY, Tiles
from xminigrid.core.goals import AgentOnTileGoal
from xminigrid.core.rules import TileNearRule
from xminigrid.environment import EnvParamsT
from xminigrid.envs.xland import XLandEnvParams
from xminigrid.rendering.rgb_render import render_tile
from xminigrid.types import EnvCarryT, IntOrArray, RuleSet, TimeStep
from xminigrid.wrappers import Wrapper


ENV_ID = "XLand-MiniGrid-R1-9x9"
GRID_SIZE = 8

RED_BALL = TILES_REGISTRY[Tiles.BALL, Colors.RED]
ORANGE_SQUARE = TILES_REGISTRY[Tiles.SQUARE, Colors.ORANGE]
PURPLE_PYRAMID = TILES_REGISTRY[Tiles.PYRAMID, Colors.PURPLE]
BROWN_HEX = TILES_REGISTRY[Tiles.HEX, Colors.BROWN]
PINK_GOAL = TILES_REGISTRY[Tiles.GOAL, Colors.PINK]
EMPTY_TILE = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]


class FullObservationWrapper(Wrapper):
    """Expose the complete grid as observation instead of the agent field of view."""

    def observation_shape(self, params: EnvParamsT) -> Union[tuple[int, int, int], dict[str, Any]]:
        base_shape = self._env.observation_shape(params)
        channels = base_shape[-1] if not isinstance(base_shape, dict) else base_shape["img"][-1]
        return params.height, params.width, channels + 3

    def _full_obs(self, timestep: TimeStep[EnvCarryT]) -> TimeStep[EnvCarryT]:
        agent_mask = jnp.zeros((*timestep.state.grid.shape[:2], 1), dtype=timestep.state.grid.dtype)
        pocket_grid = jnp.zeros((*timestep.state.grid.shape[:2], 2), dtype=timestep.state.grid.dtype)
        agent_y, agent_x = timestep.state.agent.position
        agent_mask = agent_mask.at[agent_y, agent_x, 0].set(1)
        pocket_grid = pocket_grid.at[agent_y, agent_x].set(timestep.state.agent.pocket)
        full_observation = jnp.concatenate([timestep.state.grid, agent_mask, pocket_grid], axis=-1)
        return timestep.replace(observation=full_observation)

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        return self._full_obs(self._env.reset(params, key))

    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        return self._full_obs(self._env.step(params, timestep, action))

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]) -> np.ndarray:
        if params.render_mode != "rgb_array":
            return self._env.render(params, timestep)

        grid = np.asarray(timestep.state.grid)
        tile_size = 32
        img = np.full((grid.shape[0] * tile_size, grid.shape[1] * tile_size, 3), dtype=np.uint8, fill_value=255)
        agent_position = np.asarray(timestep.state.agent.position)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                agent_direction = int(timestep.state.agent.direction) if np.array_equal((y, x), agent_position) else None
                tile_img = np.asarray(
                    render_tile(tuple(grid[y, x].tolist()), agent_direction=agent_direction, highlight=False),
                    dtype=np.uint8,
                )
                if agent_direction is not None:
                    tile_img = _render_pocket_on_agent(tile_img, timestep.state.agent.pocket, agent_direction)
                img[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size, :] = tile_img

        return img


def _render_pocket_on_agent(tile_img: np.ndarray, pocket: jax.Array, agent_direction: int) -> np.ndarray:
    pocket = np.asarray(pocket)
    if np.array_equal(pocket, np.asarray(EMPTY_TILE)):
        return tile_img

    src = np.asarray(render_tile(tuple(pocket.tolist()), agent_direction=None, highlight=False), dtype=np.uint8)
    size = 11
    idx = (np.arange(size) * (src.shape[0] / size)).astype(np.int32)
    small = src[idx[:, None], idx[None, :]]
    color_mask = np.max(small, axis=-1) - np.min(small, axis=-1) > 35

    centers = {
        0: (8, 16),
        1: (16, 24),
        2: (24, 16),
        3: (16, 8),
    }
    cy, cx = centers[int(agent_direction)]
    y0 = max(0, cy - size // 2)
    x0 = max(0, cx - size // 2)
    y1 = min(tile_img.shape[0], y0 + size)
    x1 = min(tile_img.shape[1], x0 + size)

    out = tile_img.copy()
    patch = out[y0:y1, x0:x1]
    mask = color_mask[: y1 - y0, : x1 - x0]
    outline = _dilate_mask(mask) & ~mask
    patch[outline] = 0
    patch[mask] = small[: y1 - y0, : x1 - x0][mask]
    out[y0:y1, x0:x1] = patch
    return out


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def make_two_rule_ruleset() -> RuleSet:
    """Two-step task: combine objects into a final pink goal and stand on it."""
    return RuleSet(
        goal=AgentOnTileGoal(tile=PINK_GOAL).encode(),
        rules=jnp.asarray(
            [
                TileNearRule(tile_a=RED_BALL, tile_b=ORANGE_SQUARE, prod_tile=PURPLE_PYRAMID).encode(),
                TileNearRule(tile_a=PURPLE_PYRAMID, tile_b=BROWN_HEX, prod_tile=PINK_GOAL).encode(),
            ],
            dtype=jnp.uint8,
        ),
        init_tiles=jnp.asarray([RED_BALL, ORANGE_SQUARE, BROWN_HEX], dtype=jnp.uint8),
    )


def make_env(max_steps: int | None = None, grid_size: int = GRID_SIZE):
    env, env_params = xminigrid.make(ENV_ID, height=grid_size, width=grid_size)
    env_params = env_params.replace(ruleset=make_two_rule_ruleset())
    if max_steps is not None:
        env_params = env_params.replace(max_steps=max_steps)
    elif env_params.max_steps != 3 * (env_params.height * env_params.width):
        env_params = env_params.replace(max_steps=3 * (env_params.height * env_params.width))
    assert isinstance(env_params, XLandEnvParams)
    return FullObservationWrapper(env), env_params
