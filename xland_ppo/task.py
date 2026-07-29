from __future__ import annotations

import jax.numpy as jnp

import xminigrid
from xminigrid.core.constants import Colors, TILES_REGISTRY, Tiles
from xminigrid.core.goals import AgentOnTileGoal
from xminigrid.core.rules import AgentNearRule
from xminigrid.envs.xland import XLandEnvParams
from xminigrid.types import RuleSet


ENV_ID = "XLand-MiniGrid-R1-9x9"
OBJECT_A = TILES_REGISTRY[Tiles.SQUARE, Colors.RED]
OBJECT_B = TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]


def make_one_rule_ruleset() -> RuleSet:
    """A simple fixed XLand task: near red square -> green goal; stand on green goal."""
    return RuleSet(
        goal=AgentOnTileGoal(tile=OBJECT_B).encode(),
        rules=jnp.asarray([AgentNearRule(tile=OBJECT_A, prod_tile=OBJECT_B).encode()], dtype=jnp.uint8),
        init_tiles=jnp.asarray([OBJECT_A], dtype=jnp.uint8),
    )


def make_env(max_steps: int | None = None):
    env, env_params = xminigrid.make(ENV_ID)
    env_params = env_params.replace(ruleset=make_one_rule_ruleset())
    if max_steps is not None:
        env_params = env_params.replace(max_steps=max_steps)
    assert isinstance(env_params, XLandEnvParams)
    return env, env_params
