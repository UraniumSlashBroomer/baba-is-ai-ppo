from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp

from xland_ppo.task import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("xland_ppo/videos/smoke.mp4"))
    args = parser.parse_args()

    env, env_params = make_env()
    key = jax.random.key(args.seed)
    timestep = env.reset(env_params, key)

    frames = [env.render(env_params, timestep)]
    for action in [1, 0, 2, 0, 1, 0, 0, 0][: args.steps]:
        timestep = env.step(env_params, timestep, jnp.asarray(action))
        frames.append(env.render(env_params, timestep))
        if bool(timestep.last()):
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=3)
    print(f"env={type(env).__name__}")
    print(f"obs_shape={env.observation_shape(env_params)}")
    print(f"num_actions={env.num_actions(env_params)}")
    print(f"ruleset={env_params.ruleset}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
