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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("xland_ppo/videos/smoke_two_rule_8x8.mp4"))
    args = parser.parse_args()

    env, env_params = make_env()
    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    timestep = env.reset(env_params, reset_key)

    frames = [env.render(env_params, timestep)]
    for _ in range(args.steps):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=env.num_actions(env_params))
        timestep = env.step(env_params, timestep, jnp.asarray(action))
        frames.append(env.render(env_params, timestep))
        if bool(timestep.last()):
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=3)
    print(f"env={type(env).__name__}")
    print(f"obs_shape={env.observation_shape(env_params)}")
    print(f"num_actions={env.num_actions(env_params)}")
    print(f"height={env_params.height} width={env_params.width} max_steps={env_params.max_steps}")
    print(f"ruleset={env_params.ruleset}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
