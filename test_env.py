from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
BABA_REPO = PROJECT_ROOT / "baba-is-ai"
sys.path.insert(0, str(BABA_REPO))

import baba  # noqa: E402


ENV_ID = "env/make_win"


def main():
    # The environments are procedural. This seed makes the sampled start state
    # reproducible for this smoke test.
    np.random.seed(0)

    env = baba.make(ENV_ID)

    state = env.reset()
    encoded_state = env.gen_obs()
    grid_state = env.grid.encode()

    print(f"env_id: {ENV_ID}")
    print(f"env class: {env.__class__.__name__}")
    print(f"action_space: {env.action_space}")
    print(f"observation_space: {env.observation_space}")
    print(f"state shape: {state.shape}, dtype: {state.dtype}")
    print(f"state equals env.gen_obs(): {np.array_equal(state, encoded_state)}")
    print(f"state equals env.grid.encode(): {np.array_equal(state, grid_state)}")
    print(f"agent_pos: {env.agent_pos}")
    print(f"agent_dir: {env.agent_dir}")
    print(f"target_plan: {getattr(env, 'target_plan', None)}")
    print(f"win_rule: {getattr(env, 'win_rule', None)}")
    print(f"win_obj: {getattr(env, 'win_obj', None)}")
    print(f"ruleset: {env.get_ruleset()}")

    print("\nMatrix view:")
    print(env.render(mode="matrix"))

    print("\nDict view:")
    print(env.render(mode="dict"))

    next_state, reward, done, info = env.step(env.actions.idle)
    print("\nAfter idle step:")
    print(f"next_state shape: {next_state.shape}, dtype: {next_state.dtype}")
    print(f"reward: {reward}, done: {done}, info: {info}")


if __name__ == "__main__":
    main()
