from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from xminigrid.wrappers import DirectionObservationWrapper

from xland_ppo.nn import ActorCriticRNN
from xland_ppo.task import ENV_ID, make_env
from xland_ppo.train_one_rule import TrainConfig, load_checkpoint_params, make_train_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.checkpoint)
    if args.grid_size is not None:
        config.grid_size = int(args.grid_size)
    config.num_envs = 1
    config.num_minibatches = 1
    config.finalize()

    env, env_params = make_env(max_steps=args.max_steps, grid_size=config.grid_size)
    env = DirectionObservationWrapper(env)
    _, _, train_state = make_train_state(env, env_params, config)
    train_state = load_checkpoint_params(train_state, str(args.checkpoint))

    args.out.mkdir(parents=True, exist_ok=True)
    rng = jax.random.key(args.seed)
    summaries = []
    for episode_idx in range(args.episodes):
        rng, episode_key = jax.random.split(rng)
        summary = collect_episode(
            env=env,
            env_params=env_params,
            train_state=train_state,
            config=config,
            key=episode_key,
            out_dir=args.out / f"episode_{episode_idx:06d}",
            image_size=args.image_size,
            greedy=args.greedy,
        )
        summaries.append(summary)

    metadata = {
        "env_id": ENV_ID,
        "grid_size": config.grid_size,
        "episodes": len(summaries),
        "image_size": args.image_size,
        "greedy": args.greedy,
        "success_rate": sum(item["success"] for item in summaries) / max(1, len(summaries)),
        "episodes_summary": summaries,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in metadata.items() if k != "episodes_summary"}, indent=2, sort_keys=True))


def collect_episode(
    *,
    env,
    env_params,
    train_state: TrainState,
    config: TrainConfig,
    key: jax.Array,
    out_dir: Path,
    image_size: int,
    greedy: bool,
) -> dict:
    reset_key, policy_key = jax.random.split(key)
    timestep = env.reset(env_params, reset_key)
    hstate = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=False,
        conv_encoder=config.conv_encoder,
        dtype=jnp.bfloat16 if config.enable_bf16 else None,
    ).initialize_carry(1)
    prev_action = jnp.asarray(0, dtype=jnp.int32)
    prev_reward = jnp.asarray(0.0)

    frames = [_resize_nearest(env.render(env_params, timestep), image_size)]
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    total_reward = 0.0

    while not bool(timestep.last()) and len(actions) < int(env_params.max_steps):
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": timestep.observation["img"][None, None, ...],
                "obs_dir": timestep.observation["direction"][None, None, ...],
                "prev_action": prev_action[None, None],
                "prev_reward": prev_reward[None, None],
            },
            hstate,
        )
        if greedy:
            action = int(jnp.argmax(dist.logits).squeeze())
        else:
            policy_key, action_key = jax.random.split(policy_key)
            action = int(dist.sample(seed=action_key).squeeze())

        timestep = env.step(env_params, timestep, jnp.asarray(action))
        reward = float(timestep.reward)
        done = bool(timestep.last())
        total_reward += reward

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        frames.append(_resize_nearest(env.render(env_params, timestep), image_size))

        prev_action = jnp.asarray(action, dtype=jnp.int32)
        prev_reward = jnp.asarray(reward)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "frames.npy", np.asarray(frames, dtype=np.uint8))
    np.save(out_dir / "actions.npy", np.asarray(actions, dtype=np.int64))
    np.save(out_dir / "rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(out_dir / "dones.npy", np.asarray(dones, dtype=np.bool_))
    summary = {
        "episode": out_dir.name,
        "frames": len(frames),
        "actions": len(actions),
        "reward": total_reward,
        "success": total_reward > 0.0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _load_config(checkpoint: Path) -> TrainConfig:
    run_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return TrainConfig(checkpoint_path=str(checkpoint))

    data = json.loads(config_path.read_text())
    field_names = {field.name for field in fields(TrainConfig)}
    kwargs = {key: value for key, value in data.items() if key in field_names}
    if "video_seeds" in kwargs:
        kwargs["video_seeds"] = tuple(kwargs["video_seeds"])
    kwargs["checkpoint_path"] = str(checkpoint)
    return TrainConfig(**kwargs)


def _resize_nearest(frame: np.ndarray, image_size: int) -> np.ndarray:
    if frame.shape[0] == image_size and frame.shape[1] == image_size:
        return frame
    y_idx = (np.arange(image_size) * (frame.shape[0] / image_size)).astype(np.int32)
    x_idx = (np.arange(image_size) * (frame.shape[1] / image_size)).astype(np.int32)
    return frame[y_idx[:, None], x_idx[None, :]]


if __name__ == "__main__":
    main()
