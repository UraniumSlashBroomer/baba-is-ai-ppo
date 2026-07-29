from __future__ import annotations

import json
import time
import argparse
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

import hydra
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import wandb
from flax import serialization
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf

from xminigrid.environment import EnvParams, Environment
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper

from xland_ppo.nn import ActorCriticRNN
from xland_ppo.task import ENV_ID, make_env
from xland_ppo.utils import Transition, calculate_gae, ppo_update_networks

jax.config.update("jax_threefry_partitionable", True)


@dataclass
class TrainConfig:
    mode: str = "train"
    run_dir: str = "xland_ppo/runs/one_rule_r1_9x9"
    total_timesteps: int = 10_000_000
    updates_per_eval: int = 25
    num_envs: int = 2048
    num_steps: int = 64
    update_epochs: int = 1
    num_minibatches: int = 16
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1
    head_hidden_dim: int = 128
    conv_encoder: bool = False
    enable_bf16: bool = False
    eval_episodes: int = 128
    success_threshold: float = 1.0
    seed: int = 42
    video_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    wandb_mode: str = "disabled"

    def finalize(self) -> None:
        num_devices = jax.local_device_count()
        if self.num_envs % num_devices != 0:
            raise ValueError(f"num_envs={self.num_envs} must be divisible by local_device_count={num_devices}")
        if self.num_envs % self.num_minibatches != 0:
            raise ValueError(f"num_envs={self.num_envs} must be divisible by num_minibatches={self.num_minibatches}")
        self.num_envs_per_device = self.num_envs // num_devices
        self.max_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        self.updates_per_eval = min(self.updates_per_eval, max(1, self.max_updates))
        self.max_updates = max(1, (self.max_updates // self.updates_per_eval) * self.updates_per_eval)
        self.effective_total_timesteps = self.max_updates * self.num_steps * self.num_envs


def make_wrapped_env():
    env, env_params = make_env()
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)
    return env, env_params


def make_train_state(env: Environment, env_params: EnvParams, config: TrainConfig):
    def linear_schedule(count):
        total_ppo_steps = max(1, config.max_updates * config.num_minibatches * config.update_epochs)
        frac = 1.0 - count / total_ppo_steps
        return config.lr * jnp.maximum(frac, 0.0)

    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)
    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=False,
        conv_encoder=config.conv_encoder,
        dtype=jnp.bfloat16 if config.enable_bf16 else None,
    )
    shapes = env.observation_shape(env_params)
    init_obs = {
        "obs_img": jnp.zeros((config.num_envs_per_device, 1, *shapes["img"])),
        "obs_dir": jnp.zeros((config.num_envs_per_device, 1, shapes["direction"])),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)
    params = network.init(init_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),
    )
    return rng, init_hstate, TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def make_train_chunk(env: Environment, env_params: EnvParams, config: TrainConfig):
    @partial(jax.pmap, axis_name="devices")
    def train_chunk(rng: jax.Array, train_state: TrainState, init_hstate: jax.Array):
        rng, reset_rng = jax.random.split(rng)
        reset_rng = jax.random.split(reset_rng, config.num_envs_per_device)
        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
        prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(config.num_envs_per_device)

        def _update_step(runner_state, _):
            def _env_step(runner_state, _):
                rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state
                rng, action_rng = jax.random.split(rng)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    {
                        "obs_img": prev_timestep.observation["img"][:, None],
                        "obs_dir": prev_timestep.observation["direction"][:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    prev_hstate,
                )
                action, log_prob = dist.sample_and_log_prob(seed=action_rng)
                action = action.squeeze(1)
                value = value.squeeze(1)
                log_prob = log_prob.squeeze(1)
                timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, prev_timestep, action)
                transition = Transition(
                    done=timestep.last(),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation["img"],
                    dir=prev_timestep.observation["direction"],
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                return (rng, train_state, timestep, action, timestep.reward, hstate), transition

            initial_hstate = runner_state[-1]
            runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps)
            rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                {
                    "obs_img": timestep.observation["img"][:, None],
                    "obs_dir": timestep.observation["direction"][:, None],
                    "prev_action": prev_action[:, None],
                    "prev_reward": prev_reward[:, None],
                },
                hstate,
            )
            advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    init_hstate, transitions, advantages, targets = batch_info
                    return ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        init_hstate=init_hstate.squeeze(1),
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                    )

                rng, train_state, init_hstate, transitions, advantages, targets = update_state
                rng, shuffle_rng = jax.random.split(rng)
                permutation = jax.random.permutation(shuffle_rng, config.num_envs_per_device)
                batch = (init_hstate, transitions, advantages, targets)
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)
                batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]),
                    batch,
                )
                train_state, update_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (rng, train_state, init_hstate, transitions, advantages, targets), update_info

            update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
            rng, train_state = update_state[:2]
            return (rng, train_state, timestep, prev_action, prev_reward, hstate), loss_info

        runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)
        runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.updates_per_eval)
        rng, train_state = runner_state[:2]
        return rng, train_state, loss_info

    return train_chunk


def deterministic_rollout(env: Environment, env_params: EnvParams, train_state: TrainState, config: TrainConfig, seed: int):
    rng = jax.random.key(seed)
    timestep = env.reset(env_params, rng)
    hstate = train_state.apply_fn.__self__.initialize_carry(1) if hasattr(train_state.apply_fn, "__self__") else None
    if hstate is None:
        network = ActorCriticRNN(
            num_actions=env.num_actions(env_params),
            obs_emb_dim=config.obs_emb_dim,
            action_emb_dim=config.action_emb_dim,
            rnn_hidden_dim=config.rnn_hidden_dim,
            rnn_num_layers=config.rnn_num_layers,
            head_hidden_dim=config.head_hidden_dim,
            img_obs=False,
            conv_encoder=config.conv_encoder,
            dtype=jnp.bfloat16 if config.enable_bf16 else None,
        )
        hstate = network.initialize_carry(1)
    prev_action = jnp.asarray(0, dtype=jnp.int32)
    prev_reward = jnp.asarray(0.0)
    total_reward = 0.0
    length = 0
    actions: list[int] = []
    frames = [env.render(env_params, timestep)]
    while not bool(timestep.last()) and length < int(env_params.max_steps):
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
        action = int(jnp.argmax(dist.logits).squeeze())
        timestep = env.step(env_params, timestep, jnp.asarray(action))
        total_reward += float(timestep.reward)
        length += 1
        actions.append(action)
        frames.append(env.render(env_params, timestep))
    return {"success": total_reward > 0.0, "reward": total_reward, "length": length, "actions": actions}, frames


def evaluate(env: Environment, env_params: EnvParams, train_state: TrainState, config: TrainConfig):
    stats = [deterministic_rollout(env, env_params, train_state, config, config.seed + 10_000 + i)[0] for i in range(config.eval_episodes)]
    success_rate = sum(s["success"] for s in stats) / len(stats)
    mean_reward = sum(s["reward"] for s in stats) / len(stats)
    mean_length = sum(s["length"] for s in stats) / len(stats)
    return {"success_rate": success_rate, "mean_reward": mean_reward, "mean_length": mean_length}


def save_checkpoint(run_dir: Path, train_state: TrainState, config: TrainConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(train_state.params))
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))


def generate_videos(run_dir: Path, train_state: TrainState, config: TrainConfig, max_steps: int | None = None) -> list[dict]:
    env, env_params = make_env(max_steps=max_steps)
    env = DirectionObservationWrapper(env)
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_stats = []
    for seed in config.video_seeds:
        stats, frames = deterministic_rollout(env, env_params, train_state, config, seed)
        path = video_dir / f"validation_seed_{seed}.mp4"
        imageio.mimsave(path, frames, fps=6)
        stats = {**stats, "seed": seed, "path": str(path)}
        video_stats.append(stats)
    return video_stats


def local_debug(config: TrainConfig) -> dict:
    config.finalize()
    print("local_debug: init", flush=True)
    run_dir = Path(config.run_dir)
    env, env_params = make_wrapped_env()
    rng, init_hstate, train_state = make_train_state(env, env_params, config)
    print("local_debug: state ready", flush=True)

    rng, reset_rng = jax.random.split(rng)
    reset_rng = jax.random.split(reset_rng, config.num_envs_per_device)
    timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
    prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
    prev_reward = jnp.zeros(config.num_envs_per_device)

    def _env_step(carry, _):
        rng, train_state, prev_timestep, prev_action, prev_reward, hstate = carry
        rng, action_rng = jax.random.split(rng)
        dist, value, hstate = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": prev_timestep.observation["img"][:, None],
                "obs_dir": prev_timestep.observation["direction"][:, None],
                "prev_action": prev_action[:, None],
                "prev_reward": prev_reward[:, None],
            },
            hstate,
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_rng)
        action = action.squeeze(1)
        value = value.squeeze(1)
        log_prob = log_prob.squeeze(1)
        timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, prev_timestep, action)
        transition = Transition(
            done=timestep.last(),
            action=action,
            value=value,
            reward=timestep.reward,
            log_prob=log_prob,
            obs=prev_timestep.observation["img"],
            dir=prev_timestep.observation["direction"],
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        return (rng, train_state, timestep, action, timestep.reward, hstate), transition

    carry = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)
    print("local_debug: collect rollout", flush=True)
    carry, transitions = jax.lax.scan(_env_step, carry, None, config.num_steps)
    rng, train_state, timestep, prev_action, prev_reward, hstate = carry
    _, last_val, _ = train_state.apply_fn(
        train_state.params,
        {
            "obs_img": timestep.observation["img"][:, None],
            "obs_dir": timestep.observation["direction"][:, None],
            "prev_action": prev_action[:, None],
            "prev_reward": prev_reward[:, None],
        },
        hstate,
    )
    advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)
    print("local_debug: gae ready", flush=True)

    batch = (init_hstate[None, :], transitions, advantages, targets)
    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)
    local_hstate, local_transitions, local_advantages, local_targets = jtu.tree_map(lambda x: x[:1], batch)

    def _loss_fn(params):
        dist, value, _ = train_state.apply_fn(
            params,
            {
                "obs_img": local_transitions.obs,
                "obs_dir": local_transitions.dir,
                "prev_action": local_transitions.prev_action,
                "prev_reward": local_transitions.prev_reward,
            },
            local_hstate.squeeze(1),
        )
        adv = (local_advantages - local_advantages.mean()) / (local_advantages.std() + 1e-8)
        log_prob = dist.log_prob(local_transitions.action)
        ratio = jnp.exp(log_prob - local_transitions.log_prob)
        actor_loss = -jnp.minimum(
            adv * ratio,
            adv * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
        ).mean()
        value_loss = 0.5 * jnp.square(value - local_targets).mean()
        entropy = dist.entropy().mean()
        return actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy

    loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    print("local_debug: update ready", flush=True)
    save_checkpoint(run_dir, train_state, config)
    eval_env, eval_env_params = make_env(max_steps=16)
    eval_env = DirectionObservationWrapper(eval_env)
    eval_stats = evaluate(eval_env, eval_env_params, train_state, config)
    print("local_debug: eval ready", flush=True)
    video_stats = generate_videos(run_dir, train_state, config, max_steps=16)
    summary = {
        "local_debug": True,
        "loss": float(loss),
        "eval": eval_stats,
        "videos": video_stats,
        "env_id": ENV_ID,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def train(config: TrainConfig) -> dict:
    config.finalize()
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    wandb.init(project="xminigrid", group="one-rule", name=run_dir.name, config=asdict(config), mode=config.wandb_mode)
    env, env_params = make_wrapped_env()
    eval_env, eval_env_params = make_env()
    eval_env = DirectionObservationWrapper(eval_env)
    rng, init_hstate, train_state = make_train_state(env, env_params, config)
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())
    train_chunk = make_train_chunk(env, env_params, config)

    print(f"devices={jax.local_devices()}")
    print(
        f"max_updates={config.max_updates}, updates_per_eval={config.updates_per_eval}, "
        f"effective_total_timesteps={config.effective_total_timesteps}"
    )
    compiled = train_chunk.lower(rng, train_state, init_hstate).compile()

    total_updates = 0
    total_transitions = 0
    started = time.time()
    final_eval = {"success_rate": 0.0, "mean_reward": 0.0, "mean_length": 0.0}
    while total_updates < config.max_updates and final_eval["success_rate"] < config.success_threshold:
        rng, train_state, loss_info = jax.block_until_ready(compiled(rng, train_state, init_hstate))
        completed_updates = min(config.updates_per_eval, config.max_updates - total_updates)
        total_updates += completed_updates
        total_transitions += completed_updates * config.num_steps * config.num_envs
        host_state = unreplicate(train_state)
        final_eval = evaluate(eval_env, eval_env_params, host_state, config)
        train_metrics = jtu.tree_map(lambda x: float(unreplicate(x)[-1]), loss_info)
        row = {
            "updates": total_updates,
            "transitions": total_transitions,
            **train_metrics,
            **{f"eval/{k}": v for k, v in final_eval.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        wandb.log(row)
        print(json.dumps(row, sort_keys=True))
        save_checkpoint(run_dir, host_state, config)

    host_state = unreplicate(train_state)
    video_stats = generate_videos(run_dir, host_state, config)
    elapsed = time.time() - started
    summary = {
        "trained": final_eval["success_rate"] >= config.success_threshold,
        "elapsed_seconds": elapsed,
        "updates": total_updates,
        "transitions": total_transitions,
        "eval": final_eval,
        "videos": video_stats,
        "env_id": ENV_ID,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    wandb.summary.update(summary)
    wandb.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def config_from_hydra(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    field_names = {field.name for field in fields(TrainConfig)}
    kwargs = {key: value for key, value in data.items() if key in field_names}
    if "video_seeds" in kwargs:
        kwargs["video_seeds"] = tuple(kwargs["video_seeds"])
    return TrainConfig(**kwargs)


@hydra.main(version_base="1.3", config_path="configs", config_name="one_rule")
def main(cfg: DictConfig) -> None:
    config = config_from_hydra(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if config.mode == "local_debug":
        local_debug(config)
    elif config.mode == "train":
        train(config)
    else:
        raise ValueError(f"Unknown mode={config.mode!r}; expected 'train' or 'local_debug'.")


if __name__ == "__main__":
    # Hydra 1.3 uses a lazy help object that Python 3.14 argparse rejects.
    if hasattr(argparse.ArgumentParser, "_check_help"):
        _argparse_check_help = argparse.ArgumentParser._check_help

        def _check_help_compatible(self, action):
            if not isinstance(action.help, str):
                return
            return _argparse_check_help(self, action)

        argparse.ArgumentParser._check_help = _check_help_compatible
    main()
