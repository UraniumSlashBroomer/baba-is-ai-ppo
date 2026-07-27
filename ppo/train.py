import json
import time

import numpy as np
import torch
from torch import nn

from checkpoints import checkpoint_architecture, load_training_checkpoint, save_checkpoint
from env import make_train_env, policy_action_count
from models import build_model, is_recurrent_model, numpy_to_state
from paths import ROOT
from rollout import rollout
from utils import set_seed


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _save_periodic(model, optimizer, cfg, run_dir, update, n_updates, steps, stats, next_save_step):
    if update % cfg["checkpoint_every"] == 0 or update == n_updates:
        save_checkpoint(model, optimizer, cfg, run_dir / f"ppo_{update:04d}.pt", stats)
        save_checkpoint(model, optimizer, cfg, run_dir / "latest.pt", stats)

    save_every_n_steps = int(cfg.get("save_every_n_steps", 0))
    if next_save_step is None or steps < next_save_step:
        return next_save_step

    save_checkpoint(model, optimizer, cfg, run_dir / f"ppo_steps_{steps:08d}.pt", stats)
    save_checkpoint(model, optimizer, cfg, run_dir / "latest_step.pt", stats)
    while next_save_step <= steps:
        next_save_step += save_every_n_steps
    return next_save_step


def _latest_jsonl_stats(path):
    if not path.exists():
        return {}
    last = None
    with path.open() as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def _resume_offsets(ckpt, checkpoint):
    stats = ckpt.get("stats") or {}
    if not stats and checkpoint is not None:
        checkpoint = ROOT / checkpoint
        summary = checkpoint.with_name("latest_summary.json")
        if summary.exists():
            stats = json.loads(summary.read_text()).get("latest_train_log_stats") or {}
    if not stats and checkpoint is not None:
        run_name = checkpoint.parent.name
        stats = _latest_jsonl_stats(ROOT / "ppo" / "logs" / run_name / "train.jsonl")
    return int(stats.get("update", 0)), int(stats.get("steps", 0))


def train(cfg, render_fps=0, checkpoint=None):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if checkpoint is None:
        checkpoint = cfg.get("resume_from_checkpoint")
    if checkpoint:
        arch = checkpoint_architecture(checkpoint, device)
        cfg.update(arch)
        print(f"checkpoint_architecture={arch}")
    warmup_episodes = int(
        max(1, cfg["total_steps"] // cfg["max_episode_steps"])
        * cfg.get("warmup_episode_ratio", 0)
    )
    env = make_train_env(cfg, warmup_episodes)
    model = build_model(env.observation_space.shape, policy_action_count(env, cfg), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    update_offset = step_offset = 0
    if checkpoint:
        ckpt = load_training_checkpoint(model, optimizer, checkpoint, device)
        update_offset, step_offset = _resume_offsets(ckpt, checkpoint)
        print(f"resumed checkpoint={checkpoint} update={update_offset} steps={step_offset}")
    recurrent = is_recurrent_model(model)

    run_dir = ROOT / "ppo" / "checkpoints" / cfg["name"]
    log_dir = ROOT / "ppo" / "logs" / cfg["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    n_updates = cfg["total_steps"] // cfg["rollout_steps"]
    next_save_step = (
        int(cfg["save_every_n_steps"])
        if cfg.get("save_every", False) and int(cfg.get("save_every_n_steps", 0)) > 0
        else None
    )

    for update in range(1, n_updates + 1):
        iteration_start = time.perf_counter()
        batch = rollout(env, model, cfg, device, render_fps)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device) / 255.0
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
        old_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        recurrent_states = numpy_to_state(batch["recurrent_states"], device) if recurrent else None

        indices = np.arange(cfg["rollout_steps"])
        policy_losses, value_losses, entropy_losses = [], [], []
        update_start = time.perf_counter()
        for _ in range(cfg["update_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg["minibatch_size"]):
                mb = indices[start:start + cfg["minibatch_size"]]
                if recurrent:
                    new_logprobs, entropy, values = model.evaluate(obs[mb], actions[mb], tuple(x[mb] for x in recurrent_states))
                else:
                    new_logprobs, entropy, values = model.evaluate(obs[mb], actions[mb])
                ratio = (new_logprobs - old_logprobs[mb]).exp()

                policy_loss = torch.max(
                    -advantages[mb] * ratio,
                    -advantages[mb] * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"]),
                ).mean()
                value_loss = ((returns[mb] - values) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        update_time = time.perf_counter() - update_start
        iteration_time = time.perf_counter() - iteration_start
        global_update = update_offset + update
        steps = step_offset + update * cfg["rollout_steps"]
        stats = {
            "update": global_update,
            "steps": steps,
            "avg_shaped_return": _mean(batch["episode_returns"]),
            "avg_env_return": _mean(batch["episode_env_returns"]),
            "success_rate": _mean(batch["episode_successes"]),
            "avg_episode_length": _mean(batch["episode_lengths"]),
            "rule_assemblies": int(batch["rule_assemblies"]),
            "boundary_hits": int(batch["boundary_hits"]),
            "stuck_pushes": int(batch["stuck_pushes"]),
            "rule_dead_events": int(batch["rule_dead_events"]),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "rollout_time_sec": float(batch["rollout_time_sec"]),
            "update_time_sec": float(update_time),
            "iteration_time_sec": float(iteration_time),
            "seconds_per_step": float(iteration_time / cfg["rollout_steps"]),
            "steps_per_second": float(cfg["rollout_steps"] / iteration_time) if iteration_time > 0 else 0.0,
            "env_episode_counts": batch["env_episode_counts"],
        }

        with log_path.open("a") as f:
            f.write(json.dumps(stats) + "\n")

        print(
            f"update={global_update}/{update_offset + n_updates} steps={steps} "
            f"shaped_return={stats['avg_shaped_return']:.3f} env_return={stats['avg_env_return']:.3f} "
            f"success_rate={stats['success_rate']:.3f} avg_len={stats['avg_episode_length']:.1f} "
            f"rules={stats['rule_assemblies']} boundary_hits={stats['boundary_hits']} "
            f"stuck_pushes={stats['stuck_pushes']} dead_rules={stats['rule_dead_events']} "
            f"iter_time={iteration_time:.2f}s rollout_time={batch['rollout_time_sec']:.2f}s "
            f"update_time={update_time:.2f}s steps_per_sec={stats['steps_per_second']:.1f}"
        )

        next_save_step = _save_periodic(
            model, optimizer, cfg, run_dir, global_update, update_offset + n_updates, steps, stats, next_save_step
        )
