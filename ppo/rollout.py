import time

import numpy as np
import torch

from env import to_env_action
from models import detach_state, is_recurrent_model, state_to_numpy
from utils import obs_to_tensor, render_live_step


def compute_gae(rewards, dones, values, cfg):
    advantages = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + cfg["gamma"] * values[t + 1] * nonterminal - values[t]
        gae = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values[:-1]


def rollout(env, model, cfg, device, render_fps=0):
    rollout_start = time.perf_counter()
    obs = env.reset()
    recurrent = is_recurrent_model(model)
    recurrent_state = model.initial_state(1, device) if recurrent else None

    obs_buf, action_buf, logprob_buf = [], [], []
    reward_buf, done_buf, value_buf, recurrent_state_buf = [], [], [], []
    episode_returns, episode_env_returns, episode_successes = [], [], []
    episode_lengths = []
    env_episode_counts = {}
    episode_return = episode_env_return = episode_length = 0
    rule_assemblies = boundary_hits = stuck_pushes = rule_dead_events = 0

    for _ in range(cfg["rollout_steps"]):
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            if recurrent:
                recurrent_state_buf.append(state_to_numpy(recurrent_state))
                action, logprob, _, value, next_recurrent_state = model.act(obs_t, recurrent_state)
            else:
                action, logprob, _, value = model.act(obs_t)

        env_action = to_env_action(action.item(), cfg)
        next_obs, reward, done, info = env.step(env_action)
        render_live_step(env, render_fps)

        obs_buf.append(obs)
        action_buf.append(action.item())
        logprob_buf.append(logprob.item())
        reward_buf.append(reward)
        done_buf.append(done)
        value_buf.append(value.item())

        episode_return += reward
        episode_env_return += info.get("env_reward", reward)
        episode_length += 1
        rule_assemblies += int(info.get("rule_assembled", False))
        boundary_hits += int(info.get("hit_boundary", False))
        stuck_pushes += int(info.get("stuck_push", False))
        rule_dead_events += int(info.get("rule_dead", False))
        obs = next_obs

        if recurrent:
            recurrent_state = detach_state(next_recurrent_state)
        if done:
            env_id = info.get("env_id", getattr(env, "active_env_id", cfg.get("env_id", "unknown")))
            episode_returns.append(episode_return)
            episode_env_returns.append(episode_env_return)
            episode_successes.append(episode_env_return > 0)
            episode_lengths.append(episode_length)
            env_episode_counts[env_id] = env_episode_counts.get(env_id, 0) + 1
            episode_return = episode_env_return = episode_length = 0
            obs = env.reset()
            if recurrent:
                recurrent_state = model.initial_state(1, device)

    with torch.no_grad():
        if recurrent:
            next_value = model(obs_to_tensor(obs, device), recurrent_state)[1].item()
        else:
            next_value = model(obs_to_tensor(obs, device))[1].item()

    rewards = np.array(reward_buf, dtype=np.float32)
    dones = np.array(done_buf, dtype=np.float32)
    values = np.array(value_buf + [next_value], dtype=np.float32)
    advantages, returns = compute_gae(rewards, dones, values, cfg)

    return {
        "obs": np.array(obs_buf),
        "actions": np.array(action_buf),
        "logprobs": np.array(logprob_buf, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "recurrent_states": recurrent_state_buf,
        "episode_returns": episode_returns,
        "episode_env_returns": episode_env_returns,
        "episode_successes": episode_successes,
        "episode_lengths": episode_lengths,
        "env_episode_counts": env_episode_counts,
        "rule_assemblies": rule_assemblies,
        "boundary_hits": boundary_hits,
        "stuck_pushes": stuck_pushes,
        "rule_dead_events": rule_dead_events,
        "rollout_time_sec": time.perf_counter() - rollout_start,
    }
