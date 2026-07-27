import random
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_to_tensor(obs, device):
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def sleep_for_fps(start, fps):
    delay = 1.0 / fps - (time.perf_counter() - start)
    if delay > 0:
        time.sleep(delay)


def action_name(env, action):
    try:
        return env.actions(action).name
    except ValueError:
        return str(action)


def render_live_step(env, fps):
    if fps <= 0:
        return
    start = time.perf_counter()
    env.render(mode="human")
    sleep_for_fps(start, fps)


def render_validation_step(env, fps, env_action, timestep, reward=None, total_reward=None):
    if fps <= 0:
        return
    start = time.perf_counter()
    if getattr(env, "window", None) is None:
        env.render(mode="human")
    caption = f"action={action_name(env, env_action)} ({env_action}) | timestep={timestep}"
    if reward is not None:
        caption += f" | reward={reward:.3f}"
    if total_reward is not None:
        caption += f" | return={total_reward:.3f}"
    env.window.set_caption(caption)
    env.render(mode="human")
    sleep_for_fps(start, fps)
