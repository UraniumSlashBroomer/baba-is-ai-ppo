from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPO_DIR = PROJECT_ROOT / "ppo"
NANOWM_SRC = PROJECT_ROOT / "nano-world-model" / "src"
for path in (PPO_DIR, NANOWM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def run_env_success_validation(
    cfg,
    *,
    model,
    action_head,
    latent_codec,
    diffusion,
    dfot_sample,
    device: torch.device,
) -> Dict[str, float]:
    """Roll out the learned action head in the real env and report success rate.

    At every step we:
      1. render the real env frame,
      2. predict action from the current latent,
      3. ask NanoWM to generate a next latent conditioned on that action,
      4. discard that generated next latent,
      5. execute the action in the real env and continue from the real frame.
    """
    from env import make_env

    val_cfg = cfg.validation
    env_cfg = {
        "env_id": str(val_cfg.env_id),
        "max_episode_steps": int(val_cfg.max_episode_steps),
        "use_coord_channels": False,
        "use_idle": bool(val_cfg.use_idle),
        "use_shaped_reward": False,
        "use_stuck_push_penalty": False,
    }
    env = make_env(env_cfg)

    model_was_training = model.training
    head_was_training = action_head.training
    model.eval()
    action_head.eval()

    successes = 0
    returns = []
    lengths = []

    with torch.no_grad():
        for episode_idx in range(int(val_cfg.episodes)):
            env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < int(val_cfg.max_episode_steps):
                frame = env.render(mode="rgb_array")
                current_latent = encode_rgb_frame(frame, latent_codec, int(cfg.model.image_size), device)
                logits = action_head(current_latent[:, 0])
                action_idx = select_action(logits, greedy=bool(val_cfg.greedy))
                env_action = action_index_to_env_action(action_idx, cfg)

                if bool(val_cfg.sample_next_image):
                    action_tensor = action_index_to_model_action(action_idx, cfg, device)
                    _ = predict_and_discard_next_latent(
                        cfg,
                        model=model,
                        diffusion=diffusion,
                        dfot_sample=dfot_sample,
                        current_latent=current_latent,
                        action_tensor=action_tensor,
                        device=device,
                    )

                _, reward, done, _ = env.step(env_action)
                total_reward += float(reward)
                steps += 1

            successes += total_reward > 0
            returns.append(total_reward)
            lengths.append(steps)

    if model_was_training:
        model.train()
    if head_was_training:
        action_head.train()

    episodes = max(1, int(val_cfg.episodes))
    return {
        "val/success_rate": successes / episodes,
        "val/avg_return": float(np.mean(returns)) if returns else 0.0,
        "val/avg_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def encode_rgb_frame(frame: np.ndarray, latent_codec, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(frame, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    tensor = tensor * 2.0 - 1.0
    latent = latent_codec.encode(tensor)
    return latent.unsqueeze(1)


def select_action(logits: torch.Tensor, *, greedy: bool) -> int:
    if greedy:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def action_index_to_env_action(action_idx: int, cfg) -> int:
    return int(action_idx - int(cfg.data.action_offset))


def action_index_to_model_action(action_idx: int, cfg, device: torch.device) -> torch.Tensor:
    action = torch.zeros((1, int(cfg.model.num_frames), int(cfg.data.action_dim)), dtype=torch.float32, device=device)
    action[:, 0, action_idx] = 1.0
    return action


def predict_and_discard_next_latent(
    cfg,
    *,
    model,
    diffusion,
    dfot_sample,
    current_latent: torch.Tensor,
    action_tensor: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    shape = (
        1,
        int(cfg.model.num_frames),
        int(cfg.model.latent_channels),
        int(cfg.model.latent_size),
        int(cfg.model.latent_size),
    )
    return dfot_sample(
        diffusion=diffusion,
        model=model,
        shape=shape,
        context=current_latent,
        n_context_frames=1,
        scheduling_mode=str(cfg.validation.scheduling_mode),
        num_sampling_steps=int(cfg.validation.num_sampling_steps),
        model_kwargs={"y": None, "action": action_tensor},
        device=device,
        progress=False,
        n_generate_frames=1,
        history_stabilization_level=0.0,
    )
