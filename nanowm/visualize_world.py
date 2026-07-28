"""Visualize NanoWM one-step world predictions in Baba.

Example:
    python -m nanowm.visualize_world \
        --checkpoint nanowm/checkpoints/step_0010000.pt \
        --num-seeds 4 \
        --seed-start 100000 \
        --output-dir nanowm/visualizations/step_0010000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .policy import LatentActionHead
from .train import import_nanowm_components, resolve_device, sanity_check_latent_codec
from .validation import (
    action_index_to_env_action,
    action_index_to_model_action,
    encode_rgb_frame,
    predict_and_discard_next_latent,
    select_action,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPO_DIR = PROJECT_ROOT / "ppo"
NANOWM_SRC = PROJECT_ROOT / "nano-world-model" / "src"
for path in (PPO_DIR, NANOWM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_checkpoint_config(ckpt, args)
    cfg.device = str(device)
    cfg.validation.episodes = 1
    cfg.validation.max_episode_steps = int(args.max_episode_steps or cfg.validation.max_episode_steps)
    cfg.validation.num_sampling_steps = int(args.num_sampling_steps or cfg.validation.num_sampling_steps)
    cfg.validation.greedy = bool(args.greedy)
    cfg.validation.sample_next_image = True

    _, _, build_latent_codec, resolve_latent_codec_config, get_models, dfot_sample = import_nanowm_components()

    model = get_models(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    action_head = LatentActionHead(
        latent_channels=int(cfg.model.latent_channels),
        latent_size=int(cfg.model.latent_size),
        action_dim=int(cfg.data.action_dim),
        hidden_size=int(cfg.action_head.hidden_size),
    ).to(device)
    action_head.load_state_dict(ckpt["action_head"])
    action_head.eval()

    latent_codec_cfg = resolve_latent_codec_config(cfg)
    latent_codec = build_latent_codec(cfg).to(device).eval().requires_grad_(False)
    sanity_check_latent_codec(latent_codec, latent_codec_cfg, cfg, device)

    diffusion = build_diffusion(cfg)
    seeds = resolve_seeds(args)
    summaries = []
    for seed in seeds:
        summaries.append(
            visualize_seed(
                seed=seed,
                cfg=cfg,
                model=model,
                action_head=action_head,
                latent_codec=latent_codec,
                diffusion=diffusion,
                dfot_sample=dfot_sample,
                device=device,
                output_dir=output_dir,
                fps=float(args.fps),
            )
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {len(summaries)} videos to {output_dir}")
    print(f"wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to NanoWM checkpoint .pt")
    parser.add_argument("--output-dir", default="nanowm/visualizations/world_rollouts")
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", default=None, help="Comma-separated explicit seeds. Overrides --num-seeds/--seed-start.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_checkpoint_config(ckpt: Dict[str, Any], args: argparse.Namespace):
    if "config" not in ckpt:
        raise KeyError("Checkpoint does not contain resolved config under key 'config'.")
    cfg = OmegaConf.create(ckpt["config"])
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def build_diffusion(cfg):
    from diffusion import create_diffusion

    return create_diffusion(
        timestep_respacing="",
        noise_schedule=str(cfg.diffusion.noise_schedule),
        pred_name=str(cfg.diffusion.pred_name),
        diffusion_steps=int(cfg.diffusion.diffusion_steps),
        snr_gamma=float(cfg.diffusion.snr_gamma),
        zero_terminal_snr=bool(cfg.diffusion.zero_terminal_snr),
    )


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))


@torch.no_grad()
def visualize_seed(
    *,
    seed: int,
    cfg,
    model,
    action_head,
    latent_codec,
    diffusion,
    dfot_sample,
    device: torch.device,
    output_dir: Path,
    fps: float,
) -> Dict[str, Any]:
    from env import make_env

    seed_everything(seed)
    env_cfg = {
        "env_id": str(cfg.validation.env_id),
        "max_episode_steps": int(cfg.validation.max_episode_steps),
        "use_coord_channels": False,
        "use_idle": bool(cfg.validation.use_idle),
        "use_shaped_reward": False,
        "use_stuck_push_penalty": False,
    }
    env = make_env(env_cfg)
    seed_env(env, seed)
    env.reset()

    video_path = output_dir / f"seed_{seed}.mp4"
    writer = None
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < int(cfg.validation.max_episode_steps):
        current_frame = env.render(mode="rgb_array")
        current_latent = encode_rgb_frame(current_frame, latent_codec, int(cfg.model.image_size), device)
        logits = action_head(current_latent[:, 0])
        action_idx = select_action(logits, greedy=bool(cfg.validation.greedy))
        env_action = action_index_to_env_action(action_idx, cfg)
        action_tensor = action_index_to_model_action(action_idx, cfg, device)

        generated_latents = predict_and_discard_next_latent(
            cfg,
            model=model,
            diffusion=diffusion,
            dfot_sample=dfot_sample,
            current_latent=current_latent,
            action_tensor=action_tensor,
            device=device,
        )
        generated_frame = decode_generated_next_frame(generated_latents, latent_codec)

        _, reward, done, _ = env.step(env_action)
        total_reward += float(reward)
        real_next_frame = env.render(mode="rgb_array")

        panel = make_panel(
            real_frame=real_next_frame,
            generated_frame=generated_frame,
            caption=(
                f"seed={seed} step={step + 1} action_idx={action_idx} "
                f"env_action={env_action} reward={float(reward):.3f} return={total_reward:.3f}"
            ),
        )
        if writer is None:
            height, width = panel.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        step += 1

    if writer is not None:
        writer.release()

    success = total_reward > 0.0
    print(f"seed={seed} success={success} return={total_reward:.3f} steps={step} video={video_path}")
    return {
        "seed": seed,
        "success": bool(success),
        "return": total_reward,
        "steps": step,
        "video": str(video_path),
    }


def decode_generated_next_frame(generated_latents: torch.Tensor, latent_codec) -> np.ndarray:
    next_latent = generated_latents[:, 1]
    frame = latent_codec.decode(next_latent)
    return tensor_frame_to_rgb(frame[0])


def tensor_frame_to_rgb(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().float().cpu().clamp(-1.0, 1.0)
    frame = ((frame + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return frame.permute(1, 2, 0).numpy()


def make_panel(real_frame: np.ndarray, generated_frame: np.ndarray, caption: str) -> np.ndarray:
    real = np.asarray(real_frame, dtype=np.uint8)
    generated = np.asarray(generated_frame, dtype=np.uint8)
    if generated.shape[:2] != real.shape[:2]:
        generated_t = torch.as_tensor(generated, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        generated_t = F.interpolate(generated_t, size=real.shape[:2], mode="nearest")
        generated = generated_t[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()

    top_margin = 28
    caption_h = 34
    separator_w = 4
    height, width = real.shape[:2]
    panel = np.full((top_margin + height + caption_h, width * 2 + separator_w, 3), 24, dtype=np.uint8)
    panel[top_margin : top_margin + height, :width] = real
    panel[top_margin : top_margin + height, width + separator_w :] = generated
    put_text(panel, "real next frame", (8, 20), scale=0.55)
    put_text(panel, "generated next frame", (width + separator_w + 8, 20), scale=0.55)
    put_text(panel, caption, (8, top_margin + height + 23), scale=0.55)
    return panel


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], *, scale: float) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int) -> None:
    if hasattr(env, "seed"):
        env.seed(seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)


if __name__ == "__main__":
    main()
