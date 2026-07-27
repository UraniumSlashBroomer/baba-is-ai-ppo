"""Train NanoWorldModel on PPO Baba samples.

Usage:
    python -m nanowm.train --config nanowm/train_config.yaml
    python -m nanowm.train --config nanowm/train_config.yaml data.samples_root=samples/my_run train.max_steps=1000
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .dataset import BabaSamplesNanoWMDataset, create_train_val_datasets
from .policy import LatentActionHead
from .validation import run_env_success_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOWM_SRC = PROJECT_ROOT / "nano-world-model" / "src"
if str(NANOWM_SRC) not in sys.path:
    sys.path.insert(0, str(NANOWM_SRC))

def main() -> None:
    cfg = load_config()
    create_diffusion, sample_training_timesteps, build_latent_codec, resolve_latent_codec_config, get_models, dfot_sample = (
        import_nanowm_components()
    )
    seed_everything(int(cfg.seed))

    device = resolve_device(str(cfg.device))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_dir / "resolved_config.yaml", resolve=True)

    train_dataset, val_dataset = build_datasets(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = get_models(cfg).to(device)
    action_head = LatentActionHead(
        latent_channels=int(cfg.model.latent_channels),
        latent_size=int(cfg.model.latent_size),
        action_dim=int(cfg.data.action_dim),
        hidden_size=int(cfg.action_head.hidden_size),
    ).to(device)
    if bool(cfg.train.compile):
        model = torch.compile(model)

    latent_codec_cfg = resolve_latent_codec_config(cfg)
    latent_codec = build_latent_codec(cfg).to(device).eval().requires_grad_(False)
    sanity_check_latent_codec(latent_codec, latent_codec_cfg, cfg, device)

    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=str(cfg.diffusion.noise_schedule),
        pred_name=str(cfg.diffusion.pred_name),
        diffusion_steps=int(cfg.diffusion.diffusion_steps),
        snr_gamma=float(cfg.diffusion.snr_gamma),
        zero_terminal_snr=bool(cfg.diffusion.zero_terminal_snr),
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": float(cfg.train.lr)},
            {"params": action_head.parameters(), "lr": float(cfg.train.action_lr)},
        ],
        weight_decay=float(cfg.train.weight_decay),
    )
    wandb_run = init_wandb(cfg, output_dir)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    latest_grad_metrics: Dict[str, float] = {}
    start_time = time.time()
    model.train()
    action_head.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < int(cfg.train.max_steps):
        for batch in train_loader:
            loss, metrics = forward_loss(cfg, batch, model, action_head, latent_codec, diffusion, sample_training_timesteps, device)
            loss_value = float(loss.detach().cpu())
            (loss / int(cfg.train.gradient_accumulation)).backward()
            micro_step += 1

            if micro_step % int(cfg.train.gradient_accumulation) != 0:
                continue

            if float(cfg.train.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(action_head.parameters()),
                    float(cfg.train.grad_clip_norm),
                )
            latest_grad_metrics = collect_gradient_metrics(model, action_head)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += loss_value
            log_metrics(wandb_run, metrics, global_step)
            if global_step % int(cfg.train.log_every) == 0:
                avg_loss = running_loss / int(cfg.train.log_every)
                elapsed = time.time() - start_time
                print(
                    f"step={global_step:07d} train_loss={avg_loss:.6f} "
                    f"wm_loss={metrics['train/wm_loss']:.6f} action_loss={metrics['train/action_loss']:.6f} "
                    f"action_acc={metrics['train/action_acc']:.3f} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
                running_loss = 0.0

            if global_step % int(cfg.train.val_every) == 0:
                val_metrics = validate(cfg, val_loader, model, action_head, latent_codec, diffusion, sample_training_timesteps, device)
                env_metrics = run_env_success_validation(
                    cfg,
                    model=model,
                    action_head=action_head,
                    latent_codec=latent_codec,
                    diffusion=diffusion,
                    dfot_sample=dfot_sample,
                    device=device,
                )
                val_metrics.update(env_metrics)
                val_metrics.update(collect_diagnostic_metrics(model, action_head, latest_grad_metrics))
                log_metrics(wandb_run, val_metrics, global_step)
                print(
                    f"step={global_step:07d} val_loss={val_metrics['val/loss']:.6f} "
                    f"success_rate={val_metrics['val/success_rate']:.3f} "
                    f"avg_return={val_metrics['val/avg_return']:.3f}",
                    flush=True,
                )
                model.train()
                action_head.train()

            if global_step % int(cfg.train.save_every) == 0:
                save_checkpoint(output_dir, cfg, model, action_head, optimizer, global_step, val_loss=None)
                prune_checkpoints(output_dir, int(cfg.train.keep_last_checkpoints))

            if global_step >= int(cfg.train.max_steps):
                break

    final_metrics = validate(cfg, val_loader, model, action_head, latent_codec, diffusion, sample_training_timesteps, device)
    final_metrics.update(
        run_env_success_validation(
            cfg,
            model=model,
            action_head=action_head,
            latent_codec=latent_codec,
            diffusion=diffusion,
            dfot_sample=dfot_sample,
            device=device,
        )
    )
    final_metrics.update(collect_diagnostic_metrics(model, action_head, latest_grad_metrics))
    log_metrics(wandb_run, final_metrics, global_step)
    save_checkpoint(output_dir, cfg, model, action_head, optimizer, global_step, val_loss=final_metrics["val/loss"], name="final.pt")
    if wandb_run is not None:
        wandb_run.finish()
    print(
        f"done step={global_step:07d} final_val_loss={final_metrics['val/loss']:.6f} "
        f"final_success_rate={final_metrics['val/success_rate']:.3f}",
        flush=True,
    )


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "nanowm" / "train_config.yaml"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def import_nanowm_components():
    try:
        from diffusion import create_diffusion, sample_training_timesteps
        from diffusion.df_sample import dfot_sample
        from latent_codecs import build_latent_codec, resolve_latent_codec_config
        from models import get_models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency while importing nano-world-model: {exc.name}. "
            "Run this train script inside the environment where nano-world-model dependencies are installed."
        ) from exc
    return create_diffusion, sample_training_timesteps, build_latent_codec, resolve_latent_codec_config, get_models, dfot_sample


def build_datasets(cfg) -> tuple[BabaSamplesNanoWMDataset, BabaSamplesNanoWMDataset]:
    return create_train_val_datasets(
        cfg.data.samples_root,
        num_frames=int(cfg.model.num_frames),
        frame_interval=int(cfg.dataset.frame_interval),
        image_size=int(cfg.model.image_size),
        action_dim=int(cfg.data.action_dim),
        action_encoding=str(cfg.data.action_encoding),
        action_offset=int(cfg.data.action_offset),
        split_ratio=float(cfg.data.split_ratio),
        random_seed=int(cfg.seed),
        slice_mode=str(cfg.data.slice_mode),
        stride=int(cfg.data.stride),
        resize_mode=str(cfg.data.resize_mode),
        normalize_pixel=True,
    )


@torch.no_grad()
def validate(cfg, loader, model, action_head, latent_codec, diffusion, sample_training_timesteps, device: torch.device) -> Dict[str, float]:
    model.eval()
    action_head.eval()
    metrics_sum: Dict[str, float] = {}
    batches = 0
    for batch in loader:
        _, metrics = forward_loss(cfg, batch, model, action_head, latent_codec, diffusion, sample_training_timesteps, device)
        for key, value in metrics.items():
            val_key = key.replace("train/", "val/")
            metrics_sum[val_key] = metrics_sum.get(val_key, 0.0) + float(value)
        batches += 1
    if batches == 0:
        return {"val/loss": math.nan, "val/wm_loss": math.nan, "val/action_loss": math.nan, "val/action_acc": math.nan}
    return {key: value / batches for key, value in metrics_sum.items()}


def forward_loss(cfg, batch, model, action_head, latent_codec, diffusion, sample_training_timesteps, device: torch.device) -> tuple[torch.Tensor, Dict[str, float]]:
    video = batch["video"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True) if bool(cfg.model.use_action) else None

    with torch.no_grad():
        batch_size, num_frames, channels, height, width = video.shape
        flat_video = video.reshape(batch_size * num_frames, channels, height, width).contiguous()
        latents = latent_codec.encode(flat_video)
        latent_channels, latent_h, latent_w = latents.shape[1:]
        latents = latents.reshape(batch_size, num_frames, latent_channels, latent_h, latent_w).contiguous()

    action_logits = action_head(latents[:, 0])
    action_target = action[:, 0].argmax(dim=-1) if action is not None else None
    action_loss = F.cross_entropy(action_logits, action_target) if action_target is not None else torch.zeros((), device=device)
    action_acc = (
        (action_logits.argmax(dim=-1) == action_target).float().mean()
        if action_target is not None
        else torch.zeros((), device=device)
    )

    model_kwargs: Dict[str, Any] = {"y": None}
    if action is not None:
        model_kwargs["action"] = action

    t_shape = (latents.shape[0], latents.shape[1])
    if str(cfg.diffusion.mode) == "full_seq_diffusion":
        t_shape = (latents.shape[0],)
    elif str(cfg.diffusion.mode) != "diffusion_forcing":
        raise ValueError(f"Unsupported diffusion.mode={cfg.diffusion.mode!r}")

    t = sample_training_timesteps(
        t_shape,
        diffusion.num_timesteps,
        strategy=str(cfg.diffusion.timestep_sampling),
        logit_normal_mean=float(cfg.diffusion.logit_normal_mean),
        logit_normal_std=float(cfg.diffusion.logit_normal_std),
        device=device,
    )

    with autocast_context(cfg, device):
        if str(cfg.train.target_frames) == "all":
            wm_loss = diffusion.training_losses(model, latents, t, model_kwargs)["loss"].mean()
        else:
            wm_loss = masked_diffusion_loss(
                cfg=cfg,
                model=model,
                diffusion=diffusion,
                x_start=latents,
                t=t,
                model_kwargs=model_kwargs,
            )

    loss = wm_loss + float(cfg.train.action_loss_weight) * action_loss
    return loss, {
        "train/loss": float(loss.detach().cpu()),
        "train/wm_loss": float(wm_loss.detach().cpu()),
        "train/action_loss": float(action_loss.detach().cpu()),
        "train/action_acc": float(action_acc.detach().cpu()),
    }


def masked_diffusion_loss(cfg, model, diffusion, x_start, t, model_kwargs) -> torch.Tensor:
    noise = torch.randn_like(x_start)
    x_t = diffusion.q_sample(x_start, t, noise=noise)
    target = diffusion_target(cfg, diffusion, x_start, t, noise)
    model_output = model(x_t, t, **model_kwargs)
    if model_output.shape != target.shape:
        raise RuntimeError(f"model_output shape {tuple(model_output.shape)} != target shape {tuple(target.shape)}")

    per_frame = ((model_output - target) ** 2).mean(dim=(2, 3, 4))
    frame_mask = target_frame_mask(str(cfg.train.target_frames), per_frame.shape, per_frame.device)
    if hasattr(diffusion, "alphas_cumprod") and float(cfg.diffusion.snr_gamma) > 0:
        per_frame = per_frame * min_snr_weights(diffusion, t, float(cfg.diffusion.snr_gamma), per_frame.shape)
    return (per_frame * frame_mask).sum() / frame_mask.sum().clamp_min(1.0)


def diffusion_target(cfg, diffusion, x_start, t, noise):
    pred_name = str(cfg.diffusion.pred_name)
    if pred_name == "flow":
        return x_start - noise
    if pred_name in ("eps", "epsilon"):
        return noise
    if pred_name == "x":
        return x_start
    if pred_name == "v":
        return diffusion._predict_v(x_start, t, noise)
    raise ValueError(f"Unsupported diffusion.pred_name={pred_name!r}")


def target_frame_mask(mode: str, shape: torch.Size, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.float32, device=device)
    if mode == "next_only":
        if shape[1] < 2:
            raise ValueError("target_frames='next_only' requires model.num_frames >= 2")
        mask[:, 1] = 1.0
    elif mode == "future":
        mask[:, 1:] = 1.0
    else:
        raise ValueError("train.target_frames must be 'next_only', 'future', or 'all'")
    return mask


def min_snr_weights(diffusion, t, gamma: float, shape: torch.Size) -> torch.Tensor:
    if t.ndim == 1:
        t_for_frames = t[:, None].expand(shape)
    else:
        t_for_frames = t
    alphas = torch.as_tensor(diffusion.alphas_cumprod, device=t.device, dtype=torch.float32)
    alpha = alphas[t_for_frames.clamp(min=0)]
    snr = alpha / (1.0 - alpha).clamp(min=1e-8)
    return torch.clamp(snr.clamp(min=1e-8), max=gamma) / snr.clamp(min=1e-8)


def sanity_check_latent_codec(latent_codec, latent_codec_cfg, cfg, device: torch.device) -> None:
    probe = torch.zeros(1, 3, int(cfg.model.image_size), int(cfg.model.image_size), device=device)
    with torch.no_grad():
        z = latent_codec.encode(probe)
    expected = tuple(latent_codec_cfg.latent_shape.as_tuple())
    actual = tuple(z.shape[1:])
    if actual != expected:
        raise RuntimeError(f"Latent codec produced {actual}, expected {expected}")
    print(f"latent_codec={latent_codec_cfg.kind} shape={actual} path={latent_codec_cfg.model_path}", flush=True)


def save_checkpoint(
    output_dir: Path,
    cfg,
    model,
    action_head,
    optimizer,
    global_step: int,
    *,
    val_loss: Optional[float],
    name: Optional[str] = None,
) -> None:
    path = output_dir / (name or f"step_{global_step:07d}.pt")
    torch.save(
        {
            "step": global_step,
            "model": unwrap_compile(model).state_dict(),
            "action_head": action_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "val_loss": val_loss,
        },
        path,
    )
    print(f"saved checkpoint {path}", flush=True)


def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def collect_diagnostic_metrics(
    model,
    action_head,
    latest_grad_metrics: Dict[str, float],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(collect_weight_metrics("diagnostics/model", unwrap_compile(model)))
    metrics.update(collect_weight_metrics("diagnostics/action_head", action_head))
    metrics.update(latest_grad_metrics)
    return metrics


@torch.no_grad()
def collect_weight_metrics(prefix: str, module: torch.nn.Module) -> Dict[str, float]:
    total_params = 0
    total_abs = 0.0
    total_sq = 0.0
    max_abs = 0.0
    nonfinite = 0

    for param in module.parameters():
        data = param.detach().float()
        count = data.numel()
        if count == 0:
            continue
        finite = torch.isfinite(data)
        nonfinite += int((~finite).sum().item())
        finite_data = data[finite]
        if finite_data.numel() == 0:
            total_params += count
            continue
        abs_data = finite_data.abs()
        total_params += count
        total_abs += float(abs_data.sum().item())
        total_sq += float((finite_data * finite_data).sum().item())
        max_abs = max(max_abs, float(abs_data.max().item()))

    denom = max(1, total_params - nonfinite)
    return {
        f"{prefix}/param_count": float(total_params),
        f"{prefix}/weight_l2": total_sq ** 0.5,
        f"{prefix}/weight_abs_mean": total_abs / denom,
        f"{prefix}/weight_abs_max": max_abs,
        f"{prefix}/weight_nonfinite": float(nonfinite),
    }


def collect_gradient_metrics(model, action_head) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(_collect_gradient_metrics_for_module("diagnostics/model", unwrap_compile(model)))
    metrics.update(_collect_gradient_metrics_for_module("diagnostics/action_head", action_head))
    return metrics


def _collect_gradient_metrics_for_module(prefix: str, module: torch.nn.Module) -> Dict[str, float]:
    total_values = 0
    total_abs = 0.0
    total_sq = 0.0
    max_abs = 0.0
    tensors_with_grad = 0
    nonfinite = 0

    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        count = grad.numel()
        if count == 0:
            continue
        tensors_with_grad += 1
        finite = torch.isfinite(grad)
        nonfinite += int((~finite).sum().item())
        finite_grad = grad[finite]
        if finite_grad.numel() == 0:
            total_values += count
            continue
        abs_grad = finite_grad.abs()
        total_values += count
        total_abs += float(abs_grad.sum().item())
        total_sq += float((finite_grad * finite_grad).sum().item())
        max_abs = max(max_abs, float(abs_grad.max().item()))

    denom = max(1, total_values - nonfinite)
    return {
        f"{prefix}/grad_tensor_count": float(tensors_with_grad),
        f"{prefix}/grad_l2": total_sq ** 0.5,
        f"{prefix}/grad_abs_mean": total_abs / denom,
        f"{prefix}/grad_abs_max": max_abs,
        f"{prefix}/grad_nonfinite": float(nonfinite),
    }


def unwrap_compile(model):
    return getattr(model, "_orig_mod", model)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(cfg, device: torch.device):
    if not bool(cfg.train.mixed_precision) or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def init_wandb(cfg, output_dir: Path):
    if not bool(cfg.wandb.enabled):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("wandb.enabled=true, but wandb is not installed") from exc
    return wandb.init(
        project=str(cfg.wandb.project),
        entity=None if cfg.wandb.entity in (None, "null") else str(cfg.wandb.entity),
        mode=str(cfg.wandb.mode),
        name=str(cfg.wandb.name),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def log_metrics(wandb_run, metrics: Dict[str, float], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


if __name__ == "__main__":
    main()
