import cv2
import numpy as np
import torch

from checkpoints import checkpoint_architecture, load_model_for_checkpoint
from env import make_env, policy_action_count, to_env_action
from models import detach_state, is_recurrent_model
from paths import ROOT
from utils import obs_to_tensor, render_validation_step, set_seed


def _policy_step(model, obs, cfg, device, recurrent_state=None, greedy=False):
    with torch.no_grad():
        if is_recurrent_model(model):
            if greedy:
                logits, _, recurrent_state = model(obs_to_tensor(obs, device), recurrent_state)
                action = torch.argmax(logits, dim=-1)
            else:
                action, _, _, _, recurrent_state = model.act(obs_to_tensor(obs, device), recurrent_state)
            recurrent_state = detach_state(recurrent_state)
        elif greedy:
            logits, _ = model(obs_to_tensor(obs, device))
            action = torch.argmax(logits, dim=-1)
        else:
            action, _, _, _ = model.act(obs_to_tensor(obs, device))
    return to_env_action(action.item(), cfg), recurrent_state


def val(cfg, checkpoint, visualize=False, render_fps=0, greedy=None):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if checkpoint:
        cfg.update(checkpoint_architecture(checkpoint, device))
    if greedy is None:
        greedy = cfg.get("eval_greedy", True)
    env_cfg = dict(cfg, env_id=cfg.get("eval_env_id", cfg["env_id"]))
    env = make_env(env_cfg)
    model = load_model_for_checkpoint(env, cfg, checkpoint, device)
    model.eval()

    successes, returns = 0, []
    for ep in range(cfg["eval_episodes"]):
        obs = env.reset()
        done = False
        total_reward = 0
        timestep = 0
        recurrent_state = model.initial_state(1, device) if is_recurrent_model(model) else None
        while not done:
            env_action, recurrent_state = _policy_step(model, obs, cfg, device, recurrent_state, greedy=greedy)
            obs, reward, done, _ = env.step(env_action)
            timestep += 1
            total_reward += reward
            if visualize:
                render_validation_step(
                    env,
                    render_fps or cfg.get("eval_render_fps", 30),
                    env_action,
                    timestep,
                    reward,
                    total_reward,
                )
        successes += total_reward > 0
        returns.append(total_reward)
        print(f"episode={ep + 1} return={total_reward:.3f}")

    print(f"success_rate={successes / cfg['eval_episodes']:.3f} avg_return={np.mean(returns):.3f}")


def sample(cfg, checkpoint, visualize=False, greedy=None):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    random_policy = bool(cfg.get("random_policy", False))
    if checkpoint and not random_policy:
        cfg.update(checkpoint_architecture(checkpoint, device))
    if greedy is None:
        greedy = cfg.get("sample_greedy", False)
    env_cfg = dict(cfg, env_id=cfg.get("sample_env_id", cfg.get("eval_env_id", cfg["env_id"])))
    env = make_env(env_cfg)
    model = None
    if not random_policy:
        model = load_model_for_checkpoint(env, cfg, checkpoint, device)
        model.eval()

    out_dir = ROOT / "samples" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_start = cfg.get("sample_seed_start")

    for ep in range(cfg["sample_episodes"]):
        episode_seed = None if seed_start is None else int(seed_start) + ep
        if episode_seed is not None:
            set_seed(episode_seed)
            if hasattr(env, "seed"):
                env.seed(episode_seed)
            if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                env.action_space.seed(episode_seed)
        obs = env.reset()
        frames, actions, rewards = [env.render(mode="rgb_array")], [], []
        done = False
        recurrent_state = model.initial_state(1, device) if model is not None and is_recurrent_model(model) else None
        while not done:
            if random_policy:
                policy_action = int(np.random.randint(policy_action_count(env, cfg)))
                env_action = to_env_action(policy_action, cfg)
            else:
                env_action, recurrent_state = _policy_step(model, obs, cfg, device, recurrent_state, greedy=greedy)
            next_obs, reward, done, _ = env.step(env_action)

            actions.append(env_action)
            rewards.append(reward)
            frames.append(env.render(mode="rgb_array"))
            obs = next_obs

        episode_dir = out_dir / f"episode_{ep:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.save(episode_dir / "frames.npy", np.asarray(frames, dtype=np.uint8))
        np.save(episode_dir / "actions.npy", np.asarray(actions))
        if episode_seed is not None:
            np.save(episode_dir / "seed.npy", np.asarray(episode_seed, dtype=np.int64))

        if visualize and frames:
            video_path = out_dir / f"episode_{ep:04d}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

        seed_text = f" seed={episode_seed}" if episode_seed is not None else ""
        print(f"sampled episode={ep + 1}{seed_text} return={sum(rewards):.3f} steps={len(actions)}")
