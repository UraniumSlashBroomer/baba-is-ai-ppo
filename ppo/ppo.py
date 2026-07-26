import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
import cv2
import numpy as np
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baba-is-ai"))


def load_config(path):
    text = Path(path).read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        cfg = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip("\"'")
            cfg[key.strip()] = value
        return cfg


def make_env(cfg):
    import baba

    env = baba.make(cfg["env_id"])
    env.max_steps = cfg["max_episode_steps"]
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_to_tensor(obs, device):
    x = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    return x.unsqueeze(0)


def render_live_step(env, render_fps):
    if render_fps <= 0:
        return
    start = time.perf_counter()
    env.render(mode="human")
    delay = (1.0 / render_fps) - (time.perf_counter() - start)
    if delay > 0:
        time.sleep(delay)


def policy_action_count(env, cfg):
    return env.action_space.n if cfg.get("use_idle", False) else env.action_space.n - 1


def to_env_action(policy_action, cfg):
    return int(policy_action) if cfg.get("use_idle", False) else int(policy_action) + 1


class ShapedBabaEnv:
    def __init__(self, env, cfg, warmup_episodes=0):
        self.env = env
        self.cfg = cfg
        self.warmup_episodes = warmup_episodes
        self.episode_idx = 0
        self.rule_reward_given = False
        self.rule_dead_penalty_given = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        if self.episode_idx < self.warmup_episodes:
            np.random.seed(self.cfg.get("warmup_seed", self.cfg["seed"]))
        obs = self.env.reset()
        self.rule_reward_given = self._is_win_rule_active()
        self.rule_dead_penalty_given = False
        self.episode_idx += 1
        return obs

    def step(self, action):
        hit_boundary = self._hits_boundary(action)
        pushed_before = self._pushed_block_before(action)
        obs, env_reward, done, info = self.env.step(action)
        reward = env_reward
        stuck_push = self._is_stuck_push(pushed_before)
        use_shaped_reward = self.cfg.get("use_shaped_reward", True)

        rule_assembled = False
        if not self.rule_reward_given and self._is_win_rule_active():
            rule_assembled = True
            self.rule_reward_given = True
            if use_shaped_reward:
                reward += self.cfg.get("rule_assembly_reward", 0)

        if use_shaped_reward and hit_boundary:
            reward += self.cfg.get("boundary_penalty", 0)

        if use_shaped_reward and stuck_push:
            reward += self.cfg.get("stuck_push_penalty", 0)

        rule_dead = False
        if not self.rule_reward_given and not self.rule_dead_penalty_given and self._is_target_rule_dead():
            rule_dead = True
            self.rule_dead_penalty_given = True
            if use_shaped_reward:
                reward += self.cfg.get("rule_dead_penalty", 0)

        info = dict(info)
        info["env_reward"] = env_reward
        info["rule_assembled"] = rule_assembled
        info["hit_boundary"] = hit_boundary
        info["stuck_push"] = stuck_push
        info["rule_dead"] = rule_dead
        return obs, reward, done, info

    def _hits_boundary(self, action):
        vectors = {
            self.env.actions.up: np.array((0, -1)),
            self.env.actions.right: np.array((1, 0)),
            self.env.actions.down: np.array((0, 1)),
            self.env.actions.left: np.array((-1, 0)),
        }
        if action not in vectors:
            return False
        front_pos = np.array(self.env.agent_pos) + vectors[action]
        x, y = front_pos
        return x <= 0 or y <= 0 or x >= self.env.width - 1 or y >= self.env.height - 1

    def _pushed_block_before(self, action):
        vectors = {
            self.env.actions.up: np.array((0, -1)),
            self.env.actions.right: np.array((1, 0)),
            self.env.actions.down: np.array((0, 1)),
            self.env.actions.left: np.array((-1, 0)),
        }
        if action not in vectors:
            return None
        pos = tuple(np.array(self.env.agent_pos) + vectors[action])
        if not self._inside_grid(pos):
            return None
        cell = self.env.grid.get(*pos)
        if cell is None or not cell.is_push():
            return None
        return cell, pos

    def _is_stuck_push(self, pushed_before):
        if pushed_before is None:
            return False
        block, old_pos = pushed_before
        for y in range(self.env.height):
            for x in range(self.env.width):
                objects = self.env.grid.get(x, y, z="all")
                if block in objects:
                    return (x, y) == old_pos
        return False

    def _is_win_rule_active(self):
        win_obj = getattr(self.env, "win_obj", None)
        if win_obj is None:
            return False

        color = None
        if isinstance(win_obj, tuple):
            color, win_obj = win_obj
        win_obj = self._canonical_rule_object_name(win_obj)

        goal_rules = self.env.get_ruleset().get("is_goal", {})
        if not goal_rules.get(win_obj, False):
            return False
        if color is None:
            return True
        return color in goal_rules.get(f"{win_obj}_color", [])

    def _is_target_rule_dead(self):
        blocks = self._target_rule_blocks()
        if not blocks["object"] or not blocks["is"] or not blocks["win"]:
            return False

        for role, positions in blocks.items():
            for pos in positions:
                if self._is_dead_rule_position(pos, role):
                    return True
        return False

    def _target_rule_blocks(self):
        win_obj = getattr(self.env, "win_obj", None)
        if isinstance(win_obj, tuple):
            _, win_obj = win_obj
        win_obj = self._canonical_rule_object_name(win_obj)

        blocks = {"object": [], "is": [], "win": []}
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell = self.env.grid.get(x, y)
                if cell is None:
                    continue
                if cell.type == "rule_object" and getattr(cell, "object", None) == win_obj:
                    blocks["object"].append((x, y))
                elif cell.type == "rule_is" and not self._is_you_rule_is_position((x, y)):
                    blocks["is"].append((x, y))
                elif cell.type == "rule_property" and getattr(cell, "property", None) == "is_goal":
                    blocks["win"].append((x, y))
        return blocks

    def _canonical_rule_object_name(self, name):
        return {
            "wall": "fwall",
            "ball": "fball",
            "door": "fdoor",
            "key": "fkey",
        }.get(name, name)

    def _is_dead_rule_position(self, pos, role):
        x, y = pos
        left, right = 1, self.env.width - 2
        top, bottom = 1, self.env.height - 2

        can_be_rule_slot = self._can_be_valid_rule_slot(pos, role)
        if not self._has_legal_push(pos):
            return not can_be_rule_slot

        if (x in {left, right}) and (y in {top, bottom}):
            return not can_be_rule_slot

        on_border = x in {left, right} or y in {top, bottom}
        if on_border:
            return not can_be_rule_slot
        return False

    def _can_be_valid_rule_slot(self, pos, role):
        x, y = pos
        offsets = {"object": 0, "is": 1, "win": 2}
        offset = offsets[role]

        horizontal_start = x - offset
        if 1 <= horizontal_start and horizontal_start + 2 <= self.env.width - 2:
            return True

        vertical_start = y - offset
        if 1 <= vertical_start and vertical_start + 2 <= self.env.height - 2:
            return True

        return False

    def _has_legal_push(self, pos):
        x, y = pos
        directions = [
            np.array((0, -1)),
            np.array((1, 0)),
            np.array((0, 1)),
            np.array((-1, 0)),
        ]

        for direction in directions:
            stand_pos = np.array((x, y)) - direction
            dest_pos = np.array((x, y)) + direction
            if not self._is_playable_pos(stand_pos) or not self._is_playable_pos(dest_pos):
                continue
            if self._cell_blocks_push(dest_pos):
                continue
            if self._cell_blocks_agent_stand(stand_pos):
                continue
            return True
        return False

    def _is_playable_pos(self, pos):
        x, y = pos
        return 1 <= x <= self.env.width - 2 and 1 <= y <= self.env.height - 2

    def _cell_blocks_push(self, pos):
        cell = self.env.grid.get(*pos)
        return cell is not None and not cell.can_overlap()

    def _cell_blocks_agent_stand(self, pos):
        cell = self.env.grid.get(*pos)
        return cell is not None and not cell.can_overlap()

    def _is_you_rule_is_position(self, pos):
        x, y = pos
        candidates = [
            ((x - 1, y), (x + 1, y)),
            ((x, y - 1), (x, y + 1)),
        ]
        for obj_pos, prop_pos in candidates:
            if not self._inside_grid(obj_pos) or not self._inside_grid(prop_pos):
                continue
            obj = self.env.grid.get(*obj_pos)
            prop = self.env.grid.get(*prop_pos)
            if (
                obj is not None
                and prop is not None
                and obj.type == "rule_object"
                and getattr(obj, "object", None) == "baba"
                and prop.type == "rule_property"
                and getattr(prop, "property", None) == "is_agent"
            ):
                return True
        return False

    def _inside_grid(self, pos):
        x, y = pos
        return 0 <= x < self.env.width and 0 <= y < self.env.height


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        in_channels = obs_shape[-1]
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, obs_shape[0], obs_shape[1])
            cnn_dim = self.actor_cnn(dummy).shape[1]

        self.actor_net = nn.Sequential(
            nn.Linear(cnn_dim, hidden_size),
            nn.Tanh(),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(cnn_dim, hidden_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        actor_h = self.actor_net(self.actor_cnn(x))
        critic_h = self.critic_net(self.critic_cnn(x))
        return self.actor_head(actor_h), self.critic_head(critic_h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class SplitMlpActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        actor_h = self.actor_net(x)
        critic_h = self.critic_net(x)
        return self.actor_head(actor_h), self.critic_head(critic_h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class SharedActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.net(x.flatten(start_dim=1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def save_checkpoint(model, optimizer, cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }, path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def load_model_for_checkpoint(env, cfg, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"]
    if "actor_cnn.0.weight" in state:
        n_actions = state["actor_head.weight"].shape[0]
        hidden_size = state["actor_head.weight"].shape[1]
        model_cls = ActorCritic
    elif "actor_head.weight" in state:
        n_actions = state["actor_head.weight"].shape[0]
        hidden_size = state["actor_head.weight"].shape[1]
        model_cls = SplitMlpActorCritic
    else:
        n_actions = state["actor.weight"].shape[0]
        hidden_size = state["actor.weight"].shape[1]
        model_cls = SharedActorCritic

    if n_actions == env.action_space.n:
        cfg["use_idle"] = True
    elif n_actions == env.action_space.n - 1:
        cfg["use_idle"] = False

    model = model_cls(env.observation_space.shape, n_actions, hidden_size).to(device)
    model.load_state_dict(state)
    return model


def rollout(env, model, cfg, device, render_fps=0):
    obs = env.reset()
    obs_buf, action_buf, logprob_buf = [], [], []
    reward_buf, done_buf, value_buf = [], [], []
    episode_returns = []
    episode_successes = []
    episode_lengths = []
    episode_env_returns = []
    episode_return = 0
    episode_env_return = 0
    episode_length = 0
    rule_assemblies = 0
    boundary_hits = 0
    stuck_pushes = 0
    rule_dead_events = 0

    for _ in range(cfg["rollout_steps"]):
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
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
        if done:
            episode_returns.append(episode_return)
            episode_env_returns.append(episode_env_return)
            episode_successes.append(episode_env_return > 0)
            episode_lengths.append(episode_length)
            episode_return = 0
            episode_env_return = 0
            episode_length = 0
            obs = env.reset()

    with torch.no_grad():
        next_value = model(obs_to_tensor(obs, device))[1].item()

    rewards = np.array(reward_buf, dtype=np.float32)
    dones = np.array(done_buf, dtype=np.float32)
    values = np.array(value_buf + [next_value], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + cfg["gamma"] * values[t + 1] * nonterminal - values[t]
        gae = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values[:-1]

    return {
        "obs": np.array(obs_buf),
        "actions": np.array(action_buf),
        "logprobs": np.array(logprob_buf, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "episode_returns": episode_returns,
        "episode_env_returns": episode_env_returns,
        "episode_successes": episode_successes,
        "episode_lengths": episode_lengths,
        "rule_assemblies": rule_assemblies,
        "boundary_hits": boundary_hits,
        "stuck_pushes": stuck_pushes,
        "rule_dead_events": rule_dead_events,
    }


def train(cfg, render_fps=0):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    env = make_env(cfg)
    warmup_ratio = cfg.get("warmup_episode_ratio", 0)
    estimated_episodes = max(1, cfg["total_steps"] // cfg["max_episode_steps"])
    warmup_episodes = int(estimated_episodes * warmup_ratio)
    env = ShapedBabaEnv(env, cfg, warmup_episodes=warmup_episodes)
    obs_shape = env.observation_space.shape
    n_actions = policy_action_count(env, cfg)

    model = ActorCritic(obs_shape, n_actions, cfg["hidden_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    run_dir = ROOT / "ppo" / "checkpoints" / cfg["name"]
    log_dir = ROOT / "ppo" / "logs" / cfg["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    n_updates = cfg["total_steps"] // cfg["rollout_steps"]

    for update in range(1, n_updates + 1):
        batch = rollout(env, model, cfg, device, render_fps=render_fps)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device) / 255.0
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
        old_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(cfg["rollout_steps"])
        policy_losses, value_losses, entropy_losses = [], [], []
        for _ in range(cfg["update_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg["minibatch_size"]):
                mb = indices[start:start + cfg["minibatch_size"]]
                new_logprobs, entropy, values = model.evaluate(obs[mb], actions[mb])
                ratio = (new_logprobs - old_logprobs[mb]).exp()

                pg_loss1 = -advantages[mb] * ratio
                pg_loss2 = -advantages[mb] * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
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

        avg_return = np.mean(batch["episode_returns"]) if batch["episode_returns"] else 0
        avg_env_return = np.mean(batch["episode_env_returns"]) if batch["episode_env_returns"] else 0
        success_rate = np.mean(batch["episode_successes"]) if batch["episode_successes"] else 0
        avg_length = np.mean(batch["episode_lengths"]) if batch["episode_lengths"] else 0
        stats = {
            "update": update,
            "steps": update * cfg["rollout_steps"],
            "avg_shaped_return": float(avg_return),
            "avg_env_return": float(avg_env_return),
            "success_rate": float(success_rate),
            "avg_episode_length": float(avg_length),
            "rule_assemblies": int(batch["rule_assemblies"]),
            "boundary_hits": int(batch["boundary_hits"]),
            "stuck_pushes": int(batch["stuck_pushes"]),
            "rule_dead_events": int(batch["rule_dead_events"]),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(stats) + "\n")
        print(
            f"update={update}/{n_updates} steps={stats['steps']} "
            f"shaped_return={avg_return:.3f} env_return={avg_env_return:.3f} "
            f"success_rate={success_rate:.3f} avg_len={avg_length:.1f} "
            f"rules={batch['rule_assemblies']} boundary_hits={batch['boundary_hits']} "
            f"stuck_pushes={batch['stuck_pushes']} dead_rules={batch['rule_dead_events']}"
        )

        if update % cfg["checkpoint_every"] == 0 or update == n_updates:
            save_checkpoint(model, optimizer, cfg, run_dir / f"ppo_{update:04d}.pt")
            save_checkpoint(model, optimizer, cfg, run_dir / "latest.pt")


def val(cfg, checkpoint, visualize=False, render_fps=0):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    env = make_env(cfg)
    model = load_model_for_checkpoint(env, cfg, checkpoint, device)
    model.eval()

    successes = 0
    returns = []
    for ep in range(cfg["eval_episodes"]):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                logits, _ = model(obs_to_tensor(obs, device))
                policy_action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, _ = env.step(to_env_action(policy_action, cfg))
            total_reward += reward
            if visualize:
                render_live_step(env, render_fps or cfg.get("eval_render_fps", 30))
        successes += total_reward > 0
        returns.append(total_reward)
        print(f"episode={ep + 1} return={total_reward:.3f}")

    print(f"success_rate={successes / cfg['eval_episodes']:.3f} avg_return={np.mean(returns):.3f}")


def sample(cfg, checkpoint, visualize=False):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    env = make_env(cfg)
    model = load_model_for_checkpoint(env, cfg, checkpoint, device)
    model.eval()

    out_dir = ROOT / "samples" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg["sample_episodes"]):
        obs = env.reset()
        frames, observations, actions, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            if visualize:
                frames.append(env.render(mode="rgb_array"))
            with torch.no_grad():
                action, _, _, _ = model.act(obs_to_tensor(obs, device))
            env_action = to_env_action(action.item(), cfg)
            next_obs, reward, done, _ = env.step(env_action)

            observations.append(obs)
            actions.append(env_action)
            rewards.append(reward)
            dones.append(done)
            obs = next_obs

            if done and visualize:
                frames.append(env.render(mode="rgb_array"))

        np.savez_compressed(
            out_dir / f"episode_{ep:04d}.npz",
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones),
        )

        if visualize and frames:
            video_path = out_dir / f"episode_{ep:04d}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

        print(f"sampled episode={ep + 1} return={sum(rewards):.3f} steps={len(actions)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val", "sample"], required=True)
    parser.add_argument("--config", default="ppo/configs/make_win.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--render-fps", type=float, default=0, help="Live render FPS. Training renders only when this is > 0 or --visualize is set.")
    parser.add_argument("--override", default=None, help='JSON dict with config overrides, e.g. {"total_steps":1000}')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg.update(json.loads(args.override))

    checkpoint = args.checkpoint
    if checkpoint is None and args.mode in {"val", "sample"}:
        checkpoint = ROOT / "ppo" / "checkpoints" / cfg["name"] / "latest.pt"

    if args.mode == "train":
        render_fps = args.render_fps or (cfg.get("train_render_fps", 0) if args.visualize else 0)
        train(cfg, render_fps=render_fps)
    elif args.mode == "val":
        val(cfg, checkpoint, visualize=args.visualize, render_fps=args.render_fps)
    elif args.mode == "sample":
        sample(cfg, checkpoint, visualize=args.visualize)


if __name__ == "__main__":
    main()
