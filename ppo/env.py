import numpy as np
from gym import spaces

import paths  # noqa: F401 - ensures local baba package is importable


DIRS = (
    np.array((0, -1)),
    np.array((1, 0)),
    np.array((0, 1)),
    np.array((-1, 0)),
)
CANONICAL_RULE_OBJECTS = {
    "wall": "fwall",
    "ball": "fball",
    "door": "fdoor",
    "key": "fkey",
}


def make_env(cfg):
    return ObservationBabaEnv(make_single_env(cfg["env_id"], cfg), cfg)


def make_single_env(env_id, cfg):
    import baba

    env = baba.make(env_id)
    env._ppo_env_id = env_id
    env.max_steps = cfg["max_episode_steps"]
    return env


def train_env_ids(cfg):
    return cfg.get("env_ids") or cfg.get("train_env_ids") or [cfg["env_id"]]


def make_train_env(cfg, warmup_episodes=0):
    env_ids = train_env_ids(cfg)
    if isinstance(env_ids, str):
        env_ids = [env_ids]
    if len(env_ids) == 1:
        return ShapedBabaEnv(make_single_env(env_ids[0], cfg), cfg, warmup_episodes)
    return MultiTaskBabaEnv(env_ids, cfg, warmup_episodes)


def policy_action_count(env, cfg):
    return env.action_space.n if cfg.get("use_idle", False) else env.action_space.n - 1


def to_env_action(policy_action, cfg):
    return int(policy_action) if cfg.get("use_idle", False) else int(policy_action) + 1


def use_coord_channels(cfg):
    return bool(cfg.get("use_coord_channels", False))


def coord_observation_space(observation_space, cfg):
    if not use_coord_channels(cfg):
        return observation_space
    height, width, channels = observation_space.shape
    return spaces.Box(
        low=0,
        high=255,
        shape=(height, width, channels + 2),
        dtype=np.uint8,
    )


def coord_channels(shape):
    height, width = shape[:2]
    x = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    y = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    return np.stack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
        ),
        axis=-1,
    )


def augment_obs(obs, cfg):
    if not use_coord_channels(cfg):
        return obs
    return np.concatenate((obs, coord_channels(obs.shape)), axis=-1)


class ObservationBabaEnv:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.observation_space = coord_observation_space(env.observation_space, cfg)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return augment_obs(self.env.reset(), self.cfg)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return augment_obs(obs, self.cfg), reward, done, info


class ShapedBabaEnv:
    def __init__(self, env, cfg, warmup_episodes=0):
        self.env = env
        self.cfg = cfg
        self.warmup_episodes = warmup_episodes
        self.episode_idx = 0
        self.episode_return = 0.0
        self.rule_reward_given = False
        self.rule_dead_penalty_given = False
        self.observation_space = coord_observation_space(env.observation_space, cfg)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        if self.episode_idx < self.warmup_episodes:
            np.random.seed(self.cfg.get("warmup_seed", self.cfg["seed"]))
        obs = self.env.reset()
        self.episode_return = 0.0
        self.rule_reward_given = self._is_win_rule_active()
        self.rule_dead_penalty_given = False
        self.episode_idx += 1
        return augment_obs(obs, self.cfg)

    def step(self, action):
        hit_boundary = self._hits_boundary(action)
        pushed_before = self._pushed_block_before(action)
        obs, env_reward, done, info = self.env.step(action)
        win_bonus = 1.0 if env_reward > 0 else 0.0
        reward = win_bonus - (1.0 / self.cfg["max_episode_steps"])
        stuck_push = self._is_stuck_push(pushed_before)
        shaped = self.cfg.get("use_shaped_reward", True)

        rule_assembled = not self.rule_reward_given and self._is_win_rule_active()
        if rule_assembled:
            self.rule_reward_given = True
            if shaped:
                reward += self.cfg.get("rule_assembly_reward", 0)

        rule_dead = (
            not self.rule_reward_given
            and not self.rule_dead_penalty_given
            and self._is_target_rule_dead()
        )
        if rule_dead:
            self.rule_dead_penalty_given = True

        if self.cfg.get("use_stuck_push_penalty", True):
            reward += stuck_push * self.cfg.get("stuck_push_penalty", 0)

        if shaped:
            reward += hit_boundary * self.cfg.get("boundary_penalty", 0)
            reward += rule_dead * self.cfg.get("rule_dead_penalty", 0)

        self.episode_return += reward
        info = dict(info)
        info.update(
            env_reward=env_reward,
            env_success=env_reward > 0,
            episode_return=self.episode_return if done else None,
            rule_assembled=rule_assembled,
            hit_boundary=hit_boundary,
            stuck_push=stuck_push,
            rule_dead=rule_dead,
        )
        return augment_obs(obs, self.cfg), reward, done, info

    def _action_direction(self, action):
        actions = self.env.actions
        return {
            actions.up: DIRS[0],
            actions.right: DIRS[1],
            actions.down: DIRS[2],
            actions.left: DIRS[3],
        }.get(action)

    def _hits_boundary(self, action):
        direction = self._action_direction(action)
        if direction is None:
            return False
        x, y = np.array(self.env.agent_pos) + direction
        return x <= 0 or y <= 0 or x >= self.env.width - 1 or y >= self.env.height - 1

    def _pushed_block_before(self, action):
        direction = self._action_direction(action)
        if direction is None:
            return None
        pos = tuple(np.array(self.env.agent_pos) + direction)
        if not self._inside_grid(pos):
            return None
        cell = self.env.grid.get(*pos)
        return (cell, pos) if cell is not None and cell.is_push() else None

    def _is_stuck_push(self, pushed_before):
        if pushed_before is None:
            return False
        block, old_pos = pushed_before
        for y in range(self.env.height):
            for x in range(self.env.width):
                if block in self.env.grid.get(x, y, z="all"):
                    return (x, y) == old_pos
        return False

    def _is_win_rule_active(self):
        win_obj = getattr(self.env, "win_obj", None)
        if win_obj is None:
            return False
        color, win_obj = win_obj if isinstance(win_obj, tuple) else (None, win_obj)
        win_obj = CANONICAL_RULE_OBJECTS.get(win_obj, win_obj)
        goal_rules = self.env.get_ruleset().get("is_goal", {})
        return bool(goal_rules.get(win_obj, False)) and (
            color is None or color in goal_rules.get(f"{win_obj}_color", [])
        )

    def _is_target_rule_dead(self):
        blocks = self._target_rule_blocks()
        if not all(blocks.values()):
            return False
        return any(self._is_dead_rule_position(pos, role) for role, positions in blocks.items() for pos in positions)

    def _target_rule_blocks(self):
        win_obj = getattr(self.env, "win_obj", None)
        if isinstance(win_obj, tuple):
            _, win_obj = win_obj
        win_obj = CANONICAL_RULE_OBJECTS.get(win_obj, win_obj)

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

    def _is_dead_rule_position(self, pos, role):
        x, y = pos
        left, right = 1, self.env.width - 2
        top, bottom = 1, self.env.height - 2
        if self._is_configured_dead_rule_position(pos, role):
            return True
        can_be_rule_slot = self._can_be_valid_rule_slot(pos, role)
        on_border_or_corner = x in {left, right} or y in {top, bottom}
        return (not self._has_legal_push(pos) or on_border_or_corner) and not can_be_rule_slot

    def _is_configured_dead_rule_position(self, pos, role):
        configured = self.cfg.get("dead_rule_positions", {})
        env_id = getattr(self.env, "_ppo_env_id", self.cfg.get("env_id"))
        positions_by_role = (
            configured.get(env_id)
            or configured.get(env_id.split("#", 1)[0])
            or configured.get("*", {})
        )
        positions = positions_by_role.get(role, []) + positions_by_role.get("any", [])
        return tuple(pos) in {tuple(item) for item in positions}

    def _can_be_valid_rule_slot(self, pos, role):
        x, y = pos
        offset = {"object": 0, "is": 1, "win": 2}[role]
        horizontal_start = x - offset
        vertical_start = y - offset
        return (
            1 <= horizontal_start and horizontal_start + 2 <= self.env.width - 2
        ) or (
            1 <= vertical_start and vertical_start + 2 <= self.env.height - 2
        )

    def _has_legal_push(self, pos):
        pos = np.array(pos)
        for direction in DIRS:
            stand_pos = pos - direction
            dest_pos = pos + direction
            if self._is_playable_pos(stand_pos) and self._is_playable_pos(dest_pos):
                if not self._cell_blocks(dest_pos) and not self._cell_blocks(stand_pos):
                    return True
        return False

    def _is_playable_pos(self, pos):
        x, y = pos
        return 1 <= x <= self.env.width - 2 and 1 <= y <= self.env.height - 2

    def _cell_blocks(self, pos):
        cell = self.env.grid.get(*pos)
        return cell is not None and not cell.can_overlap()

    def _is_you_rule_is_position(self, pos):
        x, y = pos
        for obj_pos, prop_pos in [((x - 1, y), (x + 1, y)), ((x, y - 1), (x, y + 1))]:
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


class MultiTaskBabaEnv:
    def __init__(self, env_ids, cfg, warmup_episodes=0):
        self.env_ids = list(env_ids)
        self.envs = [
            ShapedBabaEnv(make_single_env(env_id, cfg), cfg, warmup_episodes)
            for env_id in self.env_ids
        ]
        self.current_idx = 0
        self.current_env = self.envs[0]

    def __getattr__(self, name):
        return getattr(self.current_env, name)

    @property
    def active_env_id(self):
        return self.env_ids[self.current_idx]

    def reset(self):
        self.current_idx = int(np.random.randint(len(self.envs)))
        self.current_env = self.envs[self.current_idx]
        return self.current_env.reset()

    def step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        info = dict(info)
        info["env_id"] = self.active_env_id
        return obs, reward, done, info
