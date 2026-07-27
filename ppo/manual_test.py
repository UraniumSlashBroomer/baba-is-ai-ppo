from env import ShapedBabaEnv, make_single_env
from utils import set_seed


def _action_name(env, action):
    try:
        return env.actions(action).name
    except ValueError:
        return str(action)


def _format_reward_details(info):
    parts = []
    env_reward = info.get("env_reward")
    if env_reward is not None:
        parts.append(f"env={env_reward:.3f}")
    for key, label in (
        ("rule_assembled", "rule"),
        ("hit_boundary", "boundary"),
        ("stuck_push", "stuck"),
        ("rule_dead", "dead_rule"),
    ):
        if info.get(key):
            parts.append(label)
    return " | ".join(parts)


def _frame_with_panel(frame, panel_height, stats):
    import cv2
    import numpy as np

    panel = np.full((panel_height, frame.shape[1], 3), (22, 24, 28), dtype=np.uint8)

    text_lines = [
        f"reward={stats['reward']:.3f}  return={stats['return']:.3f}  step={stats['step']}",
        f"action={stats['action']}  done={stats['done']}",
    ]
    if stats["details"]:
        text_lines.append(stats["details"])

    for idx, line in enumerate(text_lines):
        cv2.putText(
            panel,
            line,
            (10, 22 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    return np.vstack((frame, panel))


def _draw(pygame, screen, frame, panel_height, stats):
    frame = _frame_with_panel(frame, panel_height, stats)
    frame_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()


def test_reward_manually(cfg, fps=30):
    import pygame

    set_seed(cfg["seed"])
    env_id = cfg.get("test_env_id", cfg.get("eval_env_id", cfg["env_id"]))
    env = ShapedBabaEnv(make_single_env(env_id, cfg), cfg)

    pygame.init()
    pygame.display.set_caption(f"reward test: {env_id}")

    action_by_key = {
        pygame.K_UP: env.actions.up,
        pygame.K_RIGHT: env.actions.right,
        pygame.K_DOWN: env.actions.down,
        pygame.K_LEFT: env.actions.left,
        pygame.K_SPACE: env.actions.idle,
        pygame.K_0: env.actions.idle,
    }

    obs = env.reset()
    frame = env.render(mode="rgb_array")
    panel_height = 84
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0] + panel_height))
    clock = pygame.time.Clock()

    stats = {
        "reward": 0.0,
        "return": 0.0,
        "step": 0,
        "action": "none",
        "done": False,
        "details": "",
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in {pygame.K_ESCAPE, pygame.K_q}:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    stats.update(
                        {
                            "reward": 0.0,
                            "return": 0.0,
                            "step": 0,
                            "action": "reset",
                            "done": False,
                            "details": "",
                        }
                    )
                elif event.key in action_by_key and not stats["done"]:
                    action = action_by_key[event.key]
                    obs, reward, done, info = env.step(action)
                    stats["reward"] = float(reward)
                    stats["return"] += float(reward)
                    stats["step"] += 1
                    stats["action"] = f"{_action_name(env, action)} ({int(action)})"
                    stats["done"] = bool(done)
                    stats["details"] = _format_reward_details(info)

        if obs is not None:
            frame = env.render(mode="rgb_array")
            _draw(pygame, screen, frame, panel_height, stats)
        clock.tick(fps)

    env.close()
    pygame.quit()
