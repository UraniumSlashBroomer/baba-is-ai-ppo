import argparse
import json

from config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val", "sample", "test"], required=True)
    parser.add_argument("--config", default="ppo/configs/make_win.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--render-fps", type=float, default=0)
    parser.add_argument("--override", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.override:
        cfg.update(json.loads(args.override))

    checkpoint = args.checkpoint
    if checkpoint is None and args.mode in {"val", "sample"}:
        from paths import ROOT

        checkpoint = ROOT / "ppo" / "checkpoints" / cfg["name"] / "latest.pt"

    if args.mode == "train":
        from train import train

        render_fps = args.render_fps or (cfg.get("train_render_fps", 0) if args.visualize else 0)
        train(cfg, render_fps, args.checkpoint)
    elif args.mode == "val":
        from evaluate import val

        val(cfg, checkpoint, args.visualize or args.render_fps > 0, args.render_fps)
    elif args.mode == "test":
        from manual_test import test_reward_manually

        test_reward_manually(cfg, args.render_fps or cfg.get("eval_render_fps", 30))
    else:
        from evaluate import sample

        sample(cfg, checkpoint, args.visualize)


if __name__ == "__main__":
    main()
