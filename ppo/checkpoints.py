import json

import torch

from models import (
    CnnActorCritic,
    RecurrentActorCritic,
    SharedActorCritic,
    SplitMlpActorCritic,
    build_model,
)


def save_checkpoint(model, optimizer, cfg, path, stats=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "stats": stats,
        },
        path,
    )
    if stats is not None:
        path.with_suffix(".json").write_text(
            json.dumps({"checkpoint": str(path), "config": cfg, "stats": stats}, indent=2)
        )


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def _checkpoint_model_class(state):
    if "actor_lstm.weight_ih" in state:
        return RecurrentActorCritic, "actor_head"
    if "actor_cnn.0.weight" in state:
        return CnnActorCritic, "actor_head"
    if "actor_head.weight" in state:
        return SplitMlpActorCritic, "actor_head"
    return SharedActorCritic, "actor"


def load_model_for_checkpoint(env, cfg, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)["model"]
    model_cls, head_name = _checkpoint_model_class(state)
    head = state[f"{head_name}.weight"]
    n_actions, hidden_size = head.shape

    cfg["use_idle"] = n_actions == env.action_space.n
    model = model_cls(env.observation_space.shape, n_actions, hidden_size).to(device)
    model.load_state_dict(state)
    return model
