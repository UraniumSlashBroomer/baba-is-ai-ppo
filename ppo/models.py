import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(input_dim, hidden_size, layers=2):
    blocks = []
    for i in range(layers):
        blocks += [nn.Linear(input_dim if i == 0 else hidden_size, hidden_size), nn.ReLU()]
    return nn.Sequential(*blocks)


def cnn_features(obs_shape):
    in_channels = obs_shape[-1]
    net = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, obs_shape[0], obs_shape[1])
        out_dim = net(dummy).shape[1]
    return net, out_dim


class PolicyMixin:
    recurrent = False

    def dist_value(self, obs, state=None):
        if self.recurrent:
            logits, value, next_state = self(obs, state)
            return Categorical(logits=logits), value, next_state
        logits, value = self(obs)
        return Categorical(logits=logits), value, None

    def act(self, obs, state=None):
        dist, value, next_state = self.dist_value(obs, state)
        action = dist.sample()
        result = (action, dist.log_prob(action), dist.entropy(), value)
        return result + (next_state,) if self.recurrent else result

    def evaluate(self, obs, actions, state=None):
        dist, value, _ = self.dist_value(obs, state)
        return dist.log_prob(actions), dist.entropy(), value


class CnnActorCritic(PolicyMixin, nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        self.actor_cnn, cnn_dim = cnn_features(obs_shape)
        self.critic_cnn, _ = cnn_features(obs_shape)
        self.actor_net = mlp(cnn_dim, hidden_size, layers=1)
        self.critic_net = mlp(cnn_dim, hidden_size, layers=1)
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        actor_h = self.actor_net(self.actor_cnn(x))
        critic_h = self.critic_net(self.critic_cnn(x))
        return self.actor_head(actor_h), self.critic_head(critic_h).squeeze(-1)


class RecurrentActorCritic(PolicyMixin, nn.Module):
    recurrent = True

    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        self.feature_net, feature_dim = cnn_features(obs_shape)
        self.actor_lstm = nn.LSTMCell(feature_dim, hidden_size)
        self.critic_lstm = nn.LSTMCell(feature_dim, hidden_size)
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def initial_state(self, batch_size, device):
        shape = (batch_size, self.hidden_size)
        return tuple(torch.zeros(shape, device=device) for _ in range(4))

    def forward(self, x, state):
        x = x.permute(0, 3, 1, 2)
        features = self.feature_net(x)
        actor_h, actor_c, critic_h, critic_c = state
        actor_h, actor_c = self.actor_lstm(features, (actor_h, actor_c))
        critic_h, critic_c = self.critic_lstm(features, (critic_h, critic_c))
        return (
            self.actor_head(actor_h),
            self.critic_head(critic_h).squeeze(-1),
            (actor_h, actor_c, critic_h, critic_c),
        )


class SplitMlpActorCritic(PolicyMixin, nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        self.actor_net = mlp(obs_dim, hidden_size)
        self.critic_net = mlp(obs_dim, hidden_size)
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.actor_head(self.actor_net(x)), self.critic_head(self.critic_net(x)).squeeze(-1)


class SharedActorCritic(PolicyMixin, nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        self.net = mlp(int(np.prod(obs_shape)), hidden_size)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.net(x.flatten(start_dim=1))
        return self.actor(h), self.critic(h).squeeze(-1)


def build_model(obs_shape, n_actions, cfg):
    model_cls = RecurrentActorCritic if cfg.get("model_type", "cnn") == "lstm" else CnnActorCritic
    return model_cls(obs_shape, n_actions, cfg["hidden_size"])


def is_recurrent_model(model):
    return getattr(model, "recurrent", False)


def detach_state(state):
    return tuple(x.detach() for x in state)


def state_to_numpy(state):
    return tuple(x.detach().cpu().numpy() for x in state)


def numpy_to_state(states, device):
    if not states:
        return None
    return tuple(
        torch.tensor(np.concatenate([state[i] for state in states], axis=0), dtype=torch.float32, device=device)
        for i in range(len(states[0]))
    )
