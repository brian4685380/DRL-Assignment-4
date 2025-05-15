import torch
import torch.nn as nn


# ------------------------------------------------------------
#  Neural network building blocks
# ------------------------------------------------------------
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

def mlp(in_dim: int, hidden_dims=(256, 256), out_dim: int | None = None):
  layers: list[nn.Module] = []
  last_dim = in_dim
  for h in hidden_dims:
    layers += [nn.Linear(last_dim, h), nn.ReLU()]
    last_dim = h
  if out_dim is not None:
    layers.append(nn.Linear(last_dim, out_dim))
  return nn.Sequential(*layers)


class Actor(nn.Module):
  """Gaussian policy with tanh-squashing."""

  def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
    super().__init__()
    self.net = mlp(obs_dim, hidden_dims)
    self.mu_layer = nn.Linear(hidden_dims[-1], act_dim)
    self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)

  def _forward(self, obs):
    h = self.net(obs)
    mu = self.mu_layer(h)
    log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)
    return mu, std

  def sample(self, obs):
    mu, std = self._forward(obs)
    normal = torch.distributions.Normal(mu, std)
    x_t = normal.rsample()               # reparameterisation trick
    y_t = torch.tanh(x_t)
    action = y_t
    # Enforcing action bounds inside log-prob calculation
    log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
    log_prob = log_prob.sum(-1, keepdim=True)
    mu_tanh = torch.tanh(mu)
    return action, log_prob, mu_tanh


class QNetwork(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
    super().__init__()
    self.q = mlp(obs_dim + act_dim, hidden_dims, 1)

  def forward(self, obs, act):
    q = self.q(torch.cat([obs, act], dim=-1))
    return q.squeeze(-1)
