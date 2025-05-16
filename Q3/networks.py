import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)   # <- fix here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
    
    def forward(self, obs, deterministic=False, with_logprob=False):
        mu = self.mu(F.relu(self.fc2(F.relu(self.fc1(obs)))))
        if deterministic:
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_std = self.log_std(F.relu(self.fc2(F.relu(self.fc1(obs))))) 
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            x_t = dist.rsample()
            if with_logprob:
                log_prob = dist.log_prob(x_t).sum(axis=-1)
                log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(axis=-1)
            else:
                log_prob = None
            action = torch.tanh(x_t)
        return action, log_prob