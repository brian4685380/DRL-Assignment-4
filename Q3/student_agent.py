import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dmc import make_dmc_env



def make_env():
    # Create environment with state observations
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

# Do not modify the input of the 'act' function and the '__init__' function. 
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
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(67, 21).to(self.device)
        self.actor.load_state_dict(torch.load("best_actor.pth", map_location=self.device))

    def act(self, observation):
        action, _ = self.actor(torch.tensor(observation, dtype=torch.float32, device = self.device).unsqueeze(0))
        action = action.detach().cpu().numpy().squeeze(0)
        return action
    

def record_video(env, agent):
    import imageio
    gif_path = f'./demo.gif'

    state, info = env.reset()
    frames = []

    while True:
        frame = env.render()
        frames.append(np.array(frame))
        action = agent.act(state)
        next_state, reward, terminated, truncated, _= env.step(action)
        state = next_state

        if terminated or truncated:
            break

    imageio.mimsave(gif_path, frames, fps=30)
    print(f'GIF saved to {gif_path}')

if __name__ == "__main__":
    env = make_env()
    agent = Agent()
    record_video(env, agent)
