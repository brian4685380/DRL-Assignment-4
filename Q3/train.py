import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

from ReplayBuffer import ReplayBuffer
from networks import Actor, Critic

# ------------------------------------------------------------
#  Soft Actor‑Critic Agent (training + acting)
# ------------------------------------------------------------
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dmc import make_dmc_env

def make_env():
    # Create environment with state observations
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True

class SAC:
    def __init__(self):
        self.env = make_env()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic1 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic1_target = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_loss_fn = nn.MSELoss()
        self.target_entropy = -self.act_dim
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True)
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, BUFFER_SIZE, BATCH_SIZE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=LR)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=LR)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=LR)
    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def train(self):
        best_avg_reward = -np.inf
        for episode in range(NUM_EPISODES):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                if self.replay_buffer.size > START_STEPS:
                    with torch.no_grad():
                        action, _ = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                        action = action.detach().cpu().numpy().squeeze(0)
                else:
                    action = np.random.uniform(-1.0, 1.0, size=self.act_dim)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.store(obs, action, reward, next_obs, done)
                ep_reward += reward
                obs = next_obs

                if self.replay_buffer.size > START_STEPS:
                    batch = self.replay_buffer.sample_batch(self.device)
                    batch_obs = batch["obs"]
                    batch_action = batch["act"]
                    batch_next_obs = batch["next_obs"]
                    
                    batch_reward = batch["rew"].unsqueeze(1)  # (B,) -> (B,1)
                    batch_dones = batch["done"].unsqueeze(1)  # (B,) -> (B,1)

                    q1 = self.critic1(batch_obs, batch_action)        # should output (B,1)
                    q2 = self.critic2(batch_obs, batch_action)

                    with torch.no_grad():
                        next_action, log_prob = self.actor(batch_next_obs, deterministic=False, with_logprob=True)
                        log_prob = log_prob.unsqueeze(1)  # (B,) -> (B,1)
                        q1_next = self.critic1_target(batch_next_obs, next_action)
                        q2_next = self.critic2_target(batch_next_obs, next_action)
                        q_next = torch.min(q1_next, q2_next)  # (B,1)
                        q_target = batch_reward + GAMMA * (q_next - self.log_alpha.exp() * log_prob)

                    critic1_loss = F.mse_loss(q1, q_target)
                    critic2_loss = F.mse_loss(q2, q_target)
                    self.critic1_optim.zero_grad()
                    self.critic2_optim.zero_grad()
                    critic1_loss.backward()
                    critic2_loss.backward()
                    self.critic1_optim.step()
                    self.critic2_optim.step()

                    freeze(self.critic1)
                    freeze(self.critic2)

                    # --- Actor & alpha updates ---
                    action_pi, log_prob_pi = self.actor(batch_obs, deterministic=False, with_logprob=True)
                    q1_pi = self.critic1(batch_obs, action_pi)
                    q2_pi = self.critic2(batch_obs, action_pi)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (torch.exp(self.log_alpha).detach() * log_prob_pi - q_pi).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    alpha_loss = -(torch.exp(self.log_alpha) * (log_prob_pi + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                    unfreeze(self.critic1)
                    unfreeze(self.critic2)
                    # --- Target networks update ---
                    self.soft_update(self.critic1_target, self.critic1, TAU)
                    self.soft_update(self.critic2_target, self.critic2, TAU)
            print(f"Episode {episode}, Reward: {ep_reward:.2f}")
            if episode % EVAL_EVERY == 0:
                avg, std, final_score = self.eval(10 if episode < 1000 else 100)
                if avg > best_avg_reward:
                    best_avg_reward = avg
                    torch.save(self.actor.state_dict(), f"best_actor.pth")
                    torch.save(self.critic1.state_dict(), f"best_critic1.pth")
                    torch.save(self.critic2.state_dict(), f"best_critic2.pth")
                    torch.save(self.critic1_target.state_dict(), f"best_critic1_target.pth")
                    torch.save(self.critic2_target.state_dict(), f"best_critic2_target.pth")
                    torch.save(self.log_alpha, f"best_log_alpha.pth")
                if final_score > 460:
                    print(f"reached baseline!!!")
    def eval(self, num_episodes=100):
        ep_rewards = []
        for _ in range(num_episodes):
            sim_env = make_env()
            obs, _ = sim_env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, _ = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                action = action.detach().cpu().numpy().squeeze(0)
                next_obs, reward, terminated, truncated, _ = sim_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                obs = next_obs
            ep_rewards.append(ep_reward)
        ep_rewards = np.array(ep_rewards)
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        final_score = avg_reward - std_reward
        print(f"Eval: Avg Reward: {avg_reward}, Std Reward: {std_reward}, Final Score: {final_score}")
        return avg_reward, std_reward, final_score
    
if __name__ == "__main__":
    sac = SAC()
    sac.train()