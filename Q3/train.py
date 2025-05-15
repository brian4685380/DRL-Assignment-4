import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ReplayBuffer import ReplayBuffer
from networks import Actor, QNetwork

# ------------------------------------------------------------
#  Soft Actor‑Critic Agent (training + acting)
# ------------------------------------------------------------


class Agent:
    """Soft Actor‑Critic agent for DMC Humanoid‑Walk.

    Note: Only the 'act' method and the '__init__' signature must remain
    unchanged for the evaluation harness. All training utilities live inside
    this class, so you can simply create an instance and call the 'train'
    method before evaluation.
    """

    def __init__(self,
                 obs_dim: int = 67,
                 act_dim: int = 21,
                 device: str | torch.device | None = None,
                 seed: int | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # --- public API expected by autograder ---
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float64)

        # --- reproducibility ---
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # --- SAC hyper‑parameters ---
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.batch_size = 1024
        self.start_steps = 10_000          # purely random steps before using the policy
        self.update_after = 5_000          # how many env steps to collect before updates
        self.update_every = 1              # env steps per network update
        # delayed policy updates (1 = every iteration)
        self.policy_freq = 1

        # entropy coefficient (auto‑tuned)
        self.target_entropy = -act_dim
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=self.device,
                                      requires_grad=True)

        # --- networks ---
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim).to(self.device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # --- optimisers ---
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # --- replay buffer ---
        self.replay = ReplayBuffer(obs_dim, act_dim, size=int(2e6),
                                   batch_size=self.batch_size)

        # --- bookkeeping ---
        self.total_env_steps = 0

    # ------------------------------------------------------------------
    #  Acting API (used by the grader). During evaluation the agent
    #  will be *already trained*, so we switch to deterministic actions.
    # ------------------------------------------------------------------
    def act(self, observation):
        """Select an action given the current observation.

        This method is noise‑free after 'self.training' is set to False by the
        'evaluate' helper below.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            if getattr(self, 'deterministic', False):
                mu, _ = self.actor._forward(obs)
                action = torch.tanh(mu)
            else:
                action, _, _ = self.actor.sample(obs)
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------
    #  Public helper to enable deterministic evaluation.
    # ------------------------------------------------------------------
    def eval_mode(self):
        self.actor.eval()
        self.deterministic = True

    # ------------------------------------------------------------------
    #  Training utilities (not required by autograder, but convenient).
    # ------------------------------------------------------------------
    def _update_critic(self, batch):
        obs, act, rew, next_obs, done = (batch['obs'], batch['act'], batch['rew'],
                                         batch['next_obs'], batch['done'])
        with torch.no_grad():
            next_act, next_logp, _ = self.actor.sample(next_obs)
            target_q1 = self.q1_target(next_obs, next_act)
            target_q2 = self.q2_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2) - \
                torch.exp(self.log_alpha) * next_logp
            target = rew.unsqueeze(-1) + (1.0 -
                                          done.unsqueeze(-1)) * self.gamma * target_q

        # Q1 update
        q1 = self.q1(obs, act).unsqueeze(-1)
        q1_loss = F.mse_loss(q1, target)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        # Q2 update
        q2 = self.q2(obs, act).unsqueeze(-1)
        q2_loss = F.mse_loss(q2, target)
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        return q1_loss.item(), q2_loss.item()

    def _update_actor_and_alpha(self, batch):
        obs = batch['obs']
        act_new, logp_new, _ = self.actor.sample(obs)
        q1_new = self.q1(obs, act_new)
        q2_new = self.q2(obs, act_new)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss
        actor_loss = (torch.exp(self.log_alpha) * logp_new - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature / entropy coefficient loss
        alpha_loss = - (self.log_alpha * (logp_new +
                        self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return actor_loss.item(), alpha_loss.item(), torch.exp(self.log_alpha).item()

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)

    # ------------------------------------------------------------------
    #  Main training loop. Collects experience using the current policy
    #  (with initial random exploration), updates networks, and periodically
    #  evaluates the policy to track progress.
    # ------------------------------------------------------------------
    def train(self, make_env_fn, total_steps: int = 3_000_000,
              eval_interval: int = 100_000, num_eval_eps: int = 5):
        env = make_env_fn()
        eval_env = make_env_fn()  # separate env for evaluation to avoid cueing

        obs, _ = env.reset()
        episode_return = 0.0
        episode_len = 0

        while self.total_env_steps < total_steps:
            # -------------------- collect data --------------------
            if self.total_env_steps < self.start_steps:
                action = env.action_space.sample()
            else:
                action = self.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay.store(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_return += reward
            episode_len += 1
            self.total_env_steps += 1

            # Episode done – reset env
            if done:
                obs, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

            # -------------------- update --------------------
            if (self.total_env_steps > self.update_after and
                    self.total_env_steps % self.update_every == 0):
                batch = self.replay.sample_batch(self.device)
                # rename keys to shorter handles for convenience
                batch = {k[:3] if k.startswith(
                    'obs') else k[:4]: v for k, v in batch.items()}
                q1_l, q2_l = self._update_critic(batch)
                if self.total_env_steps % self.policy_freq == 0:
                    a_l, al_l, alpha_val = self._update_actor_and_alpha(batch)
                    # soft target network updates
                    self._soft_update(self.q1, self.q1_target)
                    self._soft_update(self.q2, self.q2_target)

            # -------------------- evaluate --------------------
            if self.total_env_steps % eval_interval == 0:
                avg_ret = self.evaluate(eval_env, num_eval_eps)
                print(f"Step: {self.total_env_steps:7d} | AvgReturn: {avg_ret:9.2f} | "
                      f"Replay: {len(self.replay):7d}")

        env.close()
        eval_env.close()

    # ------------------------------------------------------------------
    #  Helper for deterministic evaluation (no exploration noise).
    # ------------------------------------------------------------------
    def evaluate(self, env, num_episodes: int = 5):
        self.eval_mode()
        returns = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action = self.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += reward
            returns.append(ep_return)
        self.deterministic = False  # switch back to exploration mode
        return float(np.mean(returns))


if __name__ == '__main__':
    # Example usage:
    def make_env_fn():
        return gym.make('Humanoid-v4', render_mode='human')

    agent = Agent()
    agent.train(make_env_fn)