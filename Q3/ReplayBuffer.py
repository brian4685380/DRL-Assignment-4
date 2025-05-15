import numpy as np
import torch
from collections import deque
# ------------------------------------------------------------
#  Utilities
# ------------------------------------------------------------
class ReplayBuffer:
  """A simple FIFO experience replay buffer for SAC."""

  def __init__(self, obs_dim: int, act_dim: int, size: int = int(2e6),
               batch_size: int = 1024):
    self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
    self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)

    self.max_size = size
    self.batch_size = batch_size
    self.ptr = 0
    self.size = 0

  def store(self, obs, act, rew, next_obs, done):
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.next_obs_buf[self.ptr] = next_obs
    self.done_buf[self.ptr] = done

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample_batch(self, device):
    idxs = np.random.randint(0, self.size, size=self.batch_size)

    batch = dict(obs=self.obs_buf[idxs],
                 act=self.act_buf[idxs],
                 rew=self.rew_buf[idxs],
                 next_obs=self.next_obs_buf[idxs],
                 done=self.done_buf[idxs])

    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}

  def __len__(self):
    return self.size