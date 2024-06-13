import random
from collections import deque
from gymnasium import Space
from matplotlib import pyplot as plt
import numpy as np
# from sympy import EmptySequence
import torch 
import torch.nn as nn
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import torch.nn.functional as F 
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

import global_config


ExtendedExperience = namedtuple('ExtendedExperience', ['aug_state', 'delayed_states', 'action', 'reward', 'next_aug_states', 'next_delayed_states', 'done'])

class ExtendedBufferSamples(NamedTuple):
    observations: th.Tensor
    pred_observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    pred_next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class ExtendedBuffer(ReplayBuffer): 

    """ Replay buffer extending the ReplayBuffer from stable baselines for the safe agent that stores transitions with the following additional fields:"""
    pred_observations: np.ndarray
    pred_next_observations: np.ndarray

    def __init__(self, 
                 buffer_size: int, 
                 observation_space: Space, 
                 action_space: Space, 
                 device: torch.device | str = "auto", 
                 n_envs: int = 1, 
                 optimize_memory_usage: bool = False, 
                 handle_timeout_termination: bool = True):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        num_delays = global_config.NUM_DELAYS
        max_delay = global_config.MAX_DELAY

        self.pred_observations = np.zeros((self.buffer_size, self.n_envs, num_delays, self.obs_shape[0] - max_delay * self.action_dim), dtype=observation_space.dtype)
        self.pred_next_observations = np.zeros((self.buffer_size, self.n_envs, num_delays, self.obs_shape[0] - max_delay * self.action_dim), dtype=observation_space.dtype)


    def add(
        self,
        obs: np.ndarray,
        pred_obs : np.ndarray,
        next_obs: np.ndarray,
        pred_next_obs : np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.pred_observations[self.pos] = np.array(pred_obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
            self.pred_observations[(self.pos + 1) % self.buffer_size] = np.array(pred_next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)
            self.pred_next_observations[self.pos] = np.array(pred_next_obs)


        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ExtendedBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            pred_next_obs = self._normalize_obs(self.pred_observations[(batch_inds + 1) % self.buffer_size, env_indices, :,:], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            pred_next_obs = self._normalize_obs(self.pred_next_observations[batch_inds, env_indices, :,:], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self._normalize_obs(self.pred_observations[batch_inds, env_indices, :,:], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            pred_next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ExtendedBufferSamples(*tuple(map(self.to_torch, data)))
    


TransformerExperience = namedtuple('TransformerExperience', 
                                    ['observation', 'action', 'reward',
                                    'next_observation', 'next_action', 'done', 
                                    'episode_length'])


class TransformerBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    next_actions: th.Tensor
    dones: th.Tensor
    context_lengths: th.Tensor


class TransformerBuffer(ReplayBuffer):

    next_actions : np.ndarray
    context_lengths : np.ndarray

    def __init__(self, 
                 buffer_size: int, 
                 observation_space: Space, 
                 action_space: Space, 
                 device: torch.device | str = "auto", 
                 n_envs: int = 1, 
                 optimize_memory_usage: bool = False, 
                 handle_timeout_termination: bool = True, 
                 obs_mask : int = -10, 
                 max_episode_steps : int = 120, 
                 context_len : Optional[int] = global_config.MAX_DELAY):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.next_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.context_lengths = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

        self.context_len = context_len
        self.obs_mask = obs_mask

