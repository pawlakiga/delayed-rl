import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.policies import BaseModel

class ContinuousSeparateCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        # for idx in range(n_critics):
        #     q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
        #     q_net = nn.Sequential(*q_net_list)
        #     self.add_module(f"qf{idx}", q_net)
        #     self.q_networks.append(q_net)
        self.build_state_action_q_nets(features_dim, action_dim, net_arch, activation_fn)


    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        # qvalue_input = th.cat([features, actions], dim=1)
        # return tuple(q_net(qvalue_input) for q_net in self.q_networks)
        state_q_values = []
        action_q_values = []

        for idx in range(self.n_critics):
            state_q_values.append(self.state_q_nets[idx](features))
            action_q_values.append(self.action_q_nets[idx](actions))
        q_values = []
        for idx in range(self.n_critics):
            q_values.append(self.q_networks[idx](th.sum([state_q_values[idx], action_q_values[idx]], dim=1)))
        return tuple(q_values)
                        

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))
    
    def build_state_action_q_nets(self, features_dim: int, action_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> List[nn.Module]:
        state_q_nets = []
        action_q_nets = []
        for idx in range(self.n_critics):
            state_q_net_list = create_mlp(features_dim, 256, net_arch, activation_fn)
            state_q_net = nn.Sequential(*state_q_net_list)
            self.add_module(f"state_qf{idx}", state_q_net)
            state_q_nets.append(state_q_net)
            action_q_net_list = create_mlp(action_dim, 256, net_arch, activation_fn)
            action_q_net = nn.Sequential(*action_q_net_list)
            self.add_module(f"action_qf{idx}", action_q_net)
            action_q_nets.append(action_q_net)
        # Layer to add the action and state q values
        for idx in range(self.n_critics):
            q_net_list = create_mlp(256, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
        
