from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from buffer import ExtendedBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3 import SAC, DDPG, PPO

from copy import deepcopy
from general_utils import *
import global_config
from env_wrappers import RescaleAction

SelfSafeSAC = TypeVar("SelfSafeSAC", bound="SafeSAC")


class SafeSAC(SAC):

    def __init__(self, policy: str | MlpPolicy, 
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 3e-4,
                 buffer_size: int = 1000000, 
                 learning_starts: int = 100, 
                 batch_size: int = 256, 
                 tau: float = 0.005, 
                 gamma: float = 0.99, 
                 train_freq: int | Tuple[int | str] = 1, 
                 gradient_steps: int = 1, 
                 action_noise: ActionNoise | None = None, 
                 replay_buffer_class: ReplayBuffer | None = ExtendedBuffer,
                replay_buffer_kwargs: Dict[str, Any] | None = None, 
                optimize_memory_usage: bool = False, 
                ent_coef: str | float = "auto", 
                target_update_interval: int = 1, 
                target_entropy: str | float = "auto", 
                use_sde: bool = False, 
                sde_sample_freq: int = -1, 
                use_sde_at_warmup: bool = False, 
                stats_window_size: int = 100, 
                tensorboard_log: str | None = None, 
                policy_kwargs: Dict[str, Any] | None = None, 
                verbose: int = 0, 
                seed: int | None = None, 
                device: th.device | str = "auto", 
                _init_setup_model: bool = True,
                undelayed_critic: ContinuousCritic = None, 
                avg : bool = True):
        
        

        super().__init__(policy, 
                         env, 
                         learning_rate, 
                         buffer_size, 
                         learning_starts, 
                         batch_size, 
                         tau, 
                         gamma, 
                         train_freq, 
                         gradient_steps, 
                         action_noise, 
                         replay_buffer_class,
                         replay_buffer_kwargs, 
                         optimize_memory_usage,
                         ent_coef, 
                         target_update_interval, 
                         target_entropy, 
                         use_sde, 
                         sde_sample_freq, 
                         use_sde_at_warmup, 
                         stats_window_size, 
                         tensorboard_log, 
                         policy_kwargs, 
                         verbose, 
                         seed, 
                         device, 
                         _init_setup_model)

        self.num_delays = global_config.NUM_DELAYS
        if env is not None:
            self.env_model = deepcopy(env.env.env)
            self.max_delay = env.known_delay
        else : 
            self.max_delay = global_config.MAX_DELAY
        self.undelayed_critic = undelayed_critic
        
        self.avg = avg
        print(f"Replay buffer class {replay_buffer_class}")

    def _store_transition(self, 
                          replay_buffer: ExtendedBuffer, 
                          buffer_action: np.ndarray, 
                          new_obs: np.ndarray | Dict[str, np.ndarray], 
                          reward: np.ndarray, 
                          dones: np.ndarray, 
                          infos: List[Dict[str, Any]]) -> None:


        base_state = self._last_obs[0, :self.env_model.observation_space.shape[0]]
        action_queue = self._last_obs[0, self.env_model.observation_space.shape[0]:]
        # print(f"Last obs {self._last_obs.shape}, obs space {self.env_model.observation_space.shape[0]} Action queue is {action_queue}")

        # Sample some values of delay
        taus = np.random.randint(0, self.max_delay + 1, size = self.num_delays)
        # Predict the delayed states
        delayed_states = np.array([predict_state(env_model=self.env_model, cur_state=base_state, actions_queue=action_queue, delay = tau) for tau in taus])
        # Predict the next states
        # next_delayed_states = np.array([predict_next_state(env_model=self.env_model, cur_state=delayed_state, action = buffer_action) for delayed_state in delayed_states])
        next_delayed_states = np.zeros(self.replay_buffer.pred_next_observations.shape[1:])

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            delayed_states,  # type: ignore[arg-type]
            next_obs, 
            next_delayed_states,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:

        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Modified critic loss - the target q values now come from the undelayed critic
            with th.no_grad():
                # print(f"Obs: {replay_data.pred_observations[:, 0, :].shape} and actions {replay_data.actions.shape}")
                Qprime = th.tensor(np.array([self.undelayed_critic(replay_data.pred_observations[:, idx, :], replay_data.actions) for idx in range(replay_data.pred_observations.shape[1])]))
                if self.avg:
                    target_q_values = th.mean(th.min(Qprime, axis = 1)[0], axis = 0)
                else :
                    target_q_values = th.min(th.min(Qprime, axis = 1)[0], axis = 0)[0]

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss =  0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))