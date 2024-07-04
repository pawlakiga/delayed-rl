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
from stable_baselines3.common.vec_env import VecEnv


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from copy import deepcopy
from general_utils import *
import global_config

from env_wrappers import *


SelfDelayedSAC = TypeVar("SelfDelayedSAC", bound="DelayedSAC")


class DelayedSAC(SAC):

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
                 replay_buffer_class: ReplayBuffer | None = ReplayBuffer,
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
                delay : int = 0):
        
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

        self.env_model = deepcopy(env.env.env)
        self.delay = delay
        self.original_env = env

    def _store_transition(self, 
                          replay_buffer: ReplayBuffer, 
                          buffer_action: np.ndarray, 
                          new_obs: np.ndarray | Dict[str, np.ndarray], 
                          reward: np.ndarray, 
                          dones: np.ndarray, 
                          infos: List[Dict[str, Any]]) -> None:
    
        
        action_queue = infos[0]['queue']
        # Reshape if needed
        action_size = self.env_model.action_space.shape[0]
        # Reshape queue if needed
        if len(np.array(action_queue).shape) == 1 and action_size > 1: 
            action_queue = np.array(action_queue).reshape(np.array(action_queue).shape[0]//action_size, action_size)
        else : 
            action_queue = np.array(action_queue).reshape(len(action_queue), action_size)
        action_queue = np.concatenate([np.array(action_queue), buffer_action.reshape(1, action_size)], axis = 0)
        # Get the delayed action
        delayed_action = action_queue[-self.delay - 1]
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # print(f"Agent: Sample s {self._last_original_obs}, queue {infos[0]['queue']}, a {delayed_action}, new_obs {next_obs}")
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
            next_obs, 
            delayed_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def predict(self, state, deterministic = False): 
        # print(f"Predicting from {state} and queue {self.original_env.actions_queue}")
        pred_state = predict_state(self.env_model, state, self.original_env.actions_queue, delay = self.delay, timestep=0)
        # print(f"Predicted state {pred_state} from {state} and queue {self.original_env.actions_queue}")
        pred_state.reshape(1, self.env.observation_space.shape[0])
        return super().predict(np.array(pred_state), deterministic = deterministic)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)