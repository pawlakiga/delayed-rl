import gymnasium as gym

from stable_baselines3 import SAC, DDPG, PPO

from environment import *
from env_wrappers import *
from matplotlib import pyplot as plt
from general_utils import *
# from delay_model import *
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch
from wc_sac import *
from stable_baselines3.common.logger import Logger, configure
import datetime
import copy
from test_utils import *

from safe_sac import SafeSAC
from delayed_sac import DelayedSAC



# def train_default_sac(core_env : gym.Env = None,
#                       env_type : str = 'linear', 
#                       desired_state : float | list = 0.8, 
#                       n_episodes : int = 100, 
#                       ent_coef : float = 0.5, 
#                       seed : int = None, 
#                       save : bool = True):

#     if core_env is None:
#         core_env = init_core_env(env_type, desired_state, seed) 
#     env = RescaleAction(core_env)
#     model = SAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef)
#     today = datetime.date.today().strftime("%m%d")
#     model_dir = f"{global_config.MODELS_PATH}/{core_env.__class__.__name__}/Desired{desired_state}/sac"
#     log_dir = f"{global_config.LOG_PATH}/{core_env.__class__.__name__}/{today}/Desired{desired_state}/sac"
#     new_logger = configure(log_dir, ["stdout", "csv"])
#     model.set_logger(new_logger)
#     model.learn(total_timesteps = n_episodes * global_config.DEFAULT_PARAMS[core_env.__class__.__name__]['max_episode_len'])
#     if save:
#         model.save(model_dir)
#     return model, core_env

def train_default_sac(core_env : gym.Env = None,
                      env_type : str = 'linear', 
                      agent_type : str = 'sac',
                      desired_state : float | list = 0.8, 
                      n_episodes : int = 100, 
                      ent_coef : float = 0.5, 
                      seed : int = None, 
                      save : bool = True, 
                      observation_type : str = 'state',
                      randomise_setpoint : bool = False,
                      total_timesteps : int = None):

    if core_env is None:
        core_env = init_core_env(env_type, desired_state, seed) 

    core_env = init_wrappers(core_env, observation_type, randomise_setpoint)
    env = core_env

    if agent_type == 'sac':
        model = SAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef, seed=seed)
    elif agent_type == 'ddpg' : 
        n_actions = core_env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
        model = DDPG(policy='MlpPolicy', env=env, verbose=1,action_noise=action_noise)
    elif agent_type == 'ppo' :
        model = PPO(policy='MlpPolicy', env=env, verbose=1)
    else:
        raise ValueError("Invalid agent type")

    if randomise_setpoint:
        setpoint = 'randomised'
    else:
        setpoint = 'fixed'

    today = datetime.date.today().strftime("%m%d")
    model_dir = f"{global_config.MODELS_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/Desired{desired_state}/{agent_type}"
    log_dir = f"{global_config.LOG_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/{today}/Desired{desired_state}/{agent_type}"
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)
    if total_timesteps is not None:
        model.learn(total_timesteps = total_timesteps)
    else:
        model.learn(total_timesteps = n_episodes * env.unwrapped.max_episode_len)
    if save:
        model.save(model_dir)
    return model, env

def train_default_augmented_sac(core_env : gym.Env = None,
                                env_type : str = 'linear',
                                agent_type : str = 'safesac', 
                                average_q : bool = True,
                                undelayed_critic = None, 
                                desired_state : float | list = 0.8, 
                                n_episodes : int = 100, 
                                ent_coef : float = 0.5, 
                                variable_delay : bool = False,
                                random_delay : bool = True,
                                init_delay : int = None,
                                seed : int = None, 
                                save : bool = True,
                                observation_type : str = 'state', 
                                randomise_setpoint : bool = False,
                                policy_kwargs : dict = None, 
                                total_timesteps : int = None):
    # Initialise the environment
    if core_env is None:
        core_env = init_core_env(env_type, desired_state, seed)

    core_env = init_wrappers(core_env, observation_type, randomise_setpoint)
    # Initial delay
    if init_delay is None or init_delay > global_config.MAX_DELAY: 
        init_delay = np.random.randint(0, global_config.MAX_DELAY)
    # Wrap in delay and augmentation
    env = DelayAction(core_env, delay = init_delay, random_delay=random_delay, max_delay=global_config.MAX_DELAY)
    env = AugmentState(env, known_delay=global_config.MAX_DELAY)
    
    if agent_type == 'safesac':
        safe_model = SafeSAC(policy='MlpPolicy', 
                             env=env, 
                             verbose=1, 
                             ent_coef=ent_coef, undelayed_critic=undelayed_critic, avg = average_q)
    elif agent_type == 'wcsac': 
        safe_model = WorstCaseSAC(policy='MlpPolicy', 
                                  env=env, 
                                  verbose=1, 
                                  ent_coef=ent_coef, 
                                  avg = average_q)
    elif agent_type == 'augsac': 
        safe_model = SAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef)
    else:
        raise ValueError("Invalid agent type")
    
    if random_delay and agent_type == 'augsac':
        agent_type = 'augsacsafe' 

    if not random_delay:
        agent_type = agent_type + 'fixed'

    if average_q == False : 
        agent_type = agent_type + 'min'

    if randomise_setpoint:
        setpoint = 'randomised'
    else:
        setpoint = 'fixed'

    print(f"Env state space {env.observation_space.shape} and action space {env.action_space.shape}")

    today = datetime.date.today().strftime("%m%d")
    model_dir = f"{global_config.MODELS_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/Desired{desired_state}/{agent_type}"
    log_dir = f"{global_config.LOG_PATH}/{core_env.unwrapped.__class__.__name__}/{observation_type}/{setpoint}/{today}/Desired{desired_state}/{agent_type}"
    new_logger = configure(log_dir, ["stdout", "csv"])
    safe_model.set_logger(new_logger)
    # if training_phases is not None: 
    #     n_eps_array = training_phases['n_episodes']
    #     eps_length_array = training_phases['episode_length']
    #     setpoint_regions_array = training_phases['setpoint_regions']

    #     for n_eps, eps_length, setpoint_regions in zip(n_eps_array, eps_length_array, setpoint_regions_array):
    #         env.unwrapped.max_episode_len = eps_length  
    #         env.set_regions(setpoint_regions)
    #         print(f"Set episode len to {env.unwrapped.max_episode_len} and setpoint regions to {env.regions}")
    #         safe_model.learn(total_timesteps = n_eps * eps_length)
    # else:
    if total_timesteps is not None:
        safe_model.learn(total_timesteps = total_timesteps)
    else :
        safe_model.learn(total_timesteps = n_episodes * env.unwrapped.max_episode_len)
    if save:
        safe_model.save(model_dir)

    # print(f"Delay history is {env.delay_history} and setpoint history {env.desired_states_history}")
    try :
        return safe_model, core_env, env.delay_history, env.desired_states_history
    except :
        try : 
            return safe_model, core_env, env.delay_history, None
        except :
            return safe_model, core_env, None, None
        



def train_default_delayed_sac(core_env : gym.Env = None,
                      env_type : str = 'linear', 
                      agent_type : str = 'sac',
                      init_delay : int = None,
                      desired_state : float | list = 0.8, 
                      n_episodes : int = 100, 
                      ent_coef : float = 0.5, 
                      seed : int = None, 
                      save : bool = True):

    if core_env is None:
        core_env = init_core_env(env_type, desired_state, seed) 

    if init_delay is None or init_delay > global_config.MAX_DELAY: 
        init_delay = np.random.randint(0, global_config.MAX_DELAY)

    env = DelayAction(RescaleAction(core_env), delay = init_delay, random_delay=False, return_queue = True)

    if agent_type == 'delayedsac':
        model = DelayedSAC(policy='MlpPolicy', env=env, verbose=1, ent_coef=ent_coef, delay = init_delay)
    else:
        raise ValueError("Invalid agent type")
    
    today = datetime.date.today().strftime("%m%d")
    model_dir = f"{global_config.MODELS_PATH}/{core_env.__class__.__name__}/Desired{desired_state}/{agent_type}"
    log_dir = f"{global_config.LOG_PATH}/{core_env.__class__.__name__}/{today}/Desired{desired_state}/{agent_type}"
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps = n_episodes * global_config.DEFAULT_PARAMS[core_env.__class__.__name__]['max_episode_len'])
    if save:
        model.save(model_dir)
    return model, core_env
