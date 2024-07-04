from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
import global_config
from general_utils import *
from copy import deepcopy

class RescaleAction(gym.Wrapper): 
    """ Rescale the action space of the environment to a new range"""

    def __init__(self, env: gym.Env, min_action: float | int | np.ndarray = -1, max_action: float | int | np.ndarray = 1):
        """
        Initialise the wrapper
        :param env: the environment to wrap
        :param min_action: the minimum value of the initial range
        :param max_action: the maximum value of the initial range
        """
        super().__init__(env)
        self.min_action = min_action
        self.max_action = max_action
        if type(min_action) in [float, int] and len(self.env.action_space.low) > 1 : 
            space_min = np.ones_like(self.env.action_space.low) * min_action 
        else : 
            space_min = min_action
        if type(max_action) in [float, int] and len(self.env.action_space.high) > 1 : 
            space_max = np.ones_like(self.env.action_space.high) * max_action 
        else : 
            space_max = max_action
        self.action_space = gym.spaces.Box(low = space_min, high = space_max, dtype = float)

    def step(self, action): 
        """
        Take a step in the environment with the rescaled action
        """
        scaled_to_env = scale_range(action, self.min_action, self.max_action, self.env.action_space.low, self.env.action_space.high)
        self.last_action = action
        return self.env.step(scaled_to_env)
    
    @property
    def state(self):
        return self.unwrapped.state

    @state.setter
    def state(self, new_value):
        self.env.state = new_value
        
    def get_action_component(self, action) : 
        scaled_to_env = scale_range(action, self.min_action, self.max_action, self.env.action_space.low, self.env.action_space.high)
        return self.env.get_action_component(scaled_to_env)
    
    def plot_step_response(self, step_value, plot_index):
        scaled_to_env = scale_range(step_value, self.min_action, self.max_action, self.env.action_space.low, self.env.action_space.high)
        self.env.plot_step_response(scaled_to_env, plot_index)

    def set_state(self, observation):
        try:
            # For custom envs
            self.env.set_state(observation)
        except:
            # For gym envs
            self.env.state = observation


class DelayAction(gym.Wrapper): 
    """ Delay the action of the agent by a fixed number of steps"""

    def __init__(self, 
                env: gym.Env, 
                delay : int, 
                random_delay : bool = False, 
                max_delay : int = global_config.MAX_DELAY, 
                return_queue : bool = False):
        """
        Initialise the wrapper
        :param env: the environment to wrap
        :param delay: the number of steps to delay the action
        :param random_delay: whether to use a random delay, sample a new delay value at each env restart
        :param max_delay: the maximum value of the delay, used for the random option
        """
        super().__init__(env)
        self.delay = delay
        self.actions_queue = []
        self.random_delay = random_delay
        self.max_delay = max_delay
        self.return_queue = return_queue
        self.delay_history = []
        self.action_history =[]
        self.to_use = [i for i in range(0, max_delay + 1)]
        # For random initi queue
        # safe_min = self.env.action_space.low + 0.3 * (self.action_space.high - self.action_space.low)
        # safe_max = self.env.action_space.high - 0.3 * (self.action_space.high - self.action_space.low)
        # self.safe_action_space = gym.spaces.Box(low = safe_min, high = safe_max, dtype = float)
        # self.safe_action_space.seed(self.unwrapped.seed)

        # Idea with delay regions
        # self.delay_regions = [[0, global_config.MAX_DELAY//3], [global_config.MAX_DELAY//3, 2*global_config.MAX_DELAY//3], [2*global_config.MAX_DELAY//3, global_config.MAX_DELAY]]
        # print(f"Delay regions are {self.delay_regions}")
        # self.last_used_region = None

    def step(self, action) : 
        queue = deepcopy(self.actions_queue)
        # Append selected action
        self.actions_queue.append(action.reshape(self.env.action_space.shape))
        # print(f"Delayed env: Before queue {self.actions_queue} in state {self.state}")
        # Pop the action to be executed
        delayed_action = self.actions_queue.pop(0)
        # Save to return to agent
        self.last_action = delayed_action
        # print(f"Delayed env: Executed {delayed_action} and after queue {self.actions_queue}")
        # Execute the action
        if self.return_queue :
            next_state, reward, terminated, truncated, infos = self.env.step(delayed_action)
            # print(f"Delayed env: next state {next_state}")
            return next_state, reward, terminated, truncated, {'queue': queue}
        else : 
            return self.env.step(delayed_action)
        
    def reset(self, seed = None, options = None):

        # Sampled delay
        if self.random_delay : 
            self.delay = np.random.randint(0, self.max_delay + 1)
                
            # if self.last_used_region is not None :
            #     p = [1/(len(self.delay_regions)-1) for _ in range(len(self.delay_regions))]
            #     p[self.last_used_region] = 0
            # else : 
            #     p = [1/(len(self.delay_regions)) for _ in range(len(self.delay_regions))]
            
            # sel_region = np.random.choice(len(self.regions), p = p)
            # self.delay = np.random.randint(self.delay_regions[sel_region][0], self.delay_regions[sel_region][1])
            # self.last_used_region = sel_region

        # # To ensure that we choose a new delay at each episode 
        # if self.random_delay : 
        #     self.delay = np.random.choice(self.to_use)
        #     self.to_use.remove(self.delay)
        #     # print(f"Random delay is {self.delay} and to use {self.to_use}")
        #     if len(self.to_use) == 0 : 
        #         self.to_use = [i for i in range(0, self.max_delay + 1)]
        self.delay_history.append(self.delay)

        # # Initialise the action queue always with 0 (scaled to the action space of the environment)
        scaled = scale_range(np.zeros(self.action_space.shape), self.unwrapped.action_space.low, self.unwrapped.action_space.high, self.action_space.low, self.action_space.high)
        self.actions_queue = [np.ones(self.action_space.shape)*scaled  for _ in range(self.delay)] 

        # Random actions

        # sampled_action = self.safe_action_space.sample()
        # # print(f"Safe action space is {safe_action_space} and sampled action {sampled_action}")
        # self.actions_queue = [sampled_action for _ in range(self.delay)]
        # # print(f"Delay action queue is {self.actions_queue} and delay is {self.delay}")
        # self.action_history.append(sampled_action)
        # self.actions_queue = [safe_action_space.sample() for _ in range(self.delay)]
        return self.env.reset(seed = seed)   
    

class AugmentState(gym.Wrapper):
    """ Augment the state of the environment with the last n actions taken by the agent""" 

    def __init__(self, env: gym.Env, known_delay : int = global_config.MAX_DELAY):
        """
        Initialise the wrapper
        :param env: the environment to wrap
        :param known_delay: the number of actions appended to the state, default is the maximum delay
        """
        super().__init__(env)
        self.known_delay = known_delay
        augmented_actions = [self.env.action_space for _ in range(known_delay)]
        space_min = np.concatenate([self.observation_space.low] + [self.env.action_space.low for _ in range(known_delay)])
        space_max = np.concatenate([self.observation_space.high] + [self.env.action_space.high for _ in range(known_delay)])
        # self.observation_space = gym.spaces.Tuple(augmented_state_space)
        self.observation_space = gym.spaces.Box(low = space_min, high = space_max, dtype = float)
        self.agent_queue = []

    def reset(self, seed = None, options = None) : 
        state, info = super().reset(seed = seed)
        try : 
            # If the action queue is not empty, use the first action in the queue
            scaled = self.env.actions_queue[0]
        except : 
            # If the action queue is empty, meaning actual delay is 0, sample a random safe action
            # scaled = self.env.safe_action_space.sample()
            scaled = scale_range(np.zeros(self.action_space.shape), self.unwrapped.action_space.low, self.unwrapped.action_space.high, self.action_space.low, self.action_space.high)
        self.actions_queue = [np.ones(self.action_space.shape)*scaled  for _ in range(self.known_delay)] 
        # print(f"Augmented action queue is {self.actions_queue}")
        augmented_state = np.concatenate([state, np.array(self.actions_queue).reshape(len(self.actions_queue) * self.action_space.shape[0])])
        self.state = augmented_state
        return augmented_state, info

    def step(self, action): 
        # print(f"Actions queue {np.array(self.actions_queue).shape}")
        new_state, reward, terminated, truncated, info = self.env.step(action=action)
        # agent_queue = np.concatenate([self.augmented_state[new_state.shape[0]:], action]
        agent_queue = np.concatenate([self.state[new_state.shape[0] + self.action_space.shape[0]:], action])
        self.actions_queue = agent_queue #[len(agent_queue) - self.known_delay * self.action_space.shape[0]:]
        augmented_state = np.concatenate([new_state, agent_queue])
        self.state = augmented_state
        return augmented_state, reward, terminated, truncated, info
    
    @property
    def last_action(self): 
        return self.env.last_action
        
class ObserveError(gym.Wrapper):
    """ Observe the state of the error""" 

    def __init__(self, env: gym.Env):
        """
        Initialise the wrapper
        :param env: the environment to wrap
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low = -self.env.observation_space.high, high = self.env.observation_space.high, dtype = float)
        self.error = 0

    def reset(self, seed = None, options = None) :
        """
        Reset the environment and return the error
        :param seed: the seed for the environment
        :param options: the options for the environment
        """
        state, info = super().reset(seed = seed, options= options)
        self.error = self.unwrapped.desired_state - state
        return self.error, info
    
    def step(self, action): 
        """
        Take a step in the environment and return the error
        :param action: the action to take in the environment
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        self.error = self.unwrapped.desired_state - state
        return self.error, reward, terminated, truncated, info
    
    
class ObserveStateAndError(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Initialise, change observation space
        """
        super().__init__(env)
        self.error = 0
        self.observation_space = gym.spaces.Box(low = np.concatenate([self.env.observation_space.low, self.env.observation_space.low - self.env.observation_space.high]), 
                                                high = np.concatenate([self.env.observation_space.high, self.env.observation_space.high - self.env.observation_space.low]), dtype = float)
        
    def reset(self, seed = None, options = None) :
        state, info = super().reset(seed = seed, options= options)
        self.error = self.unwrapped.desired_state - state
        return np.concatenate([state, self.error]), info

    def step(self, action): 
        state, reward, terminated, truncated, info = self.env.step(action)
        self.error = self.unwrapped.desired_state - state
        return np.concatenate([state, self.error]), reward, terminated, truncated, info
    
class ObserveSetpointError(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Initialise, change observation space
        """
        super().__init__(env)
        self.error = 0
        self.observation_space = gym.spaces.Box(low = np.concatenate([self.env.observation_space.low, self.env.observation_space.low - self.env.observation_space.high]), 
                                                high = np.concatenate([self.env.observation_space.high, self.env.observation_space.high - self.env.observation_space.low]), dtype = float)
        
    def reset(self, seed = None, options = None) :
        state, info = super().reset(seed = seed, options= options)
        self.error = self.unwrapped.desired_state - state
        return np.concatenate([self.unwrapped.desired_state , self.error]), info

    def step(self, action): 
        state, reward, terminated, truncated, info = self.env.step(action)
        # print(f"Desired state {self.unwrapped.desired_state} and state {state}")
        self.error = self.unwrapped.desired_state - state
        return np.concatenate([self.unwrapped.desired_state , self.error]), reward, terminated, truncated, info
    
    def set_state(self, observation):
        observation = observation.reshape(self.observation_space.shape)
        self.unwrapped.state = observation[:self.unwrapped.observation_space.shape[0]] - observation[self.unwrapped.observation_space.shape[0]:]
        self.unwrapped.desired_state = observation[:self.unwrapped.observation_space.shape[0]]

class ObserveStateErrorIntegral(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Initialise, change observation space
        """
        super().__init__(env)
        self.error = 0
        self.integral = 0
        self.observation_space = gym.spaces.Box(low = np.concatenate([self.env.observation_space.low, 
                                                                      self.env.observation_space.low - self.env.observation_space.high, 
                                                                      (self.env.observation_space.low - self.env.observation_space.high)*self.env.max_episode_len]),
                                                high = np.concatenate([self.env.observation_space.high, 
                                                                       self.env.observation_space.high - self.env.observation_space.low, 
                                                                       (self.env.observation_space.high - self.env.observation_space.low)*self.env.max_episode_len]), dtype = float)
        
    def reset(self, seed = None, options = None) :
        state, info = super().reset(seed = seed, options= options)
        self.error = self.unwrapped.desired_state - state
        self.integral = self.error
        return np.concatenate([state, self.error, self.integral]), info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.error = self.unwrapped.desired_state - state
        self.integral += self.error
        return np.concatenate([state, self.error, self.integral]), reward, terminated, truncated, info
    
class RescaleObservation(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low = -1, high = 1, shape = self.observation_space.shape, dtype = float)

    def reset(self, seed = None, options = None) :
        state, info = super().reset(seed = seed, options= options)
        return scale_range(state, self.env.observation_space.low, self.env.observation_space.high, -1, 1), info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return scale_range(state, self.env.observation_space.low, self.env.observation_space.high, -1, 1), reward, terminated, truncated, info

    def set_state(self, observation):
        scaled_value = scale_range(observation, -1, 1, self.env.observation_space.low, self.env.observation_space.high)
        self.env.set_state(scaled_value)
    
    
class VariableActionDelay(gym.Wrapper) : 

    def __init__(self, env: gym.Env,
                 max_delay : int, 
                 init_delay : int = 0, 
                 delay_type : str = 'wear', 
                 update_freq : int = 200,
                 keep_trajectory : bool = False):
        
        super().__init__(env)
        self.max_delay = max_delay
        self.actions_queue = []
        self.init_delay = init_delay
        self.delay_type = delay_type
        if delay_type != 'external' and delay_type != 'wear' :
            print(f"Invalid variable delay type {delay_type}, setting to constant")
        self.update_freq = update_freq
        self.ep_delay_history = []
        self.keep_trajectory = keep_trajectory

    def step(self, action) : 
        queue = deepcopy(self.actions_queue[-int(self.current_delay):])
        # Append selected action
        self.actions_queue.append(action.reshape(self.env.action_space.shape))
        # Pop the action to be executed
        delayed_action = self.actions_queue[-int(self.current_delay) - 1]
        self.actions_queue.pop(0)
        # Save to return to agent
        self.last_action = delayed_action
        self.update_current_delay()
        self.ep_delay_history.append(self.current_delay)
       
        return self.env.step(delayed_action)
    
    def reset(self, seed = None, options = None):
        # Initialise the action queue always with 0
        scaled = scale_range(np.zeros(self.action_space.shape), self.unwrapped.action_space.low, self.unwrapped.action_space.high, self.action_space.low, self.action_space.high)
        self.actions_queue = [np.ones(self.action_space.shape)*scaled  for _ in range(self.max_delay)] 
        self.current_delay = self.init_delay
        if not self.keep_trajectory :
            self.ep_delay_history = []
        
        return self.env.reset(seed = seed)  
    
    def update_current_delay(self): 
        if self.keep_trajectory and len(self.ep_delay_history) > self.unwrapped.timestep:
            # print(f"Len delay history {len(self.ep_delay_history)} and timestep is {self.unwrapped.timestep}")
            self.current_delay = self.ep_delay_history[self.unwrapped.timestep]
            return
        
        if self.unwrapped.timestep == 0:
            return
        if self.delay_type == 'external' :
            if self.unwrapped.timestep % self.update_freq == 0 :
                self.current_delay = np.random.choice(range(0,self.max_delay +1))
        elif self.delay_type == 'wear' :
            if self.unwrapped.timestep  % self.update_freq == 0 :
                self.current_delay = min(self.current_delay +1, self.max_delay)
        else :
            return

    @property
    def delay(self) :
        return self.current_delay

    @property
    def state(self):
        return self.unwrapped.state
    
class RandomiseSetpoint(gym.Wrapper):
    def __init__(self, env: gym.Env, min_setpoint : float = None, max_setpoint : float = None):
        super().__init__(env)
        if min_setpoint is None : 
            min_setpoint = self.env.observation_space.low + 0.2
        if max_setpoint is None :
            max_setpoint = self.env.observation_space.high - 0.2

        self.min_setpoint = min_setpoint
        self.max_setpoint = max_setpoint
        self.desired_states_history = []
        # regions_limits = [[0.4, 1.3],[1.3, 2.2], [2.2, 3.6]]
        regions_limits = [[min_setpoint, min_setpoint + (max_setpoint - min_setpoint)/3 ], 
                          [min_setpoint + (max_setpoint - min_setpoint)/3, min_setpoint + 2*(max_setpoint - min_setpoint)/3], 
                          [min_setpoint + 2*(max_setpoint - min_setpoint)/3, max_setpoint]]
        print(f"Regions limits are {regions_limits}")
        self.regions = [gym.spaces.Box(low = regions_limits[i][0], high = regions_limits[i][1], shape = self.observation_space.shape, dtype = float) for i in range(len(regions_limits))]
        self.last_used_region = None

    def reset(self, seed = None, options = None) :

        if self.last_used_region is not None :
            p = [1/2 for _ in range(len(self.regions))]
            p[self.last_used_region] = 0
        else : 
            p = [1/3 for _ in range(len(self.regions))]
        
        sel_region = np.random.choice(len(self.regions), p = p)
        self.desired_state = self.regions[sel_region].sample()
        self.last_used_region = sel_region
        self.desired_state = np.clip(self.desired_state, self.min_setpoint, self.max_setpoint)
        self.desired_states_history.append(self.desired_state)
        state, info = super().reset(seed = seed, options= options)
        return state, info
    
    @property   
    def desired_state(self):
        return self.unwrapped.desired_state

    @desired_state.setter
    def desired_state(self, new_value):
        self.unwrapped.desired_state = new_value
    
class RandomiseObservation(gym.Wrapper) :
    def __init__(self, env: gym.Env, sigma : float = 0.1):
        super().__init__(env)
        self.sigma = sigma
    
    def step(self, action) :
        return self.env.step(action)

    @property
    def state(self):
        return self.unwrapped.state

    @state.setter
    def state(self, new_value):
        self.unwrapped.state = new_value

    def get_new_state(self, action): 
        # Calculate the new state as before 
        new_state = self.env.get_new_state(action)
        # Add noise to the new state
        new_state += np.random.normal(loc = 0, scale = self.sigma, size = new_state.shape)
        return np.clip(new_state, self.observation_space.low, self.observation_space.high)


def init_wrappers(core_env : gym.Env, 
                  observation_type: str = 'state', 
                  randomise_setpoint : bool = False, 
                  rescale_action : bool = True, 
                  rescale_observation : bool = True):
    
    if randomise_setpoint:
        print(f"Wrapping in random setpoint")
        core_env = RandomiseSetpoint(core_env)
    if observation_type == 'error':
        print(f"Wrapping in observe error")
        core_env = ObserveError(core_env)
    elif observation_type == 'state-error':
        print(f"Wrapping in observe state error")
        core_env = ObserveStateAndError(core_env)
    elif observation_type == 'state':
        print(f"Observing state without wrappers") 
    elif observation_type == 'state-error-integral':
        print(f"Wrapping in observe state error integral")
        core_env = ObserveStateErrorIntegral(core_env)

    elif observation_type == 'setpoint-error':
        print(f"Wrapping in observe setpoint error")
        core_env = ObserveSetpointError(core_env)
    else:
        raise ValueError("Invalid observation type")
    if rescale_action:
        print(f"Rescaling action")
        core_env = RescaleAction(core_env)
    if rescale_observation:
        print(f"Rescaling observation")
        core_env = RescaleObservation(core_env)
    # env = RescaleAction(core_env)
    # env = RescaleObservation(env)

    return core_env


def get_state_from_obs(env_model, observation_type, observation): 
    if observation_type == 'state':
        return observation
    elif observation_type == 'error':
        return env_model.desired_state - observation
    elif observation_type == 'state-error' or observation_type == 'state-error-integral':
        return observation[:env_model.unwrapped.observation_space.shape[0]]
    elif observation_type == 'setpoint-error':
        return observation[:env_model.unwrapped.observation_space.shape[0]] - observation[env_model.unwrapped.observation_space.shape[0]:]
    
class PendulumGymWrapper(gym.Wrapper): 
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def set_state(self, observation):
        theta = np.arctan2(observation[1], observation[0])
        theta_dot = observation[2]
        # print(f"Pend wrapper: Setting state {[theta, theta_dot]} from obs {observation}")
        self.unwrapped.state = np.array([theta, theta_dot])