from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym 
import torch
from general_utils import *
import matplotlib.pyplot as plt

import global_config


class LinearVelocity(gym.Env): 
    """ Linear velocity environment where the agent has to reach a desired velocity by applying a force
        The reward is defined as a distance between the current and desired state
    """

    metadata = {"render_modes:" : []}

    def __init__(self, 
                 m : float = global_config.DEFAULT_PARAMS['LinearVelocity']['m'], 
                 d : float = global_config.DEFAULT_PARAMS['LinearVelocity']['d'], 
                 start_state : np.array = None, 
                 desired_state : np.array = np.array([0.8]), 
                 Ts  : float = global_config.DEFAULT_PARAMS['LinearVelocity']['Ts'], 
                 render_mode = None, 
                 max_episode_len : int = global_config.DEFAULT_PARAMS['LinearVelocity']['max_episode_len'], 
                 seed = None,
                 noise_std : float = 0.0,
                 update_frequency : int = 20) -> None:
        """
        Initialisation of the environment
        :param m: mass of the object
        :param d: damping coefficient
        :param start_state: initial state of the environment, if None it will be randomly generated
        :param desired_state: desired state of the environment
        :param Ts: sampling time
        :param render_mode: mode of rendering
        :param max_episode_len: maximum length of the episode
        :param seed: seed for the environment observation space
        :param noise_std: standard deviation of the noise to add to the state
        """
        super().__init__()
        self.d = d 
        self.m = m
        self.start_state = start_state
        self.state = start_state
        self.desired_state = desired_state
        self.Ts = Ts
        self.observation_space = gym.spaces.Box(low = 0, high = 5, dtype = float)
        self.observation_space.seed(seed = seed)
        self.action_space = gym.spaces.Box(low = -1, high = 1, dtype = float)
        self.render_mode = render_mode
        self.timestep = 0
        self.max_episode_len = max_episode_len
        self.last_reward = 0
        self.last_action = np.array([0.0])
        self.noise_std = noise_std
        self.reward_scale = 1
        self.seed = seed
        np.random.seed(seed)

        try : 
            if (desired_state.shape[0] > self.start_state.shape[0]) or len(desired_state.shape) > 1:
                self.desired_trajectory = get_setpoint_trajectory(desired_state, 
                                                                max_episode_len, 
                                                                steps_per_change=int(update_frequency/self.Ts)) 
                print(f"Init: Trajectory {self.desired_trajectory}")
            else:
                self.desired_trajectory = None
                self.desired_state = desired_state
        except:
            self.desired_trajectory = None
            self.desired_state = desired_state

        print(f"Desired trajectory {self.desired_trajectory} and desired state {self.desired_state}")

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        # self.desired_state = np.array(self.desired_state).reshape(self.start_state.shape)
        if self.start_state is None : 
            self.state = self.observation_space.sample()
        else: 
            self.state = self.start_state

        if self.desired_trajectory is not None:
            self.desired_state = self.desired_trajectory[0].reshape(self.observation_space.shape)

        self.des_history = []
        self.initial_state = self.state
        self.timestep = 0
        self.last_reward = 0.0
        self.int_reward = 0.0
        self.last_action = np.array([0.0])
        # Return state, info
        return self.state, ""

    def step(self, action: float) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.desired_trajectory is not None:
            # print(f"Time step {self.timestep} and desired trajectory {len(self.desired_trajectory)}")
            self.desired_state = self.desired_trajectory[self.timestep].reshape(self.observation_space.shape)

        self.des_history.append(self.desired_state)
        # Calculate new state
        new_state = self.get_new_state(action) + np.random.normal(0, self.noise_std, self.observation_space.shape)
        # print(f"Old state {self.state}, New state {new_state} and action {action}")
        self.state = new_state 
        # print(f"Assigned state {self.state}")
        self.last_reward =  self.reward_fun(action).item()
        self.last_action = action
        self.timestep += 1 
        # Return in the form state, reward, terminated, truncated, info
        # if self.terminated(): 
            # print(f"Terminated at {self.timestep} with state {self.state} and reward {self.last_reward} and initial state {self.initial_state}")
        return self.state, self.last_reward, self.terminated(), self.truncated(), {}

    def truncated(self): 
        return self.timestep >= self.max_episode_len
    
    def terminated(self):
        return False
    
    def reward_fun(self, action): 
        # Absolute error - default
        r_distance = - np.abs(self.desired_state  - self.state) 
        # Add penalty for state space violation
        r_collision = 0 if self.state_space_violation() else 0
        # Add penalty on action
        r_action = -0.01 * action**2
        # Add reward for being close to desired
        r_goal = 1 if np.abs(self.desired_state - self.state) < 0.005 else 0
        return r_distance + r_collision + r_action + r_goal
      
    def close(self):
        return super().close()
    
    def get_new_state(self, action): 
        return self.state * (1 - self.d * self.Ts/self.m) + self.Ts / self.m * action
        
    def plot_step_response(self, step_value: np.array, plot_index: list = [0]):
        """
        Plotting the step response of the system to a constant signal
        :param step_value: value of the step signal
        :param plot_index: index of axes of the state to plot
        """
        states = []
        state, info = self.reset()
        terminated = truncated = False
        while not (terminated or truncated) : 
            states.append(state)
            state, reward, terminated, truncated, info = self.step(step_value)

        f, ax = plt.subplots(1,len(plot_index), figsize = (10,3))
        if len(plot_index) > 1 : 
            for i, plt_index in enumerate(plot_index): 
                ax[i].plot(np.array(states)[:,plt_index])
                ax[i].axhline(y = self.desired_state[plt_index], xmax=len(states)-1, color = 'green', linestyle = '--')
                ax[i].grid()
        else : 
            ax.plot(np.array(states)[:,plot_index[0]])
            ax.axhline(y = self.desired_state[plot_index[0]], xmax=len(states)-1, color = 'green', linestyle = '--')
            ax.grid()

    def get_action_component(self, action) : 
        return self.Ts / self.m * action
    
    def state_space_violation(self):
        penalty = 0 
        for i in range(len(self.state)): 
            penalty += 1 * ((self.state[i] - self.observation_space.high[i]) * float(self.state[i] > self.observation_space.high[i]) \
                        +  (self.observation_space.low[i] -  self.state[i]) * float(self.state[i] < self.observation_space.low[i]))
        return False
    
    def set_state(self, observation):
        self.state = observation

class Position(LinearVelocity): 
    """ 
    Position environment where the agent has to reach a desired position by applying a force,
    The reward is defined as a distance between the current and desired position
    """

    def __init__(self, 
                 m :float = global_config.DEFAULT_PARAMS['Position']['m'], 
                 d :float = global_config.DEFAULT_PARAMS['Position']['d'], 
                 start_state : np.array = None, 
                 desired_state : np.array  = np.array([0.0, 0.8]), 
                 Ts : float =global_config.DEFAULT_PARAMS['Position']['Ts'], 
                 render_mode=None, 
                 max_episode_len : int =global_config.DEFAULT_PARAMS['Position']['max_episode_len'], 
                 seed = None, 
                 noise_std : float = 0.0,
                 update_frequency : int = 20): 
        """
        Initialisation of the environment
        :param m: mass of the object
        :param d: damping coefficient
        :param start_state: initial state of the environment, if None it will be randomly generated
        :param desired_state: desired state of the environment
        :param Ts: sampling time
        :param render_mode: mode of rendering
        :param max_episode_len: maximum length of the episode
        :param seed: seed for the environment observation space
        :param noise_std: standard deviation of the noise to add to the state
        """
        super().__init__(m,d,start_state, desired_state, Ts, render_mode, max_episode_len, seed = seed, noise_std=noise_std, update_frequency=update_frequency)    
        # The action space is between -1 and 1 so that the agent can go backwards
        self.action_space = gym.spaces.Box(low = -1, high = 1, dtype = float)
        self.observation_space = gym.spaces.Box(low = np.array([-3,-3]), high = np.array([3, 3]), dtype = float)
        self.observation_space.seed(seed = seed)
        self.desired_state[0] = 0.0

    def reset(self, seed = None, options = None):
        
        _, info = super().reset(seed = seed)
        self.desired_state[0] = 0.0
    
        return self.state, info

    def reward_fun(self, action):
        # step_reward = - np.abs([self.state[1] - self.desired_state[1]])
        velocity, position = self.state
        v_desired, p_desired = self.desired_state
        # Distance-based reward
        r_distance = -np.abs(p_desired - position)
        # Velocity penalty
        r_velocity = -0.01 * velocity**2
        # Action penalty
        r_action = -0.01 * action**2
        # State space violation
        r_collision = 0 if self.state_space_violation() else 0

        r_goal = 1 if np.abs(p_desired - position) < 0.005 else 0
        return r_distance + r_velocity + r_action + r_collision + r_goal
    
    def get_new_state(self, action): 
        self.state = self.state.reshape(self.observation_space.shape)
        # Calculate the velocity, which is the first element of the state
        new_velocity = self.state[0] * (1 - self.d * self.Ts/self.m) + self.Ts / self.m * action.item()
        # Calculate the new position based on the velocity
        # print(f"Env : State {self.state}")
        new_position = self.state[1] + self.state[0] * self.Ts
        return np.array([new_velocity, new_position])
    
    def get_action_component(self, action):
        return np.array([self.Ts / self.m * action.item(), 0])
    

class NonLinearVelocity(LinearVelocity): 
    """ 
    Non-linear velocity environment where the agent has to reach a desired velocity by applying a force.
    The reward is defined as a distance between the current and desired state
    """

    def __init__(self, 
                 m : float = global_config.DEFAULT_PARAMS['NonLinearVelocity']['m'], 
                 d : float = global_config.DEFAULT_PARAMS['NonLinearVelocity']['d'], 
                 d_nl : float  = global_config.DEFAULT_PARAMS['NonLinearVelocity']['d_nl'], 
                 start_state : np.array =None,
                 desired_state : np.array =np.array([0.8]),
                 Ts : float = global_config.DEFAULT_PARAMS['NonLinearVelocity']['Ts'], 
                 render_mode=None, 
                 max_episode_len : int = global_config.DEFAULT_PARAMS['NonLinearVelocity']['max_episode_len'], 
                 seed = None,
                 noise_std : float = 0.0):
        
        """
        Initialisation of the environment
        :param m: mass of the object
        :param d: damping coefficient
        :param d_nl: non-linear damping coefficient
        :param start_state: initial state of the environment, if None it will be randomly generated
        :param desired_state: desired state of the environment
        :param Ts: sampling time
        :param render_mode: mode of rendering
        :param max_episode_len: maximum length of the episode
        :param seed: seed for the environment observation space
        :param noise_std: standard deviation of the noise to add to the state
        """
        super().__init__(m,d,start_state, desired_state, Ts, render_mode, max_episode_len, seed, noise_std=noise_std) 
        # Non-linear damping coefficient   
        self.d_nl = d_nl
        self.action_space = gym.spaces.Box(low = -2, high = 2, dtype = float)
        self.observation_space = gym.spaces.Box(low = 0, high = 3, dtype = float)
        self.observation_space.seed(seed = seed)
    
    def get_new_state(self, action): 
        # Calculate the next state according to M(x_{k+1} - x_k)/T_s + D * x_k + D_{nl} * x_k^2 = u_k
        return np.array([self.state[0] * (1 - self.Ts/self.m *(self.d + self.d_nl*self.state[0])) + self.Ts / self.m * action.item()]).reshape(self.observation_space.shape)
    
class RobotSteer(LinearVelocity): 
    """ Robot steering environment where the agent has to reach a desired position in 2D space, 
    with velocity and angle as part of the state by applying a force and angular velocity.
    The reward is defined as a distance between the current and desired position."""

    def __init__(self, 
                 m : float = global_config.DEFAULT_PARAMS['RobotSteer']['m'], 
                 d : float = global_config.DEFAULT_PARAMS['RobotSteer']['d'], 
                 start_state : np.array =None, 
                 desired_state : np.array =np.array([0.0, 0.9, 0.9, 0.0]), 
                 Ts : float = global_config.DEFAULT_PARAMS['RobotSteer']['Ts'], 
                 render_mode=None,
                 max_episode_len : int = global_config.DEFAULT_PARAMS['RobotSteer']['max_episode_len'], 
                 seed = None, 
                 noise_std : float = 0.0,
                 update_frequency : int = 20): 
        

        """
        Initialisation of the environment
        :param m: mass of the object
        :param d: damping coefficient
        :param start_state: initial state of the environment, if None it will be randomly generated
        :param desired_state: desired state of the environment, should contain [velocity, x position, y position, angle]
        :param Ts: sampling time
        :param render_mode: mode of rendering
        :param max_episode_len: maximum length of the episode
        :param seed: seed for the environment observation space
        :param noise_std: standard deviation of the noise to add to the state
        """
        super().__init__(m,d,start_state, desired_state, Ts, render_mode, max_episode_len, seed=seed, noise_std=noise_std, update_frequency=update_frequency)    
        self.action_space = gym.spaces.Box(low = np.array([-1,-1]), high = np.array([1, 1]), dtype = float)
        # augmented_state_space = [gym.spaces.Box(low = -3, high = 3, dtype = float) , gym.spaces.Box(low = 0, high = 5, dtype = float), gym.spaces.Box(low = 0, high = 5, dtype = float)]  + augmented_actions
        self.observation_space = gym.spaces.Box(low = np.array([-3,0,0, -np.pi]), high = np.array([3,5,5, np.pi]), dtype = float)
        self.observation_space.seed(seed = seed)
        self.desired_state[0] = 0.0

    def reset(self, seed = None,  options = None):
        _, info = super().reset(seed = seed)
        # print(f"Desired state {self.desired_state} from trajectory {self.desired_trajectory} and start state {self.start_state}")
        self.desired_state = np.array(self.desired_state).reshape(self.observation_space.shape)
        self.desired_state[0] = 0.0
        self.desired_state[3] = 0.0
        # print(f"Desired state {self.desired_state}")
        # # Always start with zero velocity
        if self.start_state is None:
            self.state[0] = 0.0
            self.state[3] = 0.0
        return self.state, info

    def reward_fun(self, action):
        # Reward as distance from the desired and penalty for going outside the state space
        velocity, x, y, angle = self.state
        force, angular_velocity = action
        x_goal, y_goal = self.desired_state[1], self.desired_state[2]

        # Distance-based reward
        r_distance = -np.sqrt((x_goal - x)**2 + (y_goal - y)**2)

        # Goal-reaching reward
        r_goal = 1 if np.sqrt((x_goal - x)**2 + (y_goal - y)**2) < 0.01 else 0

        # Velocity penalty
        r_velocity = -0.01 * (velocity**2 + angular_velocity**2)

        # Action penalty
        r_action = -0.01 * (force**2 + angular_velocity**2)

        # State space violation
        r_collision = 0 if self.state_space_violation() else 0

        return r_distance + r_goal + r_velocity + r_action + r_collision


        # step_reward = - np.sqrt(np.square(self.state[1] - self.desired_state[1]) + np.square(self.state[2] - self.desired_state[2])) 

    def get_new_state(self, executed_action): 
        self.state = self.state.reshape(self.observation_space.shape)
        # Calculate new velocity 
        new_velocity = self.state[0] * (1 - self.d * self.Ts/self.m) + self.Ts / self.m * executed_action[0]
        # New angle 
        new_angle = self.state[3] + executed_action[1] * self.Ts
        # Calculate new positions
        new_position_x = self.state[1] + self.Ts * np.cos(self.state[3]) * self.state[0] 
        new_position_y = self.state[2] + self.Ts * np.sin(self.state[3]) * self.state[0] 
        return np.array([new_velocity, new_position_x, new_position_y, new_angle])
    
class SphericalTank(LinearVelocity):
    """ Spherical tank environment where the agent has to reach a desired level by controlling the inflow.
    The reward is defined as a distance between the current and desired level"""

    def __init__(self, 
                 radius = global_config.DEFAULT_PARAMS['SphericalTank']['radius'],
                 cp = global_config.DEFAULT_PARAMS['SphericalTank']['cp'],
                 start_state : np.array = None, 
                 desired_state : np.array = np.array([0.8]), 
                 Ts : float = global_config.DEFAULT_PARAMS['SphericalTank']['Ts'], 
                 render_mode=None, 
                 max_episode_len : int = global_config.DEFAULT_PARAMS['SphericalTank']['max_episode_len'], 
                 seed = None, 
                 noise_std : float = 0.0): 
        """
        Initialisation of the environment
        :param m: mass of the object
        :param d: damping coefficient
        :param start_state: initial state of the environment, if None it will be randomly generated
        :param desired_state: desired state of the environment
        :param Ts: sampling time
        :param render_mode: mode of rendering
        :param max_episode_len: maximum length of the episode
        :param seed: seed for the environment observation space
        :param noise_std: standard deviation of the noise to add to the state
        """
        super().__init__(0.0, 0.0, start_state, desired_state, Ts, render_mode, max_episode_len, seed=seed, noise_std=noise_std)    
        self.action_space = gym.spaces.Box(low = 0, high = 15, dtype = float)
        self.radius = radius
        self.observation_space = gym.spaces.Box(low = 0.0, high = 2 * self.radius, dtype = float)
        # self.observation_space = gym.spaces.Box(low = 1.5, high = 2.5, dtype = float)
        self.observation_space.seed(seed = seed)
        self.gravity_constant = 9.81
        self.cp = cp
        if len(desired_state) > 1 :
            self.desired_trajectory = get_setpoint_trajectory(desired_state, 
                                                              max_episode_len, 
                                                              steps_per_change=int(20/self.Ts)) 
        else:
            self.desired_trajectory = None
            self.desired_state = np.array(desired_state).reshape(self.observation_space.shape)
        self.initial_state = None
        self.start_state_space = gym.spaces.Box(low = 1.0, high = 3.0, dtype = float)
        self.start_state_space.seed(seed = seed)
        

    def reset(self, seed = None, options = None):
        _, info = super().reset(seed=seed, options=options)
        # Clip start state 
        if self.start_state is None :
            self.state = self.start_state_space.sample()
        else:
            self.state = self.start_state

        self.initial_state = self.state
        if self.desired_trajectory is not None:
            self.desired_state = self.desired_trajectory[0].reshape(self.observation_space.shape)
        self.des_history = []
        # self.desired_state = np.clip(self.desired_state, 0.2, 3.8)
        # print(f"Initial state {self.state} and desired state {self.desired_state}")
        return self.state, info

    def reward_fun(self, action):
        step_reward = - 5 * np.abs(self.state- self.desired_state)#/(self.observation_space.high - self.observation_space.low)
        step_reward -= 500* self.state_space_violation() #+ 5*self.overshoot()
        
        # Action penalty
        step_reward -= 0.03 * (0.5 - action/self.action_space.high)**2
        # print(f"Action penalty for action {action} is {0.01 * (0.5 - action/self.action_space.high)**2}")

        # Goal reward
        step_reward += 1 if np.abs(self.state - self.desired_state) < 0.01 else 0

        scaled_reward = step_reward * self.reward_scale
        # Add term for overshoot
        return scaled_reward
    
    def get_new_state(self, action): 
        self.state = np.array(self.state).reshape(self.observation_space.shape)
        # Calculate the velocity, which is the first element of the state
        # print(self.state)
        
        new_height = self.state + self.Ts * (action - self.cp * np.sqrt(2 * self.gravity_constant * self.state)) /(np.pi* (2 * self.radius * self.state - self.state**2))
        # print(f"In step {self.timestep} and state {self.state} and action {action}, new height {new_height}")
        return new_height

    def terminated(self): 
        # return False
        return self.state >= self.observation_space.high or self.state <= self.observation_space.low
        
    
    def step(self, action): 
        if self.desired_trajectory is not None:
            # print(f"Time step {self.timestep} and desired trajectory {len(self.desired_trajectory)}")
            self.desired_state = self.desired_trajectory[self.timestep].reshape(self.observation_space.shape)
        self.des_history.append(self.desired_state)
        return super().step(action)
    
    def overshoot(self):
        if self.initial_state is None:
            return 0
        # Undershoot
        if self.initial_state > self.desired_state and self.state < self.desired_state - 0.1:
            return self.desired_state - self.state
        # Overshoot
        elif self.initial_state < self.desired_state and self.state > self.desired_state + 0.1:
            return self.state - self.desired_state
        else:
            return 0
    
    def state_space_violation(self):
        penalty = 0 
        for i in range(len(self.state)): 
            penalty += 1 * ((self.state[i] - self.observation_space.high[i]) * float(self.state[i] > self.observation_space.high[i]) \
                        +  (self.observation_space.low[i] -  self.state[i]) * float(self.state[i] < self.observation_space.low[i]))
        if penalty > 0:
            return max(0.1, penalty)
        return penalty

def init_core_env(env_type : str, desired_state : float | list , seed : int = None):

    if env_type == 'linear':
        core_env = LinearVelocity(desired_state=np.array([desired_state]), seed = seed)
    elif env_type == 'nonlinear':
        core_env = NonLinearVelocity(desired_state=np.array([desired_state]), seed = seed)
    elif env_type == 'position' : 
        if not isinstance(desired_state, list):
            desired_state = np.array([0, desired_state])
        else : 
            desired_state = np.array(desired_state)
        core_env = Position(desired_state=desired_state, seed = seed)
    elif env_type == 'robotsteer' :
        if not isinstance(desired_state, list):
            raise ValueError("Desired state should be a list for robotsteer")
        elif len(desired_state) == 2:
            desired_state = np.array([0, desired_state[0], desired_state[1], 0])
        core_env = RobotSteer(desired_state=desired_state, seed = seed)
    elif env_type == 'sphericaltank' :
        if not isinstance(desired_state, list):
            desired_state = np.array([desired_state])
        core_env = SphericalTank(desired_state=np.array(desired_state), seed = seed)
    else:
        raise ValueError("Invalid environment type")
    return core_env