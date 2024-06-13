import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
import global_config
import copy


def scale_range(value,  initial_min, initial_max, final_min, final_max) : 
    return (value - initial_min) * (final_max - final_min) / (initial_max - initial_min) + final_min

def predict_state(env_model, cur_state, actions_queue, delay, timestep = None):
    # print(f"actions queue {actions_queue}")
    # env_model.reset()
    if delay == 0 or len(actions_queue) == 0: 
        return cur_state
    
    # If the augmented state is passed, we need to reshape it to the original state
    if cur_state.shape[0] > env_model.observation_space.shape[0]:
        cur_state = cur_state[:env_model.observation_space.shape[0]]
    env_model.reset()
    env_model.unwrapped.desired_trajectory = None
    env_model.set_state(cur_state)
    if timestep is not None:
        # print(f"Seting env model to timestep {timestep} and env_model desired_trajectory is {len(env_model.desired_trajectory)}, max len {env_model.max_episode_len}")
        env_model.unwrapped.timestep = timestep
    # print(f"Env model state {env_model.state} and cur_state {cur_state}")
    action_size = env_model.action_space.shape[0]
    # Reshape queue if needed
    # if len(np.array(actions_queue).shape) == 1 and action_size > 1: 
    actions_queue = np.array(actions_queue).reshape(np.array(actions_queue).shape[0]//action_size, action_size)

    # print(f"delay {delay} with shape {np.array(actions_queue).shape}")
    # Predict the state with the delayed actions
    try : 
        if env_model.state_space_violation() > 0:
            return env_model.state
    except : 
        for idx in range(env_model.observation_space.shape[0]): 
            if cur_state[idx] > env_model.observation_space.high[idx] or cur_state[idx] < env_model.observation_space.low[idx]:
                return cur_state

    # If state outside gym, state space
    prev_state = cur_state
    for i, start_idx in enumerate(range(- delay, 0)):
        # print(f"Pred fun: Predicting state with action {actions_queue[start_idx]} and state {env_model.state} and timestep {timestep} and delay {delay} and start_idx {start_idx}")
        new_state, reward, terminated, truncated,  info = env_model.step(np.array(actions_queue[start_idx]).reshape(env_model.action_space.shape))
        # print(f"New state {new_state}")
        if  terminated or truncated:
            break
    return new_state

def predict_next_state(env_model, cur_state, action, timestep = None):
    # print(f"Predicting next state with action {action} and state {cur_state} and env state {env_model}")
    # If the augmented state is passed, we need to reshape it to the original state
    # env_model.reset()
    if cur_state.shape[0] > env_model.observation_space.shape[0]:
        cur_state = cur_state[:env_model.observation_space.shape[0]]
    env_model.reset()
    env_model.unwrapped.desired_trajectory = None

    if timestep is not None:
        env_model.unwrapped.timestep = timestep
    # Reshape the action 
    action = np.array(action).reshape(env_model.action_space.shape)
    env_model.set_state(cur_state)
    # print(f"Set state to {env_model.state}")
    try : 
        if env_model.state_space_violation() > 0:
            return env_model.state
    except : 
        for idx in range(env_model.observation_space.shape[0]): 
            if cur_state[idx] > env_model.observation_space.high[idx] or cur_state[idx] < env_model.observation_space.low[idx]:
                return cur_state
    
    new_state, reward, terminated,  truncated, info = env_model.step(action)
    # if terminated:
    #     return cur_state
    return new_state

    
def plot_test(env, states, rewards, actions, executed_actions, states_indexes = None):
    # Plot the results 

    if env.unwrapped.__class__.__name__ == 'RobotSteer' : 
        f, ax = plt.subplots(1,3, figsize = (15,3))
        ax[0].plot( np.array(states)[:,1] )
        ax[0].plot( np.array(states)[:,2] )
        ax[0].axhline(y = env.unwrapped.desired_state[1], xmax=len(states)-1, color = 'green', linestyle = '--')
        ax[0].axhline(y = env.unwrapped.desired_state[2], xmax=len(states)-1, color = 'pink', linestyle = '--')
        ax[0].grid()
        ax[0].legend(['x', 'y', 'x desired', 'y desired'])
        ax[0].set_title('Evolution of positions')
        ax[1].plot(np.array(states)[:,0])
        ax[1].axhline(y = env.unwrapped.desired_state[0], xmax=len(states)-1, color = 'green', linestyle = '--')
        ax[1].grid()
        ax[1].legend(['velocity', 'desired velocity'])
        ax[1].set_title('Evolution of velocity')
        ax[2].plot(np.array(states)[:,1], np.array(states)[:,2])
        ax[2].plot(env.unwrapped.start_state[1], env.unwrapped.start_state[2], 'bo')
        ax[2].plot(env.unwrapped.desired_state[1], env.unwrapped.desired_state[2], 'go')
        ax[2].plot(np.array(states)[-1, 1], np.array(states)[-1, 2], 'ro')
        ax[2].legend(['trajectory', 'start', 'goal', 'end'])
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[2].grid()
        ax[0].set_title('Position trajectory')
        plt.show()
        return 

    f, ax = plt.subplots(1,3,figsize = (20,3))  

    if np.array(states[0]).shape[0] > 1 and states_indexes is not None: 
        for idx in states_indexes:
            ax[0].plot(np.array(states)[:,idx])
            ax[0].axhline(y = env.unwrapped.desired_state[idx], xmax=len(states)-1, color = 'green', linestyle = '--')
            ax[0].grid()
        ax[0].legend(['agent'] + [f"desired state[{idx}]" for idx in states_indexes])
    else: 
        ax[0].plot(np.array(states)[:,0])
        ax[0].axhline(y = env.unwrapped.desired_state, xmax=len(states)-1, color = 'green', linestyle = '--')
        ax[0].grid()
        ax[0].legend(['agent','desired state'])
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('state')
    ax[0].set_title('Evolution of states')
    ax[1].plot(rewards)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('step reward')
    ax[1].grid()
    ax[1].set_title('Rewards')
    ax[2].plot(actions , color = 'pink')
    ax[2].plot(executed_actions)
    ax[2].legend(['agent selected action', 'executed_action'])
    # ax[2].axhline(y = env.unwrapped.desired_state, xmax=len(states)-1, color = 'green', linestyle = '--')
    ax[2].grid()
    ax[2].set_title('Selected and executed actions')
    plt.show()


def plot_multiple_tests(env, States, Actions, Rewards, delays): 
    if env.unwrapped.__class__.__name__ == 'LinearVelocity': 
        plot_idx = [0]
    elif env.unwrapped.__class__.__name__ == 'NonLinearVelocity':
        plot_idx = [0]
    elif env.unwrapped.__class__.__name__ == 'Position':
        plot_idx = [1]
    elif env.unwrapped.__class__.__name__ == 'RobotSteer':
        plot_idx = [1,2]

    f, ax = plt.subplots(1,3, figsize = (20,5))
    colors = global_config.PLOT_COLORS

    for i in range(len(States)):
        for idx in plot_idx:
            ax[0].plot(np.array(States[i])[:,idx], colors[i])
        ax[1].plot(Actions[i], colors[i])
        ax[2].plot(Rewards[i], colors[i])

    ax[0].axhline(y = env.unwrapped.desired_state[plot_idx[0]], xmax=len(States[0])-1, color = colors[-1], linestyle = '--')
    ax[0].axhline(y = env.unwrapped.desired_state[plot_idx[1]], xmax=len(States[0])-1, color = colors[-1], linestyle = '--')

    ax[0].set_xlabel(r'time [$\times10$ms]')
    ax[0].set_ylabel('state')
    ax[1].set_xlabel(r'time [$\times10$ms]')
    ax[1].set_ylabel('action')
    ax[2].set_xlabel(r'time [$\times10$ms]')
    ax[2].set_ylabel('reward')

    legend = ['Safe SAC', 'Undelayed SAC', 'Perfect delayed SAC', 'Max delayed SAC']
    ax[0].legend(legend)


def get_setpoint_trajectory(setpoints, max_episode_len, steps_per_change):
    trajectory = []
    for step in range(0, max_episode_len):
        if step % steps_per_change == 0:
            setpoint_sample = np.array(np.random.choice(setpoints))
        trajectory.append(setpoint_sample)
    return trajectory
    