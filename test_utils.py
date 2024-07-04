import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
import global_config
from env_wrappers import RescaleAction, DelayAction, AugmentState
import copy
from general_utils import *

def test_delayed_agent(agent, env, env_model, seed = None, known_delay = None, deterministic = False, state_error = False):

    states = []
    rewards = []
    actions = []
    executed_actions = []
    avg_rewards = []

    if known_delay is None : 
        known_delay = env.delay

    state, done = env.reset()
    episode_reward = 0
    t = 0
    # Play episode
    terminated = truncated = False
    while not (terminated or truncated):
        if state_error :
            states.append(env.unwrapped.state)
        else :
            states.append(state)
        try : 
            state = state[:env.observation_space.shape[0]-env.known_delay * env.action_space.shape[0]]
        except AttributeError: 
            state = state

        pred_state = predict_state(env_model, state, env.actions_queue, delay = known_delay, timestep = env.timestep)
        # if t >= 200 and t <= 220:
            # print(f"Del:State is {state} and pred_state {pred_state}")
        action, _states = agent.predict(pred_state, deterministic = deterministic)
        new_state, reward, terminated, truncated, info = env.step(action) 
        
        executed_action = env.last_action 
        if done : 
            break
        reward = np.array([reward], dtype = float)    
        state = new_state
        episode_reward += agent.gamma**t * reward
        t+= 1
        # Save history
        actions.append(action)
        executed_actions.append(executed_action)
        rewards.append(reward)
        avg_rewards.append(np.mean(rewards[-10:]))      

    return states, actions, rewards, avg_rewards, executed_actions

def test_augmented_agent(agent, env, seed = None, deterministic = False, state_error = False):

    states = []
    rewards = []
    actions = []
    executed_actions = []
    avg_rewards = []

    state, done = env.reset(seed = seed)
    episode_reward = 0
    t = 0
    # Play episode
    terminated = truncated = False
    while not (terminated or truncated):
    
        if state_error :
            states.append(env.unwrapped.state)
        else : 
            states.append(state)
        # if t >= 200 and t <= 220:
            # print(f"Aug:State is {state}")
        action,_ = agent.predict(state, deterministic = deterministic)
        new_state, reward, terminated, truncated, info = env.step(action)
        executed_action = env.last_action 
        if (terminated or truncated) : 
            break
        reward = np.array([reward], dtype = float)    
        state = new_state
        episode_reward += reward
        t+= 1
        # Save history
        actions.append(action)
        executed_actions.append(executed_action)
        
        avg_rewards.append(np.mean(rewards[-10:]))    
    print(f"Episode reward was {episode_reward}")
    return states, actions, rewards, avg_rewards, executed_actions


def test_agent(agent, env, seed = None, deterministic = False, state_error = False, return_ep_reward = False):

    states = []
    rewards = []
    actions = []
    executed_actions = []
    avg_rewards = []

    state, done = env.reset(seed = seed)
    # print(f"Initial state was {state}")
    episode_reward = 0
    t = 0
    # Play episode
    terminated = truncated = False
    while not (terminated or truncated):
        if state_error :
            states.append(env.unwrapped.state)
        else : 
            states.append(state)
        action,_ = agent.predict(state, deterministic = deterministic)
        new_state, reward, terminated, truncated, info = env.step(action)
        executed_action = env.last_action 
        
        reward = np.array([reward], dtype = float)    
        state = new_state
        episode_reward += reward
        t+= 1
        # Save history
        actions.append(action)
        executed_actions.append(executed_action)
        rewards.append(reward)
        avg_rewards.append(np.mean(rewards[-10:]))    

        if (terminated or truncated) : 
            print(f"Final reward was {reward}")
            break  
    actions.append(actions[-1])

    print(f"Episode reward was {episode_reward}")
    if return_ep_reward : 
        return states, actions, rewards, avg_rewards, executed_actions, episode_reward
    else :
        return states, actions, rewards, avg_rewards, executed_actions

def test_different_delayed_agents(agent, 
                                  core_env, 
                                  delay_range: list | range  = None, 
                                  deterministic = False, 
                                  plot = True, 
                                  augmented_agent = None, 
                                  plot_indexes = [0]):

    if delay_range is None :
        delays = range(0, global_config.MAX_DELAY+1, global_config.MAX_DELAY // 5)
    else : 
        delays = delay_range

    env_model = RescaleAction(copy.deepcopy(core_env))
    # print(f"Env {core_env.max_episode_len} and {env_model.max_episode_len}")

    for delay in delays : 
        States = []
        Actions = []
        Rewards = []
        Ep_rewards = []
        for i in range(4) : 
            test_env = DelayAction(RescaleAction(core_env), delay = delay)
            test_env = AugmentState(test_env, known_delay=global_config.MAX_DELAY)

            if i == 0 and augmented_agent is not None : 
                states, actions, rewards, avg_rewards, executed_actions = test_augmented_agent(augmented_agent, test_env, deterministic=deterministic, state_error=True)
            elif i == 1 : 
                states, actions, rewards, avg_rewards, executed_actions = test_delayed_agent(agent, test_env, env_model, known_delay = 0, deterministic=deterministic,state_error=True)
            elif i ==2: 
                states, actions, rewards, avg_rewards, executed_actions = test_delayed_agent(agent, test_env, env_model, known_delay = delay, deterministic=deterministic,state_error=True)
            else :
                states, actions, rewards, avg_rewards, executed_actions = test_delayed_agent(agent, test_env, env_model, known_delay = global_config.MAX_DELAY, deterministic=deterministic,state_error=True)

            States.append(states)
            Actions.append(actions)
            Rewards.append(rewards)
            Ep_rewards.append(np.sum([agent.gamma ** i * rewards[i] for i in range(len(rewards))]))

        if plot :
            # Plot comparison
            f, ax = plt.subplots(1,3, figsize = (20,5))
            colors = global_config.PLOT_COLORS

            for i in range(len(States)): 
                for idx in plot_indexes:
                    ax[0].plot(np.array(States[i])[:,idx], colors[i])
                # ax[0].plot(np.array(States[i])[:,0], colors[i])
                ax[1].plot(Actions[i], colors[i])
                ax[2].plot(Rewards[i], colors[i])

            for idx in plot_indexes : 
                if test_env.unwrapped.desired_trajectory is not None :
                    ax[0].stairs(test_env.unwrapped.desired_trajectory, color = colors[-1], linestyle = '--')
                else :
                    ax[0].axhline(y = test_env.unwrapped.desired_state[idx], xmax=len(states)-1, color = colors[-1], linestyle = '--')

            ax[0].set_xlabel(r'time [$\times10$ms]')
            ax[0].set_ylabel('state')
            ax[1].set_xlabel(r'time [$\times10$ms]')
            ax[1].set_ylabel('action')
            ax[2].set_xlabel(r'time [$\times10$ms]')
            ax[2].set_ylabel('reward')

            # if augmented_agent is not None : 
            
            legend = ['Safe SAC', 'Undelayed SAC', 'Perfect delayed SAC', 'Max delayed SAC']
            legend_all = []
            for legend_entry in legend:
                for idx in plot_indexes :
                    legend_all.append(legend_entry + f' state {idx}')
            # else :
            #     legend = [ 'Undelayed SAC', 'Perfect delayed SAC', 'Max delayed SAC']
                
            ax[0].legend(legend_all)

            ax[0].set_title(f'Evolution of states, test delay = {delay}')
            ax[1].set_title(f'Evolution of actions, test delay = {delay}')
            ax[2].set_title(f'Evolution of rewards, test delay = {delay}')

            ax[0].grid()
            ax[1].grid()
            ax[2].grid()
    
    return States, Actions, Rewards, Ep_rewards