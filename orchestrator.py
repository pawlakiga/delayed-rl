from general_utils import *
import gymnasium as gym

class DelayInformedAgent(): 

    def __init__(self,
                 undelayed_agent, 
                 env_model : gym.Env,
                 init_delay : int = 0) -> None:
        
        self.undelayed_agent = undelayed_agent
        self.env_model = env_model
        self._delay = init_delay
        self.init_delay = init_delay

    def predict(self, augmented_state, deterministic = True):
        # Get action queue and base state
        base_state = augmented_state[:self.env_model.observation_space.shape[0]]
        action_queue = augmented_state[self.env_model.observation_space.shape[0]:]
        # Predict delayed state 
        pred_state = predict_state(self.env_model, cur_state = base_state, actions_queue = action_queue, delay = self._delay, timestep=0)
        # print(f"Predicted state {pred_state} with delay {self._delay} from base state {base_state} and action queue {action_queue}")
        # Get action from undelayed agent
        action = self.undelayed_agent.predict(pred_state, deterministic = deterministic)
        # print(f"Action for pred state {pred_state} is {action}")
        return action
    
    @property
    def delay(self):
        return self._delay
    
    @delay.setter
    def delay(self, delay):
        self._delay = delay

    def reset(self): 
        self._delay = self.init_delay


class Orchestrator(): 
    def __init__(self, 
                 delay_informed_agent, 
                 safe_agent, 
                 delay_model, 
                 transition_steps : int,
                 certainty_threshold : float,  
                 noise_level : float,
                 init_beta : float = 0.0) -> None:
                
        self.delay_model = delay_model
        self.transition_manager = TransitionManager(delay_informed_agent = delay_informed_agent, 
                                                    safe_agent = safe_agent, 
                                                    transition_steps = transition_steps,
                                                    init_beta = init_beta,
                                                    increase_beta = True,
                                                    certainty_threshold = certainty_threshold,
                                                    noise_level = noise_level)
        self.last_augmented_state = None
        self.delay_pred_history = []
        self.delay_certainty_history = []

    def predict(self, augmented_state, deterministic = True):
        # If not first step, update delay model
        if self.last_augmented_state is not None:
            self.update(self.last_augmented_state, augmented_state)
        # Get delay certainty from delay model
        delay_certainty = self.delay_model.get_certainty()
        # Get prediction
        delay_prediction = self.delay_model.get_predicted_delay()
        self.delay_pred_history.append(delay_prediction)
        self.delay_certainty_history.append(delay_certainty)
        # Get action from transition manager
        action = self.transition_manager.predict(augmented_state, delay_prediction, delay_certainty, deterministic = deterministic)
        # Save augmented state
        self.last_augmented_state = augmented_state
        return action
    
    def update(self, augmented_state, new_augmented_state, action_queue = None):
        # Update delay model errors
        # base_state_shape = self.transition_manager.delay_informed_agent.env_model.observation_space.shape[0]
        # action_queue = np.concatenate([augmented_state[base_state_shape:], 
        #                                new_augmented_state[-self.transition_manager.delay_informed_agent.env_model.action_space.shape[0]:]])
        # state = augmented_state[:base_state_shape]
        # new_state = new_augmented_state[:base_state_shape]
        self.delay_model.update_errors(self.transition_manager.delay_informed_agent.env_model, augmented_state, new_augmented_state, action_queue)

    def reset(self):
        self.delay_model.reset()
        self.transition_manager.reset()
        


class TransitionManager():
    def __init__(self, 
                 delay_informed_agent : DelayInformedAgent, 
                 safe_agent, 
                 transition_steps : int = 50,
                 init_beta : float = 0.0, 
                 increase_beta : bool = True,
                 certainty_threshold : float = 0.0,
                 noise_level : float = 0.0) -> None:
        
        self.delay_informed_agent = delay_informed_agent
        self.safe_agent = safe_agent
        self.transition_steps = transition_steps
        self.beta = init_beta
        self.init_beta = init_beta
        # Transition mode
        self.increase_beta = increase_beta
        
        if self.beta == 0:
            self.increase_beta = True
        self.init_increase = self.increase_beta
        self.certainty_threshold = certainty_threshold
        self.noise_level = noise_level
        self.beta_history = []

    def predict(self, augmented_state,  delay_prediction, delay_certainty, deterministic = True):
        # Get action from delay informed agent
        self.delay_informed_agent.delay = delay_prediction
        delay_informed_action, _  = self.delay_informed_agent.predict(augmented_state, deterministic = deterministic)
        # Get safe action from safe agent
        safe_action, _ = self.safe_agent.predict(augmented_state, deterministic = True)
        self.decay_beta(delay_certainty)
        self.beta_history.append(self.beta)
        # Return combination 
        # print(f"Delay informed action: {delay_informed_action}, Safe action: {safe_action}, Beta: {self.beta}")
        combined_action = self.beta * delay_informed_action + (1-self.beta) * safe_action
        # Add noise 
        # noise = np.random.normal(0, self.noise_level * (self.delay_informed_agent.env_model.action_space.high - self.delay_informed_agent.env_model.action_space.low), combined_action.shape)
        noise = np.random.normal(0, self.noise_level, combined_action.shape)
        combined_action += noise    
        return np.clip(combined_action, self.delay_informed_agent.env_model.action_space.low,self.delay_informed_agent.env_model.action_space.high) , None
    
    def decay_beta(self, delay_certainty):
        self.trigger_transition(delay_certainty)
        if self.increase_beta : 
            self.beta = np.clip(self.beta + 1/self.transition_steps, 0, 1)
        else:
            self.beta = np.clip(self.beta - 1/self.transition_steps, 0, 1)
        
    def trigger_transition(self, delay_certainty):
        if delay_certainty >= self.certainty_threshold: 
            self.increase_beta = True
        else:
            self.increase_beta = False

    def reset(self):
        self.beta = self.init_beta
        # Transition mode
        self.increase_beta = self.init_increase
        if self.beta == 0:
            self.increase_beta = True
        self.delay_informed_agent.reset()



