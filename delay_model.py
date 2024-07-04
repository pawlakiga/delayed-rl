from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.svm import LinearSVR, SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import copy
from general_utils import *
import global_config

from scipy.special import softmax 

class DelayEstimator():

    def __init__(self , model_type = 'linear', scaling_factor = 100, num_models = 1) -> None:
        # Initialise the regression model
        if model_type == 'linear': 
            reg = SGDRegressor(warm_start = True)
            # reg = LinearRegression()
        elif model_type == 'random-forest' : 
            reg = RandomForestRegressor(n_estimators=6)
        elif model_type == 'mlp':
            reg = MLPRegressor(hidden_layer_sizes=[256, 128, 32])
        else : 
            print('Invalid model type selected')
            return
        
        self.models = []
        for _ in range(num_models) : 
            self.models.append(copy.deepcopy(reg))
        self.scaling_factor = scaling_factor
        self.test_size = 0.2
        self.weights = np.zeros(num_models)
        self.last_predicted = 0
        self.last_predicted_error = 10.0
        self.stable = False
        self.trained = False
        
    def train_model(self, inputs, targets, test_split : bool = True):
        # Create arrays
        X = np.array(inputs)
        Y = np.array(targets)

        # Squeeze Y
        if len(Y.shape) > 2 :  
            Y = Y.squeeze(1)
        if not (Y.shape[-1] == len(self.models) or (len(Y.shape) == 1 and len(self.models) == 1 )): 
            print(f"Invalid data size - number of outputs does not match the number of models")
            return 
        
        # Scale the outputs
        Y *= self.scaling_factor

        # Set the weights for multiple output models
       
        self.weights = [np.max(Y[:,idx]) - np.min(Y[:,idx]) for idx in range(Y.shape[-1])]
        self.weights /= np.sum(self.weights)

        # Create test train split
        if test_split : 
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size)
            # Train the model for every dimension of the output
            for y_idx in range(Y.shape[-1]): 
                y_train = Y_train[:,y_idx]
                self.models[y_idx].fit(X_train, y_train)

            preds = []
            for idx in range(Y.shape[-1]): 
                preds.append(self.models[idx].predict(X_test))

            mses = [mean_squared_error(Y_test[:,idx], preds[idx]) for idx in range(Y.shape[-1])]
            scores = [self.models[idx].score(X_test, Y_test[:,idx]) for idx in range(Y.shape[-1])]
        else: 
            for y_idx in range(Y.shape[-1]): 
                y_train = Y[:,y_idx]
                self.models[y_idx].fit(X, y_train)
            preds = []
            for idx in range(Y.shape[-1]): 
                preds.append(self.models[idx].predict(X))

            mses = [mean_squared_error(Y[:,idx], preds[idx]) for idx in range(Y.shape[-1])]
            scores = [self.models[idx].score(X, Y[:,idx]) for idx in range(Y.shape[-1])]

        self.trained = True
        return mses, scores
    
    def predict(self, X) : 
        preds = []
        for idx in range(len(self.models)): 
                preds.append(self.models[idx].predict(X)/self.scaling_factor)
        return preds

    def get_predicted_delay(self, env) : 

        action = env.action_space.sample()
        delta_t = [self.models[idx].predict(np.concatenate([env.state, action]).reshape(1, env.state.shape[0] + action.shape[0]))/self.scaling_factor  for idx in range(len(self.models))]
        # print(f"Deltas are {delta_t}")
        # print(f"Components are {[env.unwrapped.get_action_component(delayed_action) - env.unwrapped.get_action_component(action) for delayed_action in env.actions_queue]}")
        diffs = [np.sqrt(np.square(np.array(delta_t).squeeze(-1) 
                                   - env.unwrapped.get_action_component(delayed_action) 
                                   + env.unwrapped.get_action_component(action))) 
                                   for delayed_action in env.actions_queue]
        # print(f"Diff is {diffs}")
        if len(self.models) == 1 :
            agg = diffs
            argmin = np.argmin(agg)
            # if np.abs(env.unwrapped.get_action_component(env.actions_queue[argmin]) - env.unwrapped.get_action_component(action) - self.last_predicted_error) <= 5e-2:
            #     self.stable = True
            #     return self.last_predicted_delay
            self.last_predicted_error = env.unwrapped.get_action_component(env.actions_queue[argmin]) - env.unwrapped.get_action_component(action)

        else : 
            agg = np.matmul(np.array(diffs), self.weights)
            # if np.abs(np.matmul(env.unwrapped.get_action_component(env.actions_queue[argmin]), self.weights) - self.last_predicted_error) <= 5e-2:
            #     self.stable = True
            #     return self.last_predicted_delay
            self.last_predicted_error =  np.matmul(env.unwrapped.get_action_component(env.actions_queue[argmin]), self.weights)
        
        self.stable = False

        self.last_predicted_delay = len(env.actions_queue) - argmin
        return len(env.actions_queue) - argmin

class SoftmaxDelayEstimator(DelayEstimator): 
    def __init__(self, model_type='linear', scaling_factor=100, num_models=1, max_delay = 50) -> None:
        super().__init__(model_type, scaling_factor, num_models)
        self.max_delay = max_delay
        self.probs = np.zeros(max_delay + 1)

    def accumulate_diffs(self, env, state, action, actions_queue, target): 
        if not self.trained: 
            delta_t = target
        else : 

            delta_t = [self.models[idx].predict(np.concatenate([state, action]).reshape(1, state.shape[0] + action.shape[0]))/self.scaling_factor  for idx in range(len(self.models))]

        diffs = [np.sqrt(np.square(np.array(delta_t).squeeze() 
                                    - env.unwrapped.get_action_component(delayed_action) 
                                    + env.unwrapped.get_action_component(action))) 
                                    for delayed_action in actions_queue]
        if len(self.models) == 1 :
            # print(np.array(diffs).shape)
            agg = np.array(diffs).squeeze(-1)
        else: 
            # print(agg.shape)
            agg = np.matmul(np.array(diffs), self.weights)#.squeeze(-1)
            # print(agg.shape)
        
        self.probs = self.probs * 0.99 + agg

    def get_predicted_delay(self, env):

        selected = np.random.choice(range(self.max_delay+1), p = softmax(- self.scaling_factor * self.probs))

        self.stable = False
        self.last_predicted_delay = len(self.probs) - selected  -1
        return len(self.probs) - selected  -1


class IterativeEstimator():
    """Delay estimator using the model error to predict the delay"""
    def __init__(self,
                 state_shape, 
                 max_delay = global_config.MAX_DELAY, 
                 decay : float = 0.99, 
                 scaling_factor : int = 100,
                 useful_axes : list = None) -> None:
        """
        Initialisation
        :param state_shape: Shape of the state of the environment
        :param max_delay: Maximum delay to consider
        :param decay: Decay factor for the model errors to give more importance to new samples
        :param scaling_factor: Scaling factor for the softmax function

        """
        self.max_delay = max_delay
        self.model_errors = np.zeros((max_delay + 1, state_shape))
        self.decay = decay
        self.scaling_factor = scaling_factor
        self.last_dist = np.ones(max_delay + 1) / (max_delay + 1)    
        self.useful_axes = useful_axes

    def update_errors(self, env_model, state, new_state, action_queue = None): 
        """
        Function to update the model errors with the new state
        :param env: Environment, delayed real process, can be augmented
        :param env_model: Environment model
        :param state: Current state
        :param action_queue: Action queue of the delayed environment, should contain also the current action?
        :param new_state: The true new state of the environment
        """
        # If augmented environment
        if state.shape[0] > env_model.observation_space.shape[0]: 
            action_queue = np.concatenate([state[env_model.observation_space.shape[0]:], new_state[-env_model.action_space.shape[0]:]])
            state = state[:env_model.observation_space.shape[0]]
            new_state = new_state[:env_model.observation_space.shape[0]]

        elif action_queue is None:
            # print(f"Action queue is None, cannot update errors")
            return
        # print(f"Action queue is {action_queue}")
        # Reshape the action queue if needed for multi-dimensional actions
        if len(action_queue.shape) == 1 and env_model.action_space.shape[0] > 1: 
            
            action_queue = np.array(action_queue).reshape(action_queue.shape[0]//env_model.action_space.shape[0], env_model.action_space.shape[0])

        new_errors = []
        for delay in range(0, self.max_delay+1): 
            # Predict the next state from delayed action
            # Get old state 
    
            pred_state = predict_next_state(env_model=env_model, cur_state=state, action=action_queue[len(action_queue) - delay - 1])
            new_pred_state = env_model.state
            env_model.reset()
            env_model.set_state(new_state)
            new_true_state = env_model.state
            # print(f"Pred state {pred_state} from {state}")
            # Append the error
            new_errors.append(np.sqrt(np.square(new_true_state - new_pred_state)))
        # print(f"New errors min {np.argmin(new_errors, axis = 0)} are {new_errors}")
        # Update the model errors
        # print(f"Updating errors from {self.model_errors} with shape {self.model_errors.shape}")
        self.model_errors = self.model_errors * self.decay +  np.array(new_errors).reshape(self.model_errors.shape)

    def get_predicted_delay(self): 
        """
        Function to get the predicted delay
        """
        probs = self.get_probs()
        # return np.random.choice(range(self.max_delay+1), p = probs)
        # return np.argmin(self.model_errors)
        # print(f"Pred delay is {np.argmax(probs)} with probs {probs}")
        return np.argmax(probs)

    def get_certainty(self):
        """
        Certainty of the delay prediction, calculated as the max probability
        """
        # If we have multiple dimensions, use sum 
        probs = self.get_probs()

        max_prob = np.max(probs)
        
        return max_prob
    
    def get_probs(self): 
        # If we have multiple dimensions, use sum 
        if len(self.model_errors.shape) > 1 : 
            if self.useful_axes is not None: 
                probs = softmax(-self.scaling_factor * np.sum(self.model_errors[:,self.useful_axes], axis = -1))
            else:
                probs = softmax(-self.scaling_factor * np.sum(self.model_errors, axis = -1))
        else :
            probs = softmax(-self.scaling_factor * self.model_errors)

        return probs
    
    def predict(self, env): 
        """
        Function to predict the delay and certainty
        :param env: Environment, delayed real process
        """
        return self.get_predicted_delay(env), self.get_certainty()

    def reset(self): 
        """
        Function to reset the model errors
        """
        self.model_errors = np.zeros((self.max_delay + 1, self.model_errors.shape[-1]))
        self.last_dist = np.ones(self.max_delay + 1) / (self.max_delay + 1)
