# Reinforcement Learning for systems with delays
Author: Iga Pawlak


The package contains files developed in the scope of the master thesis addressing the problem of delays, particularly unknown continuous and variable ones, in Reinforcement Learning. 

### File structure
- `buffer.py`  - definition of an extended replay buffer used in by safe agents
- `delay_model.py` - delay estimator 
- `environment.py` - definition of simulation environments used in the thesis
- `env_wrappers.py` - environments wrappers to rescale action, add delay and augment the state 
- `general_utils.py` - utility functions used in other objects, to scale actions, predict the next states using the model 
- `global_config.py` - definition of default values of variables, such as maximum considered delay or default parameters of the environments
- `safe_sac.py`
