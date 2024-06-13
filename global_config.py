# Global config
MAX_DELAY = 10
NUM_DELAYS = 11

# DEFAULT PARAMETERS FOR EVERY ENVIRONMENT
# Linear Velocity 
linear_velocity_params = {
    'm': 1, 
    'd': -0.01, 
    'Ts': 0.1, 
    'max_episode_len': 120
}

# Non Linear Velocity
non_linear_velocity_params = {
    'm': 1, 
    'd': 0.01, 
    'd_nl': -0.008, 
    'Ts': 0.1, 
    'max_episode_len': 120
}

# Position
position_params = {
    'm': 0.5, 
    'd': 0.5, 
    'Ts': 0.1, 
    'max_episode_len': 120
}

# Robot Steer
robot_steer_params = {
    'm': 0.5, 
    'd': 0.5, 
    'Ts': 0.3, 
    'max_episode_len': 120
}
# Spherical tank
spherical_tank_params = {
    'cp' : 1.3,
    'radius' : 2,
    'Ts' : 0.1,
    'max_episode_len' : 200
}
# Pendulum
pendulum_params = {
    'max_episode_len' : 200
}


DEFAULT_PARAMS = {
    'LinearVelocity': linear_velocity_params,
    'NonLinearVelocity': non_linear_velocity_params,
    'Position': position_params,
    'RobotSteer': robot_steer_params,
    'SphericalTank': spherical_tank_params,
    'PendulumEnv': pendulum_params
}




PLOT_COLORS = ['#3399FF', '#C768A1', '#FF8000', '#00CC66', '#CC0000']

# Models path
MODELS_PATH = 'models'

# Log path 
LOG_PATH = 'logs/train'
TEST_LOG_PATH = 'logs/test'

COLORS = {
    'cold_orange' : '#FFA07A',
    'warm_blue' : '#5d7aff', 
    'warm_green' : '#4bc64d',
    'warm_yellow' : '#f0cc6b',
    'warm_purple' : '#bb7aff',
    'cold_red' : '#dd405c',
    'mid_green' : '#54a048',
    'dark_pink' : '#d360a6',
    'dark_mint' : '#51ae9a',
    'dark_orange' : '#c44b0b',

}
