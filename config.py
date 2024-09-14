# General Environment Settings
MAX_AIRCRAFT = 10  # Maximum number of aircraft considered in the environment
MAX_FLIGHTS_PER_AIRCRAFT = 10  # Maximum number of flights per aircraft
ROWS_STATE_SPACE = 1 + MAX_AIRCRAFT  # Number of rows in the state space
COLUMNS_STATE_SPACE = 1 + 2 + 3 * MAX_FLIGHTS_PER_AIRCRAFT # Number of columns in the state space: 1 for ac id, 2 for ac unavail, 3 for each flight (id, start, end)

# Time Settings for intervals
TIMESTEP_HOURS = 1  # Length of each timestep in hours

# Hyperparameters for the simulation
LEARNING_RATE = 0.001  # Learning rate for RL agent
EPOCHS = 100  # Number of training epochs

# Reward and Penalty Values
CONFLICT_PENALTY = 500  # Penalty for conflicts
DELAY_PENALTY_PER_MINUTE = 5  # Penalty per minute of delay
RESOLVED_CONFLICT_REWARD = 5000  # Reward for resolving a conflict
EXCESS_FLIGHTS_PENALTY = 10  # Penalty per excess flight assigned to an aircraft

# Environment Settings
MIN_TURN_TIME = 0  # Minimum gap between flights for the same aircraft


# Model Settings (if you have a model-based environment)
MODEL_LAYERS = 3  # Number of layers in the model (if used)
UNITS_PER_LAYER = 128  # Number of units per layer

# Logging and Debug Settings
DEBUG_MODE = True  # Turn on/off debug mode