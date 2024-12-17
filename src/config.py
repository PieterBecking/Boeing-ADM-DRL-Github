import numpy as np

# General Environment Settings
MAX_AIRCRAFT = 6  # Maximum number of aircraft considered in the environment
MAX_FLIGHTS_PER_AIRCRAFT = 20  # Maximum number of flights per aircraft
ROWS_STATE_SPACE = 1 + MAX_AIRCRAFT  # Number of rows in the state space
COLUMNS_STATE_SPACE = 1 + 2 + 3 * MAX_FLIGHTS_PER_AIRCRAFT # Number of columns in the state space: 1 for ac id, 2 for ac unavail, 3 for each flight (id, start, end)

# Calculate the flattened action space size
ACTION_SPACE_SIZE = (MAX_AIRCRAFT + 1) * (MAX_FLIGHTS_PER_AIRCRAFT + 1)  # Number of possible actions (+1 for the zero for no action and cancellations)


# Data Generation Settings
DEPARTURE_AFTER_END_RECOVERY = 1  # how many hours after the end of the recovery period can a generated flight depart

# Time Settings for intervals
TIMESTEP_HOURS = 1  # Length of each timestep in hours


DUMMY_VALUE = -999  # Dummy value for padding


# Reward and Penalty Values
RESOLVED_CONFLICT_REWARD = 5000     # Reward for resolving a conflict
DELAY_MINUTE_PENALTY = 11.5           # Penalty per minute of delay
MAX_DELAY_PENALTY = 2500            # Maximum penalty for delay
NO_ACTION_PENALTY = 0               # Penalty for no action while conflict(s) exist
CANCELLED_FLIGHT_PENALTY = 5000    # Penalty for cancelling a flight
LAST_MINUTE_THRESHOLD = 120           # Threshold for last-minute changes in minutes
LAST_MINUTE_FLIGHT_PENALTY = 455      # Penalty for last-minute flight changes  
AHEAD_BONUS_PER_MINUTE = 0.1                # Reward for proactive flight changes
TIME_MINUTE_PENALTY = 1                 # penalty for every minute passed, each timestep cumulatively
TERMINATION_REWARD = 500                  # Reward for terminating the episode


# Environment Settings
MIN_TURN_TIME = 0  # Minimum gap between flights for the same aircraft
MIN_BREAKDOWN_PROBABILITY = 0

# Logging and Debug Settings
DEBUG_MODE = False # Turn on/off debug mode
DEBUG_MODE_TRAINING = False  # Turn on/off debug mode for training
DEBUG_MODE_REWARD = False   # Turn on/off debug mode for reward calculation
DEBUG_MODE_PRINT_STATE = False         # Turn on/off debug mode for printing state
DEBUG_MODE_CANCELLED_FLIGHT = False  # Turn on/off debug mode for cancelled flight
DEBUG_MODE_VISUALIZATION = False
DEBUG_MODE_BREAKDOWN = False  # Turn on/off debug mode for breakdowns (so rolling the dice etc)
DEBUG_MODE_ACTION = False
DEBUG_MODE_STOPPING_CRITERIA = False
DEBUG_MODE_SCHEDULING = False
DEBUG_MODE_REWARD_LAST_MINUTE_PENALTY = False  # Turn on/off debug mode for reward calculation last minute penalty
DEBUG_MODE_REWARD_RESOLVED_CONFLICTS = False