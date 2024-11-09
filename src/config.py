# General Environment Settings
MAX_AIRCRAFT = 3  # Maximum number of aircraft considered in the environment
MAX_FLIGHTS_PER_AIRCRAFT = 12  # Maximum number of flights per aircraft
ROWS_STATE_SPACE = 1 + MAX_AIRCRAFT  # Number of rows in the state space
COLUMNS_STATE_SPACE = 1 + 2 + 3 * MAX_FLIGHTS_PER_AIRCRAFT # Number of columns in the state space: 1 for ac id, 2 for ac unavail, 3 for each flight (id, start, end)

# Time Settings for intervals
TIMESTEP_HOURS = 1  # Length of each timestep in hours


DUMMY_VALUE = -999  # Dummy value for padding

# Reward and Penalty Values
RESOLVED_CONFLICT_REWARD = 1000     # Reward for resolving a conflict
DELAY_MINUTE_PENALTY = 6           # Penalty per minute of delay
MAX_DELAY_PENALTY = 1000000            # Maximum penalty for delay
NO_ACTION_PENALTY = 5               # Penalty for no action while conflict(s) exist
CANCELLED_FLIGHT_PENALTY = 1000    # Penalty for cancelling a flight

# Environment Settings
MIN_TURN_TIME = 0  # Minimum gap between flights for the same aircraft


# Logging and Debug Settings
DEBUG_MODE = True  # Turn on/off debug mode
DEBUG_MODE_TRAINING = True  # Turn on/off debug mode for training
DEBUG_MODE_REWARD = False  # Turn on/off debug mode for reward calculation
DEBUG_MODE_PRINT_STATE = True  # Turn on/off debug mode for printing state
DEBUG_MODE_CANCELLED_FLIGHT = False  # Turn on/off debug mode for cancelled flight
DEBUG_MODE_VISUALIZATION = True

# Data Generation Settings
DEPARTURE_AFTER_END_RECOVERY = 1  # how many hours after the end of the recovery period can a generated flight depart



# Constants for breakdown probabilities
BREAKDOWN_PROBABILITY = 0.9  # Probability of aircraft breaking down during the day
BREAKDOWN_DURATION = 60  # Duration of breakdown in minutes
INDICATION_TIME_BEFORE_BREAKDOWN = 120  # Time before breakdown to provide indication to the agent in minutes

MIN_TURN_TIME = 0  # Minimum turnaround time in minutes