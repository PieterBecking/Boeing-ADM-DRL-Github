import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config import *
from scripts.utils import *
import time
import random

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type):
        """Initializes the AircraftDisruptionEnv class.

        Args:
            aircraft_dict (dict): Dictionary containing aircraft information.
            flights_dict (dict): Dictionary containing flight information.
            rotations_dict (dict): Dictionary containing rotation information.
            alt_aircraft_dict (dict): Dictionary containing alternative aircraft information.
            config_dict (dict): Dictionary containing configuration information.
            env_type (str): Type of environment ('myopic' or 'proactive').
        """
        super(AircraftDisruptionEnv, self).__init__()
        
        # Store the environment type ('myopic' or 'proactive')
        self.env_type = env_type  
        
        # Constants for environment configuration
        self.max_aircraft = MAX_AIRCRAFT
        self.columns_state_space = COLUMNS_STATE_SPACE + 1  # Adjust for new format (probability + start/end times + flights)
        self.rows_state_space = ROWS_STATE_SPACE

        self.config_dict = config_dict

        # Define the recovery period based on provided configuration
        start_date = config_dict['RecoveryPeriod']['StartDate']
        start_time = config_dict['RecoveryPeriod']['StartTime']
        end_date = config_dict['RecoveryPeriod']['EndDate']
        end_time = config_dict['RecoveryPeriod']['EndTime']
        self.start_datetime = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
        self.end_datetime = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
        self.timestep = timedelta(hours=TIMESTEP_HOURS)

        # Aircraft information and indexing
        self.aircraft_ids = list(aircraft_dict.keys())
        self.aircraft_id_to_idx = {aircraft_id: idx for idx, aircraft_id in enumerate(self.aircraft_ids)}

        self.conflicted_flights = {}  # Tracks flights in conflict due to past departure and prob == 1.0
    

        # Flight information and indexing
        # if flights_dict is empty, flights_dict is empty
        # print(f"*****flights_dict: {flights_dict}")
        if flights_dict is None:
            flights_dict = {}  # Initialize as empty dict if None

        if flights_dict:
            self.flight_ids = list(flights_dict.keys())
            self.flight_id_to_idx = {flight_id: idx for idx, flight_id in enumerate(self.flight_ids)}
        else:
            self.flight_ids = []
            self.flight_id_to_idx = {}

        # Filter out flights with '+' in DepTime (next day flights)
        this_day_flights = [flight_info for flight_info in flights_dict.values() if '+' not in flight_info['DepTime']]

        # Determine the earliest possible event in the environment
        self.earliest_datetime = min(
            min(datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + flight_info['DepTime'], '%d/%m/%y %H:%M') for flight_info in this_day_flights),
            self.start_datetime
        )

        # Define observation and action spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.rows_state_space * self.columns_state_space,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.action_space.n,),
                dtype=np.uint8
            )
        })

        # Action space: select a flight and an aircraft
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Store the dictionaries as class attributes
        self.alt_aircraft_dict = alt_aircraft_dict

        self.rotations_dict = rotations_dict
        self.flights_dict = flights_dict
        self.aircraft_dict = aircraft_dict

        # Deep copies of initial data to reset the environment later
        self.initial_aircraft_dict = copy.deepcopy(aircraft_dict)
        self.initial_flights_dict = copy.deepcopy(flights_dict)
        self.initial_rotations_dict = copy.deepcopy(rotations_dict)
        self.initial_alt_aircraft_dict = copy.deepcopy(alt_aircraft_dict)

        # Track environment state related to delays and conflicts
        self.environment_delayed_flights = {}   # Tracks delays for flights {flight_id: delay_minutes}
        self.penalized_delays = {}           # Set of penalized delays
        
        self.penalized_conflicts = set()        # Set of penalized conflicts
        self.resolved_conflicts = set()         # Set of resolved conflicts
        self.penalized_cancelled_flights = set()  # To keep track of penalized cancelled flights

        self.cancelled_flights = set()

        # Initialize empty containers for breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Initialize a dictionary to store unavailabilities
        self.unavailabilities_dict = {}

        # Initialize the environment state without generating probabilities
        self.current_datetime = self.start_datetime
        self.state = self._get_initial_state()

    def _get_initial_state(self):
        """Initializes the state matrix for the environment.
        
        Returns:
            np.ndarray: The initial state matrix.
        """

        # Initialize state matrix with NaN values
        state = np.full((self.rows_state_space, self.columns_state_space), np.nan)

        # Calculate current time and remaining recovery period in minutes
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        time_until_end_minutes = (self.end_datetime - self.current_datetime).total_seconds() / 60

        # Insert the current_time_minutes and time_until_end_minutes in the first row
        for i in range(0, self.columns_state_space // 2, 2):  # Start at 0 and step by 2 for the half of the columns
            if i + 1 < self.columns_state_space:  # Check to ensure i+1 is in range
                state[0, i] = current_time_minutes  # Current time
                state[0, i + 1] = time_until_end_minutes  # Time until end of recovery period

        # self.something_happened = False

        # List to keep track of flights to remove from dictionaries
        flights_to_remove = set()

        # Set to collect actual flights in state space
        active_flights = set()

        # Populate state matrix with aircraft and flight information
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break  # Only process up to the maximum number of aircraft

            # Store aircraft index instead of ID
            state[idx + 1, 0] = idx + 1  # Use numerical index instead of string ID

            # Check for predefined unavailabilities and assign actual probability         
            if aircraft_id in self.alt_aircraft_dict:
                unavails = self.alt_aircraft_dict[aircraft_id]
                if not isinstance(unavails, list):
                    unavails = [unavails]
                breakdown_probability = unavails[0].get('Probability', 1.0)

                # Get earliest start and latest end time
                start_times = []
                end_times = []
                for unavail_info in unavails:
                    unavail_start_time = datetime.strptime(unavail_info['StartDate'] + ' ' + unavail_info['StartTime'], '%d/%m/%y %H:%M')
                    unavail_end_time = datetime.strptime(unavail_info['EndDate'] + ' ' + unavail_info['EndTime'], '%d/%m/%y %H:%M')
                    start_times.append((unavail_start_time - self.earliest_datetime).total_seconds() / 60)
                    end_times.append((unavail_end_time - self.earliest_datetime).total_seconds() / 60)

                if start_times:
                    unavail_start_minutes = min(start_times)
                    unavail_end_minutes = max(end_times)

            # Check for uncertain breakdowns
            elif aircraft_id in self.uncertain_breakdowns:
                breakdown_info = self.uncertain_breakdowns[aircraft_id][0]  # Get first breakdown
                breakdown_probability = breakdown_info['Probability']  # Use existing probability
                unavail_start_minutes = (breakdown_info['StartTime'] - self.earliest_datetime).total_seconds() / 60
                unavail_end_minutes = (breakdown_info['EndTime'] - self.earliest_datetime).total_seconds() / 60

            else:
                # No unavailability, set default values
                breakdown_probability = 0.0
                unavail_start_minutes = np.nan
                unavail_end_minutes = np.nan

            # Store the unavailability information in the unavailabilities dictionary
            self.unavailabilities_dict[aircraft_id] = {
                'Probability': breakdown_probability,
                'StartTime': unavail_start_minutes,
                'EndTime': unavail_end_minutes
            }

            # In the myopic env, the info for uncertain breakdowns is not shown
            if breakdown_probability != 1.0 and self.env_type == 'myopic':
                breakdown_probability = np.nan  # Set to NaN if not 1.00
                unavail_start_minutes = np.nan
                unavail_end_minutes = np.nan

            # In the proactive env, the info for unrealized breakdowns is also not shown
            if np.isnan(breakdown_probability):
                breakdown_probability = np.nan  # Set to NaN if not 1.00
                unavail_start_minutes = np.nan
                unavail_end_minutes = np.nan

            # Store probability and unavailability times
            state[idx + 1, 1] = breakdown_probability
            state[idx + 1, 2] = unavail_start_minutes
            state[idx + 1, 3] = unavail_end_minutes

            # Gather and store flight times (starting from column 4)
            flight_times = []
            for flight_id, rotation_info in self.rotations_dict.items():
                if flight_id in self.flights_dict and rotation_info['Aircraft'] == aircraft_id:
                    flight_info = self.flights_dict[flight_id]
                    dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
                    arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)

                    dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                    arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60

                    # Exclude flights that have already departed and are in conflict
                    if dep_time_minutes < current_time_minutes:
                        # Flight has already departed
                        if breakdown_probability == 1.00 and not np.isnan(unavail_start_minutes) and not np.isnan(unavail_end_minutes):
                            # There is an unavailability with prob == 1.00
                            # Check if the flight overlaps with the unavailability
                            if dep_time_minutes < unavail_end_minutes and arr_time_minutes > unavail_start_minutes:
                                if DEBUG_MODE_CANCELLED_FLIGHT:
                                    print(f"REMOVING FLIGHT {flight_id} DUE TO UNAVAILABILITY AND PAST DEPARTURE")
                                # Flight is in conflict with unavailability
                                flights_to_remove.add(flight_id)
                                continue

                    flight_times.append((flight_id, dep_time_minutes, arr_time_minutes))
                    active_flights.add(flight_id)  # Add to active flights set

            # Sort flights by departure time
            flight_times.sort(key=lambda x: x[1])

            # Store flight information starting from column 4
            for i, (flight_id, dep_time, arr_time) in enumerate(flight_times):
                col_start = 4 + (i * 3)
                if col_start + 2 < self.columns_state_space:
                    state[idx + 1, col_start] = flight_id
                    state[idx + 1, col_start + 1] = dep_time
                    state[idx + 1, col_start + 2] = arr_time

        # Update flight_id_to_idx with only the active flights
        self.flight_id_to_idx = {
            flight_id: idx for idx, flight_id in enumerate(sorted(active_flights))
        }

        # Remove past flights from dictionaries
        for flight_id in flights_to_remove:
            self.remove_flight(flight_id)

        return state

    def process_observation(self, state):
        """Processes the observation by applying a mask and flattening the state and mask.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            dict: A dictionary containing the processed state and action mask.
        """
        # Create a mask where 1 indicates valid values, 0 indicates NaN
        mask = np.where(np.isnan(state), 0, 1)
        # Replace NaN with the dummy value
        state = np.nan_to_num(state, nan=DUMMY_VALUE)
        # Flatten both state and mask
        state_flat = state.flatten()
        mask_flat = mask.flatten()
        
        # Use get_action_mask to generate the action mask
        action_mask = self.get_action_mask()

        # Create the observation dictionary
        obs_with_mask = {
            'state': state.flatten(),
            'action_mask': action_mask
        }
        return obs_with_mask


    
    def fix_state(self, state):
        # Go over all starttimes and endtimes (columns 2 and 3 for unavailabilities and then for flights: 5, 6, 8, 9, 11, 12, ...)
        # If endtime is smaller than starttime, add 1440 minutes to endtime
        for i in range(1, self.rows_state_space):
            if not np.isnan(state[i, 2]) and not np.isnan(state[i, 3]) and state[i, 2] > state[i, 3]:
                state[i, 3] += 1440
            for j in range(4, self.columns_state_space - 2, 3):
                if not np.isnan(state[i, j + 1]) and not np.isnan(state[i, j + 2]) and state[i, j + 1] > state[i, j + 2]:
                    state[i, j + 2] += 1440

    def remove_flight(self, flight_id):
        """Removes the specified flight from the dictionaries."""
        # Remove from flights_dict
        if flight_id in self.flights_dict:
            del self.flights_dict[flight_id]

        # Remove from rotations_dict
        if flight_id in self.rotations_dict:
            del self.rotations_dict[flight_id]

        # Mark the flight as canceled
        self.cancelled_flights.add(flight_id)


    def step(self, action_index):
        """Executes a step in the environment based on the provided action.

        This function processes the action taken by the agent, checks for conflicts, updates the environment state,
        and returns the new state, reward, termination status, truncation status, and additional info.

        Args:
            action (tuple or list): The action to be taken by the agent.

        Returns:
            tuple: A tuple containing the processed state, reward, terminated flag, truncated flag, and additional info.
        """

        # Fix the state before processing the action
        self.fix_state(self.state)


        # Print the current state if in debug mode
        if DEBUG_MODE_PRINT_STATE:
            print_state_nicely_proactive(self.state)
            print("")

        # Extract the action values from the action
        flight_action, aircraft_action = self.map_index_to_action(action_index)

        # Check if the flight action is valid
        if DEBUG_MODE_ACTION:
            print(f"***Flight action: {flight_action}")
            print(f"***self.flight_id_to_idx.keys(): {self.flight_id_to_idx.keys()}")
        
        if flight_action != 0:
            # Check if the flight_action exists in our valid flight IDs
            if flight_action not in self.flight_id_to_idx.keys():
                raise ValueError(f"Invalid flight action: {flight_action}")

        # Validate the action
        self.validate_action(flight_action, aircraft_action)

        # Print the processed action and chosen action
        if DEBUG_MODE:
            print(f"Processed action: {action_index} of type: {type(action_index)}")
            print(f"Chosen action: flight {flight_action}, aircraft {aircraft_action}")

        # Initialize info dictionary
        info = {}

        # Get pre-action conflicts
        pre_action_conflicts = self.get_current_conflicts()
        unresolved_uncertainties = self.get_unresolved_uncertainties()

        # print("-----1-----")
        # print(f"pre_action_conflicts: {pre_action_conflicts}")

        # Process uncertainties before handling flight operations
        self.process_uncertainties()

        if len(pre_action_conflicts) == 0 and len(unresolved_uncertainties) == 0:
            # Handle the case when there are no conflicts
            # print("-----2-----")
            processed_state, reward, terminated, truncated, info = self.handle_no_conflicts(flight_action, aircraft_action)
        else:
            # print("-----3-----")
            # Resolve the conflict based on the action
            processed_state, reward, terminated, truncated, info = self.handle_flight_operations(flight_action, aircraft_action, pre_action_conflicts)

        # Update the processed state after processing uncertainties
        processed_state = self.process_observation(self.state)

        return processed_state, reward, terminated, truncated, info

    def extract_action_value(self, action):
        """Extracts the flight and aircraft action values from the flattened action.

        Args:
            action (int): The flattened action index.

        Returns:
            tuple: The flight action and aircraft action values.
        """
        if action < 0 or action >= ACTION_SPACE_SIZE:
            raise ValueError("Invalid action index")

        flight_action = action // (len(self.aircraft_ids) + 1)  # Integer division to get flight action
        aircraft_action = action % (len(self.aircraft_ids) + 1)  # Modulus to get aircraft action

        return flight_action, aircraft_action

    def validate_action(self, flight_action, aircraft_action):
        """Validates the provided action values.

        Args:
            flight_action (int): The flight action value to be validated.
            aircraft_action (int): The aircraft action value to be validated.

        Raises:
            ValueError: If the action is not valid.
        """
        # Get valid actions
        valid_flight_actions = self.get_valid_flight_actions()
        valid_aircraft_actions = self.get_valid_aircraft_actions()

        # Check if flight_action is valid
        if flight_action not in valid_flight_actions:
            raise ValueError(f"Invalid flight action: {flight_action}")

        # Check if aircraft_action is valid
        if aircraft_action not in valid_aircraft_actions:
            raise ValueError(f"Invalid aircraft action: {aircraft_action}")

        # No action case
        if flight_action == 0:
            # Treat as 'no action'
            return


    def process_uncertainties(self):
        """Processes breakdown uncertainties directly from the state space.

        Probabilities evolve stochastically over time but are capped at [0.05, 0.95].
        When the current datetime + timestep reaches the breakdown start time,
        resolve the uncertainty fully to 0.00 or 1.00 by rolling the dice.
        """
        if DEBUG_MODE:
            print(f"Current datetime: {self.current_datetime}")

        # Iterate over each aircraft's row in the state space to check for unresolved breakdowns
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            # Get probability, start, and end time from the state space
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            start_minutes = self.unavailabilities_dict[aircraft_id]['StartTime']
            end_minutes = self.unavailabilities_dict[aircraft_id]['EndTime']

            # Only process unresolved breakdowns
            if prob != 0.00 and prob != 1.00:
                # Check for valid start and end times
                if not np.isnan(start_minutes) and not np.isnan(end_minutes) and not np.isnan(prob):
                    breakdown_start_time = self.earliest_datetime + timedelta(minutes=start_minutes)
                else:
                    # No start or end time, skip processing
                    continue

                # Apply random progression to probability
                random_variation = np.random.uniform(-0.05, 0.05)  # Random adjustment
                bias = 0.05 * (1 - prob) if prob > 0.5 else -0.05 * prob  # Bias toward extremes
                progression = random_variation + bias
                new_prob = prob + progression

                # Cap probabilities at [0.05, 0.95]
                new_prob = max(0.05, min(0.95, new_prob))
                self.unavailabilities_dict[aircraft_id]['Probability'] = new_prob

                if self.env_type == "proactive":
                    self.state[idx + 1, 1] = new_prob

                if DEBUG_MODE:
                    print(f"Aircraft {aircraft_id}: Probability updated from {prob:.2f} to {new_prob:.2f}")

                if self.current_datetime + self.timestep >= breakdown_start_time:
                    if DEBUG_MODE_BREAKDOWN:
                        print(f"Rolling the dice for breakdown with updated probability {new_prob} starting at {breakdown_start_time}")

                    # Roll the dice
                    if np.random.rand() < new_prob:
                        if DEBUG_MODE_BREAKDOWN:
                            print(f"Breakdown confirmed for aircraft {aircraft_id} with probability {new_prob:.2f}")
                        self.state[idx + 1, 1] = 1.00  # Confirm the breakdown
                        self.state[idx + 1, 2] = start_minutes  # Update start time
                        self.state[idx + 1, 3] = end_minutes
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 1.00
                    else:
                        if DEBUG_MODE_BREAKDOWN:
                            print(f"Breakdown not occurring for aircraft {aircraft_id}")
                        
                        if self.env_type == "proactive":
                            self.state[idx + 1, 1] = 0.00  # Resolve as no breakdown
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 0.00

                    # Update alt_aircraft_dict if necessary
                    if aircraft_id in self.alt_aircraft_dict:
                        if isinstance(self.alt_aircraft_dict[aircraft_id], dict):
                            self.alt_aircraft_dict[aircraft_id] = [self.alt_aircraft_dict[aircraft_id]]
                        elif isinstance(self.alt_aircraft_dict[aircraft_id], str):
                            # Handle case where entry is a string
                            self.alt_aircraft_dict[aircraft_id] = [{
                                'StartDate': breakdown_start_time.strftime('%d/%m/%y'),
                                'StartTime': breakdown_start_time.strftime('%H:%M'),
                                'EndDate': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%d/%m/%y'),
                                'EndTime': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%H:%M'),
                                'Probability': self.state[idx + 1, 1]  # Updated probability
                            }]
                        for breakdown_info in self.alt_aircraft_dict[aircraft_id]:
                            breakdown_info['Probability'] = self.state[idx + 1, 1]



    def handle_no_conflicts(self, flight_action, aircraft_action):
        """Handles the case when there are no conflicts in the current state.

        This function updates the current datetime, checks if the episode is terminated,
        updates the state, and returns the appropriate outputs.
        """

        # store the departure time of the flight that is being acted upon (before the action is taken)
        if flight_action != 0:
            original_flight_action_departure_time = self.flights_dict[flight_action]['DepTime']
        else:
            original_flight_action_departure_time = None

        next_datetime = self.current_datetime + self.timestep
        if next_datetime >= self.end_datetime:
            terminated, reason = self._is_done()
            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")
                processed_state = self.process_observation(self.state)
                truncated = False
                reward = 0  # Assuming zero reward when episode ends without conflicts
                return processed_state, reward, terminated, truncated, {}

        self.current_datetime = next_datetime
        self.state = self._get_initial_state()
        

        # Call _calculate_reward even when there are no conflicts
        reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time)

        # Since there are no conflicts, return the new state with zero reward
        terminated, reason = self._is_done()
        truncated = False
        processed_state = self.process_observation(self.state)
        reward = 0  # Assuming zero reward when there are no conflicts

        if terminated:
            if DEBUG_MODE_STOPPING_CRITERIA:
                print(f"Episode ended: {reason}")

        return processed_state, reward, terminated, truncated, {}


    def handle_flight_operations(self, flight_action, aircraft_action, pre_action_conflicts):
        """
        Handles flight operation decisions and resolves conflicts.

        This method processes the agent's actions to either maintain the current state, cancel a flight, 
        or reschedule it to a different aircraft. It updates the system state accordingly, resolves 
        any conflicts, and computes the rewards based on the chosen action.

        Args:
            flight_action (int): The index of the flight action chosen by the agent. 
                                Use 0 to skip the flight operation.
            aircraft_action (int): The index of the aircraft action chosen by the agent. 
                                Use 0 to cancel the flight.
            pre_action_conflicts (set): The set of conflicts present before the action is taken.

        Returns:
            tuple: A tuple containing:
                - processed_state: The updated system state after the action.
                - reward (float): The reward value calculated based on the resolved conflicts.
                - terminated (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was prematurely stopped.
                - info (dict): Additional diagnostic information.
        """

        # store the departure time of the flight that is being acted upon (before the action is taken)
        if flight_action != 0:
            original_flight_action_departure_time = self.flights_dict[flight_action]['DepTime']
        else:
            original_flight_action_departure_time = None

        if flight_action == 0:
            # No action taken
            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time)

            terminated, reason = self._is_done()
            truncated = False

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state = self.process_observation(self.state)
            return processed_state, reward, terminated, truncated, {}
        elif aircraft_action == 0:
            # Cancel the flight
            self.cancel_flight(flight_action)
            if DEBUG_MODE_CANCELLED_FLIGHT:
                print(f"Cancelled flight {flight_action}")

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time)

            terminated, reason = self._is_done()
            truncated = False

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state = self.process_observation(self.state)
            return processed_state, reward, terminated, truncated, {}
        else:
            # Reschedule the flight to the selected aircraft
            selected_flight_id = flight_action
            selected_aircraft_id = self.aircraft_ids[aircraft_action - 1]

            # Check if the flight is in rotations_dict
            if selected_flight_id not in self.rotations_dict:
                # Flight has been canceled or does not exist
                print(f"Flight {selected_flight_id} has been canceled or does not exist.")
                
                # Proceed to next timestep
                next_datetime = self.current_datetime + self.timestep
                self.current_datetime = next_datetime
                self.state = self._get_initial_state()
                
                # Handle this case appropriately
                reward = self._calculate_reward(pre_action_conflicts, pre_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time)
                terminated, reason = self._is_done()
                truncated = False
                processed_state = self.process_observation(self.state)
                return processed_state, reward, terminated, truncated, {}

            current_aircraft_id = self.rotations_dict[selected_flight_id]['Aircraft']

            if selected_aircraft_id == current_aircraft_id:
                # Delay the flight by scheduling it on the same aircraft
                # Get unavailability end time for the aircraft
                aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1  # Adjust for state index
                unavail_end = self.state[aircraft_idx, 3]  # Unavailability end time in minutes from earliest_datetime

                if np.isnan(unavail_end):
                    # No unavailability end time, cannot proceed
                    # In this case, set unavail_end to current time
                    unavail_end = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

                unavail_end_datetime = self.earliest_datetime + timedelta(minutes=unavail_end)
                new_dep_time = unavail_end_datetime + timedelta(minutes=MIN_TURN_TIME)
                new_dep_time_minutes = (new_dep_time - self.earliest_datetime).total_seconds() / 60

                # Schedule the flight on the same aircraft starting from new_dep_time_minutes
                # The schedule_flight_on_aircraft method will handle adjusting subsequent flights
                self.schedule_flight_on_aircraft(
                    selected_aircraft_id, selected_flight_id, new_dep_time_minutes, current_aircraft_id, None
                )

            else:
                # Swap the flight to the selected aircraft
                # Update rotations_dict
                self.rotations_dict[selected_flight_id]['Aircraft'] = selected_aircraft_id

                # Remove flight from current aircraft's schedule
                current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1
                for j in range(4, self.columns_state_space - 2, 3):
                    if self.state[current_aircraft_idx, j] == selected_flight_id:
                        self.state[current_aircraft_idx, j] = np.nan
                        self.state[current_aircraft_idx, j + 1] = np.nan
                        self.state[current_aircraft_idx, j + 2] = np.nan
                        break

                # Schedule flight on new aircraft
                # Get dep and arr times
                flight_info = self.flights_dict[selected_flight_id]
                dep_time_str = flight_info['DepTime']
                arr_time_str = flight_info['ArrTime']
                dep_time = parse_time_with_day_offset(dep_time_str, self.start_datetime)
                arr_time = parse_time_with_day_offset(arr_time_str, self.start_datetime)
                dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60

                self.schedule_flight_on_aircraft(selected_aircraft_id, selected_flight_id, dep_time_minutes, current_aircraft_id, arr_time_minutes)

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time)

            terminated, reason = self._is_done()
            truncated = False

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state = self.process_observation(self.state)
            return processed_state, reward, terminated, truncated, {}

    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, current_aircraft_id, arr_time=None, delayed_flights=None):
        """Schedules a flight on an aircraft.

        This function schedules a flight on an aircraft, taking into account unavailability periods and conflicts with existing flights.
        It updates the state and flights dictionary accordingly.
        
        Args:
            aircraft_id (str): The ID of the aircraft to schedule the flight on.
            flight_id (str): The ID of the flight to schedule.
            dep_time (float): The departure time of the flight in minutes from earliest_datetime.
            current_aircraft_id (str): The ID of the current aircraft.
            arr_time (float, optional): The arrival time of the flight in minutes from earliest_datetime. Defaults to None.
            delayed_flights (set, optional): A set of flight IDs that have already been delayed. Defaults to None.
        """
        if DEBUG_MODE_SCHEDULING:
            print("\n=== Starting schedule_flight_on_aircraft ===")
            print(f"Scheduling flight {flight_id} on aircraft {aircraft_id}")
            print(f"Initial dep_time: {dep_time}, arr_time: {arr_time}")

        if delayed_flights is None:
            delayed_flights = set()
        
        aircraft_idx = self.aircraft_id_to_idx[aircraft_id] + 1  # Adjust for state indexing

        # Get the original flight times and duration
        original_dep_time = parse_time_with_day_offset(
            self.flights_dict[flight_id]['DepTime'], self.start_datetime
        )
        original_arr_time = parse_time_with_day_offset(
            self.flights_dict[flight_id]['ArrTime'], self.start_datetime
        )
        original_dep_minutes = (original_dep_time - self.earliest_datetime).total_seconds() / 60
        flight_duration = (original_arr_time - original_dep_time).total_seconds() / 60
        original_arr_minutes = (original_arr_time - self.earliest_datetime).total_seconds() / 60

        if DEBUG_MODE_SCHEDULING:
            print(f"Original departure minutes: {original_dep_minutes}")
            print(f"Flight duration: {flight_duration}")

        # Ensure dep_time is not earlier than original departure time
        dep_time = max(dep_time, original_dep_minutes)

        if arr_time is None:
            arr_time = dep_time + flight_duration
        else:
            flight_duration = arr_time - dep_time

        # Check for unavailability conflicts
        unavail_info = self.unavailabilities_dict.get(aircraft_id, {})
        unavail_start = unavail_info.get('StartTime', np.nan)
        unavail_end = unavail_info.get('EndTime', np.nan)
        unavail_prob = unavail_info.get('Probability', 0.0)

        if DEBUG_MODE_SCHEDULING:
            print(f"\nUnavailability check:")
            print(f"Current aircraft: {current_aircraft_id}, Target aircraft: {aircraft_id}")
            print(f"Unavailability - Start: {unavail_start}, End: {unavail_end}, Prob: {unavail_prob}")

        # Check if flight overlaps with unavailability
        has_unavail_overlap = False
        if (not np.isnan(unavail_start) and 
            not np.isnan(unavail_end) and 
            unavail_prob > 0.0):  # Only check for overlap if there's an actual unavailability
            
            # Convert times to ensure proper comparison
            flight_start = float(original_dep_minutes)
            flight_end = float(original_arr_minutes)
            unavail_start_time = float(unavail_start)
            unavail_end_time = float(unavail_end)
            
            # Check for any overlap between flight and unavailability period
            if max(flight_start, unavail_start_time) < min(flight_end, unavail_end_time):
                has_unavail_overlap = True
                
            if DEBUG_MODE_SCHEDULING:
                print(f"\nChecking overlap:")
                print(f"Flight: {flight_start} -> {flight_end}")
                print(f"Unavail: {unavail_start_time} -> {unavail_end_time}")
                print(f"Overlap detected: {has_unavail_overlap}")



        current_ac_is_same_as_target_ac = aircraft_id == current_aircraft_id
        if not current_ac_is_same_as_target_ac:
            self.something_happened = True

        if DEBUG_MODE_SCHEDULING:
            print(f"****** current_ac_is_same_as_target_ac: {current_ac_is_same_as_target_ac}") 

        if current_ac_is_same_as_target_ac and not has_unavail_overlap:
            if DEBUG_MODE_SCHEDULING:
                print("****** No unavailability overlap and current aircraft is the same as target aircraft - Keeping original schedule")
            self.something_happened = False
            return

        if has_unavail_overlap:
            if DEBUG_MODE_SCHEDULING:
                print("\nFlight overlaps with unavailability period!")
                print(f"Flight times - Dep: {dep_time}, Arr: {arr_time}")
                print(f"Unavail times - Start: {unavail_start}, End: {unavail_end}")

            if aircraft_id == current_aircraft_id:
                if unavail_prob > 0.00:
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 1: Current aircraft with prob > 0.00 - Moving flight after unavailability")
                    dep_time = max(dep_time, unavail_end + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                    self.something_happened = True
                else:
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 2: Current aircraft with prob = 0.00 - Keeping original schedule")
                    self.something_happened = False
            else:
                if unavail_prob == 1.00:
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 3: Different aircraft with prob = 1.00 - Moving flight after unavailability")
                    dep_time = max(dep_time, unavail_end + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                    self.something_happened = True
                else:
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 4: Different aircraft with prob < 1.00 - Allowing overlap")
                    self.something_happened = True

        # Now handle conflicts with other flights
        scheduled_flights = []
        for j in range(4, self.columns_state_space - 2, 3):
            existing_flight_id = self.state[aircraft_idx, j]
            existing_dep_time = self.state[aircraft_idx, j + 1]
            existing_arr_time = self.state[aircraft_idx, j + 2]
            if not np.isnan(existing_flight_id) and not np.isnan(existing_dep_time) and not np.isnan(existing_arr_time):
                scheduled_flights.append((existing_flight_id, existing_dep_time, existing_arr_time))

        # Check for conflicts with existing flights
        for existing_flight_id, existing_dep_time, existing_arr_time in scheduled_flights:
            if existing_flight_id == flight_id:
                continue  # Skip the same flight
            if dep_time < existing_arr_time + MIN_TURN_TIME and arr_time + MIN_TURN_TIME > existing_dep_time:
                # Conflict detected
                # Decide whether to delay new flight or existing flight

                # Compute the minimum delays required
                delay_new_flight = existing_arr_time + MIN_TURN_TIME - dep_time
                delay_existing_flight = arr_time + MIN_TURN_TIME - existing_dep_time

                delay_new_flight = max(0, delay_new_flight)
                delay_existing_flight = max(0, delay_existing_flight)

                if delay_new_flight <= delay_existing_flight:
                    # Delay new flight
                    dep_time += delay_new_flight
                    dep_time = max(dep_time, original_dep_minutes)  # Ensure not earlier than original dep time
                    arr_time = dep_time + flight_duration

                    # Update delay tracking
                    total_delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + total_delay

                    # Debugging: Print the delayed departure and arrival times
                    if DEBUG_MODE_SCHEDULING:
                        print(f"Delayed departure time for flight {flight_id} to {dep_time} minutes due to conflict.")
                        print(f"Delayed arrival time for flight {flight_id} to {arr_time} minutes.")
                else:
                    # Delay existing flight
                    if existing_flight_id in delayed_flights:
                        # To avoid infinite loop, delay new flight
                        dep_time += delay_new_flight
                        dep_time = max(dep_time, original_dep_minutes)  # Ensure not earlier than original dep time
                        arr_time = dep_time + flight_duration

                        # Update delay tracking
                        total_delay = dep_time - original_dep_minutes
                        self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + total_delay

                        # Debugging: Print the delayed departure and arrival times
                        if DEBUG_MODE_SCHEDULING:
                            print(f"Delayed departure time for flight {flight_id} to {dep_time} minutes due to conflict with existing flight.")
                            print(f"Delayed arrival time for flight {flight_id} to {arr_time} minutes.")
                    else:
                        delayed_flights.add(existing_flight_id)
                        new_dep_time = existing_dep_time + delay_existing_flight
                        new_arr_time = existing_arr_time + delay_existing_flight
                        # Ensure existing flight is not scheduled earlier than original departure
                        original_existing_dep_time = parse_time_with_day_offset(
                            self.flights_dict[existing_flight_id]['DepTime'], self.start_datetime
                        )
                        original_existing_dep_minutes = (original_existing_dep_time - self.earliest_datetime).total_seconds() / 60
                        new_dep_time = max(new_dep_time, original_existing_dep_minutes)
                        new_arr_time = new_dep_time + (existing_arr_time - existing_dep_time)

                        delay_existing = new_dep_time - existing_dep_time
                        self.environment_delayed_flights[existing_flight_id] = self.environment_delayed_flights.get(existing_flight_id, 0) + delay_existing

                        # Update state for existing flight
                        for k in range(4, self.columns_state_space - 2, 3):
                            if self.state[aircraft_idx, k] == existing_flight_id:
                                self.state[aircraft_idx, k + 1] = new_dep_time
                                self.state[aircraft_idx, k + 2] = new_arr_time
                                break
                        # Update flights_dict for existing flight
                        self.update_flight_times(existing_flight_id, new_dep_time, new_arr_time)
                        # Recursively resolve conflicts for existing flight
                        self.schedule_flight_on_aircraft(aircraft_id, existing_flight_id, new_dep_time, current_aircraft_id, new_arr_time, delayed_flights)

        # Now, update the flight's times in the state
        for j in range(4, self.columns_state_space - 2, 3):
            if self.state[aircraft_idx, j] == flight_id:
                self.state[aircraft_idx, j + 1] = dep_time
                self.state[aircraft_idx, j + 2] = arr_time
                break
            elif np.isnan(self.state[aircraft_idx, j]):
                self.state[aircraft_idx, j] = flight_id
                self.state[aircraft_idx, j + 1] = dep_time
                self.state[aircraft_idx, j + 2] = arr_time
                break

        # Update flights_dict
        self.update_flight_times(flight_id, dep_time, arr_time)

        # Debugging: Print the final departure and arrival times
        if DEBUG_MODE_SCHEDULING:
            print(f"Final departure time for flight {flight_id}: {dep_time} minutes.")
            print(f"Final arrival time for flight {flight_id}: {arr_time} minutes.")



        
    def cancel_flight(self, flight_id):
        """Cancels the specified flight.

        This function removes the flight from the rotations dictionary, the flights dictionary, and the state.
        It also marks the flight as cancelled and removes it from the state.
        
        Args:
            flight_id (str): The ID of the flight to cancel.
        """
        # Remove the flight from rotations_dict
        if flight_id in self.rotations_dict:
            del self.rotations_dict[flight_id]

        # Remove the flight from flights_dict
        if flight_id in self.flights_dict:
            del self.flights_dict[flight_id]

        # Mark the flight as cancelled
        self.cancelled_flights.add(flight_id)

        # Remove the flight from the state
        for idx in range(1, self.rows_state_space):
            for j in range(4, self.columns_state_space - 2, 3):
                existing_flight_id = self.state[idx, j]
                if existing_flight_id == flight_id:
                    # Remove flight from state
                    self.state[idx, j] = np.nan
                    self.state[idx, j + 1] = np.nan
                    self.state[idx, j + 2] = np.nan

        self.something_happened = True


    def update_flight_times(self, flight_id, dep_time_minutes, arr_time_minutes):
        """Updates the flight times in the flights dictionary.

        This function converts the departure and arrival times from minutes to datetime format and updates the
        corresponding entries in the flights dictionary.

        Args:
            flight_id (str): The ID of the flight to update.
            dep_time_minutes (float): The new departure time in minutes.
            arr_time_minutes (float): The new arrival time in minutes.
        """
        # Convert minutes to datetime
        dep_time = self.earliest_datetime + timedelta(minutes=dep_time_minutes)
        arr_time = self.earliest_datetime + timedelta(minutes=arr_time_minutes)

        # Update flights_dict with new dates and times
        if DEBUG_MODE:
            print("Updating flight times for flight", flight_id)
            print(" - previous times:", self.flights_dict[flight_id]['DepTime'], self.flights_dict[flight_id]['ArrTime'])
            print(" - new times:", dep_time.strftime('%H:%M'), arr_time.strftime('%H:%M'))
        
        self.flights_dict[flight_id]['DepDate'] = dep_time.strftime('%d/%m/%y')
        self.flights_dict[flight_id]['DepTime'] = dep_time.strftime('%H:%M')
        self.flights_dict[flight_id]['ArrDate'] = arr_time.strftime('%d/%m/%y')
        self.flights_dict[flight_id]['ArrTime'] = arr_time.strftime('%H:%M')



    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, flight_action, aircraft_action, original_flight_action_departure_time):
        """Calculates the reward based on the current state of the environment.

        This function evaluates the reward based on conflict resolutions, delays, cancelled flights, and inaction penalties.
        It returns the total reward for the action taken.

        Args:
            resolved_conflicts (set): The set of conflicts that were resolved during the action.
            remaining_conflicts (set): The set of conflicts that remain after the action.
            flight_action (int): The flight action taken by the agent.
            aircraft_action (int): The aircraft action taken by the agent.
            original_flight_action_departure_time (str): The departure time of the flight that is being acted upon (before the action is taken)
        Returns:
            float: The calculated reward for the action.
        """
        reward = 0

        if DEBUG_MODE_REWARD:
            print("")
            print(f"Reward for action: flight {flight_action}, aircraft {aircraft_action}")

        # Exclude conflicts resolved via cancellation
        resolved_conflicts_non_cancellation = set()
        for conflict in resolved_conflicts:
            flight_id = conflict[1]
            if flight_id not in self.cancelled_flights:
                resolved_conflicts_non_cancellation.add(conflict)

        # 1. **Reward for resolving conflicts**
        conflict_resolution_reward = RESOLVED_CONFLICT_REWARD * len(resolved_conflicts_non_cancellation)
        reward += conflict_resolution_reward
        if DEBUG_MODE_REWARD:
            num_resolved_conflicts = len(resolved_conflicts_non_cancellation)
            print(f"  +{conflict_resolution_reward} for resolving {num_resolved_conflicts} conflicts (excluding cancellations): {resolved_conflicts_non_cancellation}")

        # 2. **Penalty for delays**
        delay_penalty = 0
        for flight_id, new_start_time in self.environment_delayed_flights.items():
            # Check if the flight_id is already in penalized_delays
            if flight_id in self.penalized_delays:
                # Compare the new start time with the previously recorded one
                if new_start_time != self.penalized_delays[flight_id]:
                    # Apply the delay penalty for the new delay
                    delay = new_start_time - self.penalized_delays[flight_id]
                    delay_penalty += delay
                    self.penalized_delays[flight_id] = new_start_time  # Update to the new start time
            else:
                # If not penalized yet, add the delay
                delay_penalty += new_start_time
                self.penalized_delays[flight_id] = new_start_time  # Record the new start time

        delay_penalty_total = delay_penalty * DELAY_MINUTE_PENALTY

        capped_delay_penalty = False
        if delay_penalty_total > MAX_DELAY_PENALTY:
            capped_delay_penalty = True
            delay_penalty_total = MAX_DELAY_PENALTY
        else:
            delay_penalty_total = delay_penalty_total

        reward -= delay_penalty_total

        if DEBUG_MODE_REWARD:
            penalty_info = f"  -{delay_penalty_total} for delays ({delay_penalty} minutes)"
            if capped_delay_penalty:
                penalty_info += " (capped at maximum allowed penalty)"
            print(penalty_info)

        # 3. **Penalty for cancelled flights**
        cancel_penalty = 0
        for flight_id in self.cancelled_flights:
            if flight_id not in self.penalized_cancelled_flights:
                cancel_penalty += CANCELLED_FLIGHT_PENALTY
                self.penalized_cancelled_flights.add(flight_id)  # Mark flight as penalized

        reward -= cancel_penalty

        if DEBUG_MODE_REWARD:
            print(f"  -{cancel_penalty} penalty for cancelled flights")

        # 4. **Penalty for taking no action when conflicts exist**
        inaction_penalty = 0
        if flight_action == 0 and len(remaining_conflicts) > 0:
            inaction_penalty = NO_ACTION_PENALTY
            reward -= inaction_penalty
        if DEBUG_MODE_REWARD:
            print(f"  -{inaction_penalty} for inaction with conflicts")

        # 5. **Bonus for proactive changes **
        ahead_of_time_bonus = 0
        
        if flight_action == 0:
            self.something_happened = False
        if flight_action != 0 and self.something_happened:
            selected_flight_id = flight_action
            original_dep_time = parse_time_with_day_offset(original_flight_action_departure_time, self.start_datetime)
            original_dep_minutes = (original_dep_time - self.earliest_datetime).total_seconds() / 60
            current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
            
            # we are currently one timestep ahead of the actual time the action is taken
            time_at_action = current_time_minutes - (timedelta(hours=TIMESTEP_HOURS).total_seconds() / 60)
            time_to_departure = original_dep_minutes - time_at_action

            ahead_of_time_bonus = AHEAD_BONUS_PER_MINUTE * time_to_departure

        reward += ahead_of_time_bonus
        
        if DEBUG_MODE_REWARD:
            if ahead_of_time_bonus > 0:
                print(f"  +{ahead_of_time_bonus} bonus for proactive action ({time_to_departure:.1f} minutes ahead)")
            else:
                print(f"  -{ahead_of_time_bonus} bonus for proactive action")


        # 6. **Penalty per minute passed**
        time_penalty = (self.current_datetime - self.earliest_datetime).total_seconds() / 60 * TIME_MINUTE_PENALTY
        reward -= time_penalty
        if DEBUG_MODE_REWARD:
            print(f"  -{time_penalty} penalty for time passed")


        if DEBUG_MODE_REWARD:
            print("_______________")
            print(f"{reward} total reward for action: flight {flight_action}, aircraft {aircraft_action}")

        return reward

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        # Generate a random seed based on current time if none provided
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32 - 1)
        
        # Set random seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        
        # Rest of the reset method remains unchanged
        self.current_datetime = self.start_datetime
        self.actions_taken = set()

        self.something_happened = False

        # Deep copy the initial dictionaries
        self.aircraft_dict = copy.deepcopy(self.initial_aircraft_dict)
        self.flights_dict = copy.deepcopy(self.initial_flights_dict)
        self.rotations_dict = copy.deepcopy(self.initial_rotations_dict)
        self.alt_aircraft_dict = copy.deepcopy(self.initial_alt_aircraft_dict)

        # Clear and regenerate breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Calculate total simulation minutes
        total_simulation_minutes = (self.end_datetime - self.start_datetime).total_seconds() / 60

        # # Generate breakdowns for each aircraft
        # for aircraft_id in self.aircraft_ids:
        #     if aircraft_id in self.alt_aircraft_dict:
        #         continue

        #     breakdown_probability = np.random.uniform(0, 1)  # Set realistic probability
        #     if breakdown_probability > MIN_BREAKDOWN_PROBABILITY:  # Set a minimum threshold if desired
        #         max_breakdown_start = total_simulation_minutes - BREAKDOWN_DURATION
        #         if max_breakdown_start > 0:
        #             breakdown_start_minutes = np.random.uniform(0, max_breakdown_start)
        #             breakdown_start = self.start_datetime + timedelta(minutes=breakdown_start_minutes)
                    
        #             # Generate a random breakdown duration for this specific breakdown
        #             breakdown_duration = np.random.uniform(60, 600)  # Random duration between 60 and 600 minutes
        #             breakdown_end = breakdown_start + timedelta(minutes=breakdown_duration)

        #             self.uncertain_breakdowns[aircraft_id] = [{
        #                 'StartTime': breakdown_start,
        #                 'EndTime': breakdown_end,
        #                 'StartDate': breakdown_start.date(),
        #                 'EndDate': breakdown_end.date(),
        #                 'Probability': breakdown_probability,
        #                 'Resolved': False  # Initially unresolved
        #             }]

        #             if DEBUG_MODE:
        #                 print(f"Aircraft {aircraft_id} has an uncertain breakdown scheduled at {breakdown_start} with probability {breakdown_probability:.2f}")

        self.state = self._get_initial_state()

        self.swapped_flights = []  # Reset the swapped flights list
        self.environment_delayed_flights = {}  # Reset the delayed flights list
        self.penalized_delays = {}  # Reset the penalized delays
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()  # Reset penalized cancelled flights

        self.cancelled_flights = set()

        # Process the state into an observation as a NumPy array
        processed_state = self.process_observation(self.state)

        if DEBUG_MODE:
            print(f"State space shape: {self.state.shape}")
        
        return processed_state, {}


    def get_current_conflicts(self):
        """Retrieves the current conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods, considering only unavailabilities with probability 1.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts = set()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability != 1.0:  # Only consider unavailability with probability 1.00
                continue  # Skip if probability is not 1.00

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Check if the flight's departure is in the past (relative to current time)
                        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
                        if flight_dep < current_time_minutes:
                            continue  # Skip past flights

                        if flight_id in self.cancelled_flights:
                            continue  # Skip cancelled flights

                        # Check for overlaps with unavailability periods with prob = 1.00
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id, flight_dep, flight_arr)
                            current_conflicts.add(conflict_identifier)

        return current_conflicts

    def _is_done(self):
        """Checks if the episode is finished.

        This function determines if the current time has reached or exceeded the end time of the simulation
        or if there are no remaining conflicts and all uncertainties have been resolved.

        Returns:
            tuple: (bool, str) indicating if the episode is done and the reason.
        """
        current_conflicts = self.get_current_conflicts()  # Get current conflicts
        if DEBUG_MODE:
            print(f"Current conflicts before checking done: {current_conflicts}")  # Debugging statement

        # Check for unresolved uncertainties
        unresolved_uncertainties = self.get_unresolved_uncertainties()


        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        elif len(current_conflicts) == 0 and len(unresolved_uncertainties) == 0:
            if DEBUG_MODE:
                print("No remaining conflicts or uncertainties detected.")  # Debugging statement
            return True, "No remaining conflicts or uncertainties."
        
        return False, ""

    def get_unresolved_uncertainties(self):
        """Retrieves the uncertainties that have not yet been resolved.

        Returns:
            list: A list of unresolved uncertainties currently present in the environment.
        """
        unresolved_uncertainties = []
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if prob != 0.00 and prob != 1.00 and not np.isnan(prob):
                # Uncertainty not yet resolved
                start_minutes = self.unavailabilities_dict[aircraft_id]['StartTime']
                breakdown_start_time = self.earliest_datetime + timedelta(minutes=start_minutes)
                if self.current_datetime < breakdown_start_time:
                    unresolved_uncertainties.append((aircraft_id, prob))
        return unresolved_uncertainties


    # Note: get_valid_actions is no longer needed due to action_space change

    def get_valid_flight_actions(self):
        """Generates a list of valid flight actions based on flights in state space."""
        # Calculate current time in minutes from earliest_datetime
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        # Get all valid flight IDs from the state space
        valid_flight_ids = set()
        for idx in range(1, self.rows_state_space):
            for j in range(4, self.columns_state_space - 2, 3):
                flight_id = self.state[idx, j]
                if not np.isnan(flight_id):
                    flight_id = int(flight_id)
                    # Check if flight hasn't departed yet
                    dep_time = self.state[idx, j + 1]
                    if dep_time >= current_time_minutes and flight_id not in self.cancelled_flights:
                        valid_flight_ids.add(flight_id)

        # Convert to sorted list and add 'no action' option
        valid_flight_ids = sorted(list(valid_flight_ids))
        
        # Update flight_id_to_idx mapping using actual flight IDs
        self.flight_id_to_idx = {
            flight_id: flight_id - 1 for flight_id in valid_flight_ids
        }

        if DEBUG_MODE_ACTION:
            print(f"Valid flight indices: {valid_flight_ids}")
        
        # Return [0] + actual flight IDs instead of creating sequential indices
        return [0] + valid_flight_ids



    def get_valid_aircraft_actions(self):
        """Generates a list of valid aircraft actions for the agent.

        Returns:
            list: A list of valid aircraft actions that the agent can take.
        """
        return list(range(len(self.aircraft_ids) + 1))  # 0 to len(aircraft_ids)

    def get_action_mask(self):
        valid_flight_actions = self.get_valid_flight_actions()
        valid_aircraft_actions = self.get_valid_aircraft_actions()

        action_mask = np.zeros(self.action_space.n, dtype=np.uint8)

        for flight_action in valid_flight_actions:
            for aircraft_action in valid_aircraft_actions:
                if flight_action == 0:
                    # Only allow (flight_action=0, aircraft_action=0)
                    if aircraft_action != 0:
                        continue
                index = self.map_action_to_index(flight_action, aircraft_action)
                if index < self.action_space.n:
                    action_mask[index] = 1


        return action_mask

    def map_action_to_index(self, flight_action, aircraft_action):
        """Maps the (flight, aircraft) action pair to a single index in the flattened action space.

        Args:
            flight_action (int): The index of the flight action.
            aircraft_action (int): The index of the aircraft action.

        Returns:
            int: The corresponding index in the flattened action space.
        """
        return flight_action * (len(self.aircraft_ids) + 1) + aircraft_action
    
    def map_index_to_action(self, index):
        """Maps the flattened action space index to the corresponding (flight, aircraft) action pair.

        Args:
            index (int): The index in the flattened action space.

        Returns:
            tuple: A tuple containing the flight and aircraft actions.
        """
        flight_action = index // (len(self.aircraft_ids) + 1)
        aircraft_action = index % (len(self.aircraft_ids) + 1)
        return flight_action, aircraft_action
