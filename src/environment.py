import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config import *
from scripts.utils import *

# import the state plotter
from scripts.visualizations import StatePlotter

MIN_BREAKDOWN_PROBABILITY = 0

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict):
        super(AircraftDisruptionEnv, self).__init__()

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

        # Flight information and indexing
        self.flight_ids = list(flights_dict.keys())
        self.flight_id_to_idx = {flight_id: idx for idx, flight_id in enumerate(self.flight_ids)}

        # Filter out flights with '+' in DepTime (next day flights)
        this_day_flights = [flight_info for flight_info in flights_dict.values() if '+' not in flight_info['DepTime']]

        # Determine the earliest possible event in the environment
        self.earliest_datetime = min(
            min(datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + flight_info['DepTime'], '%d/%m/%y %H:%M') for flight_info in this_day_flights),
            self.start_datetime
        )

        # Define observation and action spaces
        self.state_space_size = (self.rows_state_space, self.columns_state_space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.rows_state_space * self.columns_state_space * 2,), dtype=np.float32
        )

        # Action space: select a flight and an aircraft
        self.action_space = spaces.MultiDiscrete([len(self.flight_ids) + 1, len(self.aircraft_ids) + 1])

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
        self.penalized_delays = set()           # Set of penalized delays
        self.penalized_conflicts = set()        # Set of penalized conflicts
        self.resolved_conflicts = set()         # Set of resolved conflicts
        self.penalized_cancelled_flights = set()  # To keep track of penalized cancelled flights

        self.cancelled_flights = set()

        # Initialize empty containers for breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Initialize the environment state without generating probabilities
        self.current_datetime = self.start_datetime
        self.state = self._get_initial_state()

    def _get_initial_state(self):
        """Initializes the state matrix for the environment.

        This function creates a state matrix filled with NaN values, calculates the current time and remaining recovery period,
        and populates the state matrix with aircraft and flight information. It also tracks the earliest conflict information.

        Returns:
            np.ndarray: The initialized state matrix representing the current state of the environment.
        """
        # Initialize state matrix with NaN values
        state = np.full((self.rows_state_space, self.columns_state_space), np.nan)

        # Calculate current time and remaining recovery period in minutes
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        time_until_end_minutes = (self.end_datetime - self.current_datetime).total_seconds() / 60

        # Insert the current_time_minutes and time_until_end_minutes in the first row
        state[0, 0] = current_time_minutes  # Current time
        state[0, 1] = time_until_end_minutes  # Time until end of recovery period

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
                    flight_times.append((flight_id, dep_time_minutes, arr_time_minutes))

            # Sort flights by departure time
            flight_times.sort(key=lambda x: x[1])

            # Store flight information starting from column 4
            for i, (flight_id, dep_time, arr_time) in enumerate(flight_times):
                col_start = 4 + (i * 3)
                if col_start + 2 < self.columns_state_space:
                    state[idx + 1, col_start] = flight_id
                    state[idx + 1, col_start + 1] = dep_time
                    state[idx + 1, col_start + 2] = arr_time

        # Store earliest conflict information in first row
        earliest_conf_aircraft_idx = None
        earliest_conf_flight_id = None
        earliest_conf_dep_time = None

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break  # Only process up to the maximum number of aircraft

            breakdown_probability = state[idx + 1, 1]
            if breakdown_probability != 1.0:
                continue  # Only consider unavailabilities with probability 1

            unavail_start = state[idx + 1, 2]
            unavail_end = state[idx + 1, 3]

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = state[idx + 1, j]
                    flight_dep = state[idx + 1, j + 1]

                    if not np.isnan(flight_dep):
                        if flight_dep < current_time_minutes:
                            self.cancelled_flights.add(flight_id)
                            continue
                        if flight_id in self.cancelled_flights:
                            continue

                        # Check for conflicts between flight and unavailability periods
                        if flight_dep < unavail_end and flight_dep >= unavail_start:
                            if earliest_conf_dep_time is None or flight_dep < earliest_conf_dep_time:
                                earliest_conf_dep_time = flight_dep
                                earliest_conf_flight_id = flight_id
                                earliest_conf_aircraft_idx = idx + 1

        return state

    def process_observation(self, state):
        """Processes the observation by applying a mask and flattening the state and mask.

        This function creates a mask to indicate valid values in the state, replaces NaN values with a dummy value,
        and concatenates the flattened state and mask into a single observation.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The processed observation including the state and mask.
        """
        # Create a mask where 1 indicates valid values, 0 indicates NaN
        mask = np.where(np.isnan(state), 0, 1)
        # Replace NaN with the dummy value
        state = np.nan_to_num(state, nan=DUMMY_VALUE)
        # Flatten both state and mask
        state_flat = state.flatten()
        mask_flat = mask.flatten()
        # Concatenate state and mask
        obs_with_mask = np.concatenate([state_flat, mask_flat])
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


    def step(self, action=None):
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
            print_state_nicely(self.state)
            print("")

        # Extract the action values from the action
        flight_action, aircraft_action = self.extract_action_value(action)

        # Validate the action
        self.validate_action(flight_action, aircraft_action)

        # Print the processed action and chosen action
        if DEBUG_MODE:
            print(f"Processed action: {action} of type: {type(action)}")
        print(f"Chosen action: flight {flight_action}, aircraft {aircraft_action}")

        # Initialize info dictionary
        info = {}

        # Process uncertainties
        self.process_uncertainties()

        # Print the state after processing uncertainties
        print_state_nicely(self.state)

        # Get pre-action conflicts
        pre_action_conflicts = self.get_current_conflicts()

        if len(pre_action_conflicts) == 0:
            # Handle the case when there are no conflicts
            return self.handle_no_conflicts()

        else:
            # Resolve the conflict based on the action
            return self.resolve_conflict(flight_action, aircraft_action, pre_action_conflicts)

    def extract_action_value(self, action):
        """Extracts the action values from the provided action.

        This function handles different types of action inputs, such as lists or arrays,
        and extracts the flight and aircraft action values accordingly.

        Args:
            action (list or array): The action provided by the agent.

        Returns:
            tuple: The flight action and aircraft action values.
        """
        if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
            flight_action = action[0]
            aircraft_action = action[1]
        else:
            raise ValueError("Invalid action format")
        return flight_action, aircraft_action

    def validate_action(self, flight_action, aircraft_action):
        """Validates the provided action values.

        This function ensures that the action is within the set of valid actions.

        Args:
            flight_action (int): The flight action value to be validated.
            aircraft_action (int): The aircraft action value to be validated.

        Raises:
            ValueError: If the action is not valid.
        """
        # Check if flight_action and aircraft_action are within the action_space bounds
        if not (0 <= flight_action <= len(self.flight_ids)):
            raise ValueError(f"Invalid flight action: {flight_action}")
        if not (0 <= aircraft_action <= len(self.aircraft_ids)):
            raise ValueError(f"Invalid aircraft action: {aircraft_action}")

        # No action case
        if flight_action == 0 or aircraft_action == 0:
            # Treat as 'no action'
            return

        # No longer check if the selected flight is conflicting
        # Since we allow swapping any flight regardless of conflicts
        selected_flight_id = self.flight_ids[flight_action - 1]
        selected_aircraft_id = self.aircraft_ids[aircraft_action - 1]

    def process_uncertainties(self):
        """Processes breakdown uncertainties directly from the state space.
        For any probability that isn't 0.00 or 1.00, if the current datetime + timestep
        has reached or passed the breakdown start time, resolve the uncertainty by rolling a dice.
        """
        print(f"Current datetime: {self.current_datetime}")

        # Iterate over each aircraft's row in the state space to check for unresolved breakdowns
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            # Get probability, start, and end time from the state space
            prob = self.state[idx + 1, 1]
            start_minutes = self.state[idx + 1, 2]
            end_minutes = self.state[idx + 1, 3]
            
            # Only process unresolved breakdowns
            if prob != 0.00 and prob != 1.00:
                breakdown_start_time = self.earliest_datetime + timedelta(minutes=start_minutes)
                
                # If within the current time step, resolve the breakdown
                if self.current_datetime + self.timestep >= breakdown_start_time:
                    print(f"Rolling the dice for breakdown with initial probability {prob} starting at {breakdown_start_time}")
                    
                    # Roll the dice
                    if np.random.random() <= prob:
                        print(f"Breakdown confirmed for aircraft {aircraft_id} with probability {prob}")
                        self.state[idx + 1, 1] = 1.00  # Confirm the breakdown
                    else:
                        print(f"Breakdown not occurring for aircraft {aircraft_id}")
                        self.state[idx + 1, 1] = 0.00  # Resolve as no breakdown

                    # Ensure `self.alt_aircraft_dict[aircraft_id]` is a list of dictionaries
                    if aircraft_id in self.alt_aircraft_dict:
                        if isinstance(self.alt_aircraft_dict[aircraft_id], dict):
                            self.alt_aircraft_dict[aircraft_id] = [self.alt_aircraft_dict[aircraft_id]]
                        elif isinstance(self.alt_aircraft_dict[aircraft_id], str):
                            # Handle case where entry is a string by converting it to a structured dictionary
                            self.alt_aircraft_dict[aircraft_id] = [{
                                'StartDate': breakdown_start_time.strftime('%d/%m/%y'),
                                'StartTime': breakdown_start_time.strftime('%H:%M'),
                                'EndDate': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%d/%m/%y'),
                                'EndTime': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%H:%M'),
                                'Probability': self.state[idx + 1, 1]  # Updated probability
                            }]
                        
                        # Update the probability in `self.alt_aircraft_dict`
                        for breakdown_info in self.alt_aircraft_dict[aircraft_id]:
                            breakdown_info['Probability'] = self.state[idx + 1, 1]


    def handle_no_conflicts(self):
        """Handles the case when there are no conflicts in the current state.

        This function updates the current datetime, checks if the episode is terminated,
        updates the state, and returns the appropriate outputs.

        Returns:
            tuple: A tuple containing the processed state, reward, terminated flag, truncated flag, and info dictionary.
        """
        next_datetime = self.current_datetime + self.timestep
        if next_datetime >= self.end_datetime:
            terminated, reason = self._is_done()
            if terminated:
                print(f"Episode ended: {reason}")
                processed_state = self.process_observation(self.state)
                truncated = False
                reward = 0  # Assuming zero reward when episode ends without conflicts
                return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

        self.current_datetime = next_datetime
        self.state = self._get_initial_state()

        # Since there are no conflicts, return the new state with zero reward
        terminated, reason = self._is_done()
        truncated = False
        processed_state = self.process_observation(self.state)
        reward = 0  # Assuming zero reward when there are no conflicts

        if terminated:
            print(f"Episode ended: {reason}")

        return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}


    def resolve_conflict(self, flight_action, aircraft_action, pre_action_conflicts):
        """Resolves the conflicts in the current state based on the provided action.

        This function handles swapping or delaying any flight as specified by the agent.

        Args:
            flight_action (int): The flight action value provided by the agent.
            aircraft_action (int): The aircraft action value provided by the agent.
            pre_action_conflicts (set): The set of conflicts before taking the action.

        Returns:
            tuple: A tuple containing the processed state, reward, terminated flag, truncated flag, and info dictionary.
        """
        if flight_action == 0 or aircraft_action == 0:
            # Treat as no action taken
            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action)

            terminated, reason = self._is_done()
            truncated = False

            if terminated:
                print(f"Episode ended: {reason}")

            processed_state = self.process_observation(self.state)
            return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

        else:
            selected_flight_id = self.flight_ids[flight_action - 1]
            selected_aircraft_id = self.aircraft_ids[aircraft_action - 1]

            # Get the current aircraft assigned to the flight
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
                    selected_aircraft_id, selected_flight_id, new_dep_time_minutes, None
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

                self.schedule_flight_on_aircraft(selected_aircraft_id, selected_flight_id, dep_time_minutes, arr_time_minutes)

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action)

            terminated, reason = self._is_done()
            truncated = False

            if terminated:
                print(f"Episode ended: {reason}")

            processed_state = self.process_observation(self.state)
            return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, arr_time=None, delayed_flights=None):
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

        # Ensure dep_time is not earlier than original departure time
        dep_time = max(dep_time, original_dep_minutes)

        if arr_time is None:
            # Calculate new arrival time based on flight duration
            arr_time = dep_time + flight_duration
        else:
            # If arr_time is provided, update flight_duration accordingly
            flight_duration = arr_time - dep_time

        # Check for unavailability conflicts
        unavail_start = self.state[aircraft_idx, 2]
        unavail_end = self.state[aircraft_idx, 3]
        unavail_prob = self.state[aircraft_idx, 1]  # Assuming unavailability probability is stored in this position

        if not np.isnan(unavail_start) and not np.isnan(unavail_end):
            # Check if desired dep_time overlaps with unavailability
            if dep_time < unavail_end and arr_time > unavail_start:
                # If unavailability probability is below 1.00, do not adjust dep_time
                if unavail_prob < 1.00:
                    pass  # No adjustment needed
                else:
                    # Adjust dep_time to after unavailability end plus MIN_TURN_TIME
                    dep_time = max(dep_time, unavail_end + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)  # Ensure not earlier than original dep time
                    arr_time = dep_time + flight_duration

                    # Track the delay
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay

                    # Debugging: Print the adjusted departure and arrival times
                    print(f"Adjusted departure time for flight {flight_id} to {dep_time} minutes after unavailability.")
                    print(f"Adjusted arrival time for flight {flight_id} to {arr_time} minutes.")

        # Proceed with existing scheduled_flights check and conflict resolution

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
                        self.schedule_flight_on_aircraft(aircraft_id, existing_flight_id, new_dep_time, new_arr_time, delayed_flights)

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
        print(f"Final departure time for flight {flight_id}: {dep_time} minutes.")
        print(f"Final arrival time for flight {flight_id}: {arr_time} minutes.")

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



    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, flight_action, aircraft_action):
        """Calculates the reward based on the current state of the environment.

        This function evaluates the reward based on conflict resolutions, delays, cancelled flights, and inaction penalties.
        It returns the total reward for the action taken.

        Args:
            resolved_conflicts (set): The set of conflicts that were resolved during the action.
            remaining_conflicts (set): The set of conflicts that remain after the action.
            flight_action (int): The flight action taken by the agent.
            aircraft_action (int): The aircraft action taken by the agent.

        Returns:
            float: The calculated reward for the action.
        """
        reward = 0

        if DEBUG_MODE_REWARD:
            print(f"Chosen action: flight {flight_action}, aircraft {aircraft_action}")

        # 1. **Reward for resolving conflicts**
        conflict_resolution_reward = RESOLVED_CONFLICT_REWARD * len(resolved_conflicts)
        reward += conflict_resolution_reward
        if DEBUG_MODE_REWARD:
            print(f"  +{conflict_resolution_reward} for resolving {len(resolved_conflicts)} conflicts")

        # 2. **Penalty for delays**
        delay_penalty = 0
        for flight_id, delay in self.environment_delayed_flights.items():
            if flight_id not in self.penalized_delays:
                delay_penalty += delay
                self.penalized_delays.add(flight_id)  # Ensure we don't penalize the same delay multiple times

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
        if (flight_action == 0 or aircraft_action == 0) and len(remaining_conflicts) > 0:
            inaction_penalty = NO_ACTION_PENALTY
            reward -= inaction_penalty
        if DEBUG_MODE_REWARD:
            print(f"  -{inaction_penalty} for inaction with conflicts")

        if DEBUG_MODE_REWARD:
            print("_______________")
            print(f"{reward} total reward for action: flight {flight_action}, aircraft {aircraft_action}")

        return reward

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state.

        This function reinitializes the environment, including resetting the current time, clearing previous states,
        and generating new breakdowns for the aircraft. It also processes the state into an observation.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the processed initial state and an empty dictionary.
        """
        self.current_datetime = self.start_datetime
        self.actions_taken = set()

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
        self.penalized_delays = set()  # Reset the penalized delays
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()  # Reset penalized cancelled flights

        self.cancelled_flights = set()

        # Process the state into an observation as a NumPy array
        processed_state = self.process_observation(self.state)

        if DEBUG_MODE:
            print(f"State space shape: {self.state.shape}")
        
        return np.array(processed_state, dtype=np.float32), {}

    def get_current_conflicts(self):
        """Retrieves the current conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods, considering only unavailabilities with probability 1.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts = set()
        self.cancelled_flights = set()  # Track cancelled flights

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.state[idx + 1, 1]
            if breakdown_probability == 0.0 or np.isnan(breakdown_probability):
                continue  # Skip if probability is zero or NaN

            unavail_start = self.state[idx + 1, 2]
            unavail_end = self.state[idx + 1, 3]

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

                        # Check for overlaps with unavailability periods with prob > 0
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
        print(f"Current conflicts before checking done: {current_conflicts}")  # Debugging statement

        # Check for unresolved uncertainties
        unresolved_uncertainties = self.get_unresolved_uncertainties()
        print(f"Unresolved uncertainties: {unresolved_uncertainties}")  # Debugging statement

        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        elif len(current_conflicts) == 0 and len(unresolved_uncertainties) == 0:
            print("No remaining conflicts or uncertainties detected.")  # Debugging statement
            return True, "No remaining conflicts or uncertainties."
        
        return False, ""

    def get_unresolved_uncertainties(self):
        """Retrieves the uncertainties that have not yet been resolved.

        Returns:
            list: A list of unresolved uncertainties currently present in the environment.
        """
        unresolved_uncertainties = []
        for aircraft_id, breakdowns in self.uncertain_breakdowns.items():
            for breakdown_info in breakdowns:
                if not breakdown_info.get('Resolved', False):
                    unresolved_uncertainties.append((aircraft_id, breakdown_info))
        return unresolved_uncertainties

    # Note: get_valid_actions is no longer needed due to action_space change

    def get_valid_flight_actions(self):
        """Generates a list of valid flight actions for the agent.

        Returns:
            list: A list of valid flight actions that the agent can take.
        """
        conflicting_flight_indices = []
        conflicting_flight_ids = set()
        for conflict in self.get_current_conflicts():
            flight_id = conflict[1]
            if flight_id not in conflicting_flight_ids:
                conflicting_flight_ids.add(flight_id)
                flight_idx = self.flight_id_to_idx[flight_id] + 1  # +1 for action 0 being 'no action'
                conflicting_flight_indices.append(flight_idx)
        return [0] + conflicting_flight_indices

    def get_valid_aircraft_actions(self):
        """Generates a list of valid aircraft actions for the agent.

        Returns:
            list: A list of valid aircraft actions that the agent can take.
        """
        return list(range(len(self.aircraft_ids) + 1))  # 0 to len(aircraft_ids)
