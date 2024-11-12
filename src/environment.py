import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config import *
from scripts.utils import *



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
        
        # Action space: index of the aircraft to reassign (0 to number of aircraft)
        self.full_action_space = spaces.Discrete(len(self.aircraft_ids) + 1)
        self.action_space = self.full_action_space

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
        state[0, 2] = current_time_minutes
        state[0, 3] = time_until_end_minutes

        # Populate state matrix with aircraft and flight information
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break  # Only process up to the maximum number of aircraft

            # Store aircraft index instead of ID
            state[idx + 1, 0] = idx + 1  # Use numerical index instead of string ID

            # Initialize breakdown probability and unavailability times
            breakdown_probability = 0.0
            unavail_start_minutes = np.nan
            unavail_end_minutes = np.nan

            # Check for predefined unavailabilities (probability = 1.0)
            if aircraft_id in self.alt_aircraft_dict:
                breakdown_probability = 1.0
                unavails = self.alt_aircraft_dict[aircraft_id]
                if not isinstance(unavails, list):
                    unavails = [unavails]
                
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

        state[0, 0] = earliest_conf_aircraft_idx
        state[0, 1] = earliest_conf_flight_id

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
    


        
    def step(self, action=None):
        """Executes a step in the environment based on the provided action.

        This function processes the action taken by the agent, checks for conflicts, updates the environment state,
        and returns the new state, reward, and termination status.

        Args:
            action (int or list, optional): The action to be taken by the agent. Defaults to None.

        Returns:
            tuple: A tuple containing the processed state, reward, termination status, truncation status, and additional info.
        """

        if DEBUG_MODE_PRINT_STATE:
            print_state_nicely(self.state)
            print("")

        if isinstance(action, (list, np.ndarray)) and np.ndim(action) > 0:
            action_value = action[0]  # Extract the first element if it's a list or array
        else:
            action_value = action  # Use the scalar action directly 

        valid_actions = self.get_valid_actions()
        self.action_space = spaces.Discrete(len(valid_actions))

        if action is not None:
            assert action in valid_actions, f"Invalid action: {action}"

        if DEBUG_MODE:
            print(f"Processed action: {action} of type: {type(action)}")
        
        # Print the chosen action
        print(f"Chosen action: {action_value}")  # Added print statement for chosen action

        agent_acted = False
        info = {} 

        while not agent_acted:
            # Print current datetime
            print(f"Current datetime: {self.current_datetime}")

            # Print the set of cancelled flights
            print(f"Cancelled flights: {self.cancelled_flights}")  # Added print statement for cancelled flights

            pre_action_conflicts = self.get_current_conflicts()

            # Process uncertainties
            for aircraft_id in list(self.uncertain_breakdowns.keys()):  # Use list to avoid runtime modification
                breakdowns = self.uncertain_breakdowns[aircraft_id]
                updated_breakdowns = []
                for breakdown_info in breakdowns:
                    breakdown_start_time = breakdown_info['StartTime']
                    breakdown_end_time = breakdown_info['EndTime']
                    if self.current_datetime >= breakdown_start_time and not breakdown_info.get('Resolved', False):
                        # Time to resolve this uncertainty
                        print(f"Now we are rolling the dice on uncertainty {breakdown_info['Probability']}...")
                        if np.random.random() <= breakdown_info['Probability']:
                            # The breakdown occurs
                            print(f"Uncertainty {breakdown_info['Probability']} becomes an actual unavailability")

                            # Set probability to 1.00 to mark it as confirmed
                            breakdown_info['Probability'] = 1.00  # Explicitly set to 1

                            # Update self.state with new start and end times and probability
                            aircraft_index = self.aircraft_id_to_idx[aircraft_id] + 1  # Adjust for state indexing
                            self.state[aircraft_index, 1] = 1.00  # Probability column
                            self.state[aircraft_index, 2] = (breakdown_start_time - self.earliest_datetime).total_seconds() / 60
                            self.state[aircraft_index, 3] = (breakdown_end_time - self.earliest_datetime).total_seconds() / 60

                            breakdown_key = (aircraft_id, breakdown_start_time)
                            self.current_breakdowns[breakdown_key] = breakdown_info
                            breakdown_info['Resolved'] = True  # Mark as resolved

                            # Add to alt_aircraft_dict
                            if aircraft_id in self.alt_aircraft_dict:
                                if not isinstance(self.alt_aircraft_dict[aircraft_id], list):
                                    self.alt_aircraft_dict[aircraft_id] = [self.alt_aircraft_dict[aircraft_id]]
                                # Append the breakdown without resetting the list
                                self.alt_aircraft_dict[aircraft_id].append({
                                    'StartDate': breakdown_info['StartTime'].strftime('%d/%m/%y'),
                                    'StartTime': breakdown_info['StartTime'].strftime('%H:%M'),
                                    'EndDate': breakdown_info['EndTime'].strftime('%d/%m/%y'),
                                    'EndTime': breakdown_info['EndTime'].strftime('%H:%M'),
                                })
                            else:
                                # Initialize new entry in alt_aircraft_dict
                                self.alt_aircraft_dict[aircraft_id] = [{
                                    'StartDate': breakdown_info['StartTime'].strftime('%d/%m/%y'),
                                    'StartTime': breakdown_info['StartTime'].strftime('%H:%M'),
                                    'EndDate': breakdown_info['EndTime'].strftime('%d/%m/%y'),
                                    'EndTime': breakdown_info['EndTime'].strftime('%H:%M'),
                                }]
                            if DEBUG_MODE:
                                print(f"Aircraft {aircraft_id} has broken down at {breakdown_start_time} with probability {self.state[aircraft_index, 1]}")
                        else:
                            # The breakdown does not occur
                            print(f"Uncertainty {breakdown_info['Probability']} does NOT become an actual unavailability")
                            breakdown_info['Resolved'] = True  # Mark as resolved
                            if DEBUG_MODE:
                                print(f"Aircraft {aircraft_id} did not break down at {breakdown_start_time}")

                    if breakdown_info.get('Resolved', False):
                        # Remove resolved breakdown from uncertainties
                        continue  # Do not add to updated_breakdowns
                    else:
                        updated_breakdowns.append(breakdown_info)
                if updated_breakdowns:
                    self.uncertain_breakdowns[aircraft_id] = updated_breakdowns
                else:
                    del self.uncertain_breakdowns[aircraft_id]

            # Print the state after processing all uncertainties
            print_state_nicely(self.state)

            if len(pre_action_conflicts) == 0:
                next_datetime = self.current_datetime + self.timestep
                if next_datetime >= self.end_datetime:
                    terminated, reason = self._is_done()
                    if terminated:
                        print(f"Episode ended: {reason}")
                        processed_state = self.process_observation(self.state)
                        truncated = False
                        return np.array(processed_state, dtype=np.float32), 0, terminated, truncated, {}

                self.current_datetime = next_datetime
                self.state = self._get_initial_state()

                post_action_conflicts = self.get_current_conflicts()
                continue


            action_explain = "No action taken" if action_value == 0 else f"Action: Aircraft {self.aircraft_ids[action_value - 1]}"
            conflicting_aircraft = None
            conflicting_flight_id = None

            earliest_conflict_time = None
            for idx, aircraft_id in enumerate(self.aircraft_ids):
                if idx >= self.max_aircraft:
                    continue

                breakdown_probability = self.state[idx + 1, 1]
                if breakdown_probability != 1.0:
                    continue  # Only consider unavailabilities with probability 1

                unavail_start = self.state[idx + 1, 2]
                unavail_end = self.state[idx + 1, 3]

                if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                    for j in range(4, self.columns_state_space - 2, 3):
                        flight_id = self.state[idx + 1, j]
                        flight_dep = self.state[idx + 1, j + 1]
                        flight_arr = self.state[idx + 1, j + 2]

                        if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                            if flight_dep < (self.current_datetime - self.earliest_datetime).total_seconds() / 60:
                                self.cancelled_flights.add(flight_id)
                                continue  # Skip this flight as it's already cancelled

                            if flight_id in self.cancelled_flights:
                                continue  # Skip cancelled flights

                            if flight_dep < unavail_end and flight_arr > unavail_start:
                                if (earliest_conflict_time is None or flight_dep < earliest_conflict_time):
                                    earliest_conflict_time = flight_dep
                                    conflicting_aircraft = aircraft_id
                                    conflicting_flight_id = flight_id

            if conflicting_aircraft is None:
                next_datetime = self.current_datetime + self.timestep
                self.current_datetime = next_datetime
                self.state = self._get_initial_state()

                post_action_conflicts = self.get_current_conflicts()
                resolved_conflicts = pre_action_conflicts - post_action_conflicts
                reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, action_value)

                terminated, reason = self._is_done()
                truncated = False

                if terminated:
                    print(f"Episode ended: {reason}")

                processed_state = self.process_observation(self.state)
                return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

            conflicting_idx = self.aircraft_id_to_idx[conflicting_aircraft]

            if action_value == 0:
                next_datetime = self.current_datetime + self.timestep
                self.current_datetime = next_datetime
                self.state = self._get_initial_state()

                post_action_conflicts = self.get_current_conflicts()
                resolved_conflicts = []
                reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, action_value)

                terminated, reason = self._is_done()
                truncated = False

                if terminated:
                    print(f"Episode ended: {reason}")

                processed_state = self.process_observation(self.state)
                return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

            selected_aircraft_id = self.aircraft_ids[action_value - 1]
            selected_idx = self.aircraft_id_to_idx[selected_aircraft_id]

            for j in range(4, self.columns_state_space - 2, 3):
                if self.state[conflicting_idx + 1, j] == conflicting_flight_id:
                    conf_dep_time = self.state[conflicting_idx + 1, j + 1]
                    conf_arr_time = self.state[conflicting_idx + 1, j + 2]
                    break

            if selected_aircraft_id == conflicting_aircraft:
                unavail_end = self.state[conflicting_idx + 1, 3]
                new_dep_time = unavail_end + MIN_TURN_TIME
                delay = new_dep_time - conf_dep_time
                conf_dep_time = new_dep_time
                conf_arr_time += delay
                self.environment_delayed_flights[conflicting_flight_id] = self.environment_delayed_flights.get(conflicting_flight_id, 0) + delay
                self.state[conflicting_idx + 1, j + 1] = conf_dep_time
                self.state[conflicting_idx + 1, j + 2] = conf_arr_time
                self.update_flight_times(conflicting_flight_id, conf_dep_time, conf_arr_time)
                self.schedule_flight_on_aircraft(selected_aircraft_id, conflicting_flight_id, conf_dep_time, conf_arr_time)

                post_action_conflicts = self.get_current_conflicts()
                resolved_conflicts = pre_action_conflicts - post_action_conflicts
                reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, action_value)

                next_datetime = self.current_datetime + self.timestep
                self.current_datetime = next_datetime
                self.state = self._get_initial_state()

                terminated, reason = self._is_done()
                truncated = False

                if terminated:
                    print(f"Episode ended: {reason}")

                processed_state = self.process_observation(self.state)
                return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

            if conflicting_flight_id:
                if conflicting_flight_id in self.cancelled_flights:
                    if DEBUG_MODE:
                        print(f"Skipping tail swap for cancelled flight {conflicting_flight_id}")
                else:
                    self.swapped_flights.append((conflicting_flight_id, selected_aircraft_id))

                    self.rotations_dict[conflicting_flight_id]['Aircraft'] = selected_aircraft_id

                    for j in range(4, self.columns_state_space - 2, 3):
                        if self.state[conflicting_idx + 1, j] == conflicting_flight_id:
                            self.state[conflicting_idx + 1, j] = np.nan
                            self.state[conflicting_idx + 1, j + 1] = np.nan
                            self.state[conflicting_idx + 1, j + 2] = np.nan
                            break

                    self.schedule_flight_on_aircraft(selected_aircraft_id, conflicting_flight_id, conf_dep_time, conf_arr_time)

            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, action_value)

            terminated, reason = self._is_done()
            truncated = False

            if terminated:
                print(f"Episode ended: {reason}")

            processed_state = self.process_observation(self.state)
            return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}


    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, arr_time, delayed_flights=None):
        """Schedules a flight on a specified aircraft and resolves any conflicts.

        This function checks for conflicts with existing flights on the aircraft and adjusts the flight times accordingly.
        It updates the state and the flights dictionary with the new scheduled times.

        Args:
            aircraft_id (str): The ID of the aircraft on which to schedule the flight.
            flight_id (str): The ID of the flight to be scheduled.
            dep_time (float): The scheduled departure time in minutes.
            arr_time (float): The scheduled arrival time in minutes.
            delayed_flights (set, optional): A set of flights that have been delayed. Defaults to None.
        """
        if delayed_flights is None:
            delayed_flights = set()

        aircraft_idx = self.aircraft_id_to_idx[aircraft_id] + 1  # Adjust for state indexing
        scheduled_flights = []
        for j in range(4, self.columns_state_space - 2, 3):  # Adjusted starting index to 5
            existing_flight_id = self.state[aircraft_idx, j]
            existing_dep_time = self.state[aircraft_idx, j + 1]
            existing_arr_time = self.state[aircraft_idx, j + 2]
            if not np.isnan(existing_flight_id) and not np.isnan(existing_dep_time) and not np.isnan(existing_arr_time):
                scheduled_flights.append((existing_flight_id, existing_dep_time, existing_arr_time))

        # Check for conflicts
        for existing_flight_id, existing_dep_time, existing_arr_time in scheduled_flights:
            if existing_flight_id == flight_id:
                continue  # Skip the same flight
            if dep_time < existing_arr_time + MIN_TURN_TIME and arr_time + MIN_TURN_TIME > existing_dep_time:
                # Conflict detected
                delay_new_flight = existing_arr_time + MIN_TURN_TIME - dep_time
                delay_existing_flight = arr_time + MIN_TURN_TIME - existing_dep_time

                delay_new_flight = max(0, delay_new_flight)
                delay_existing_flight = max(0, delay_existing_flight)

                if delay_new_flight <= delay_existing_flight:
                    # Delay new flight
                    dep_time += delay_new_flight
                    arr_time += delay_new_flight
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay_new_flight
                    # Flight times will be updated after conflict resolution
                else:
                    # Delay existing flight
                    if existing_flight_id in delayed_flights:
                        # To avoid infinite loop, delay new flight
                        dep_time += delay_new_flight
                        arr_time += delay_new_flight
                        self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay_new_flight
                    else:
                        delayed_flights.add(existing_flight_id)
                        new_dep_time = existing_dep_time + delay_existing_flight
                        new_arr_time = existing_arr_time + delay_existing_flight
                        self.environment_delayed_flights[existing_flight_id] = self.environment_delayed_flights.get(existing_flight_id, 0) + delay_existing_flight
                        # Update state for existing flight
                        for k in range(4, self.columns_state_space - 2, 3):  # Adjusted starting index to 5
                            if self.state[aircraft_idx, k] == existing_flight_id:
                                self.state[aircraft_idx, k + 1] = new_dep_time
                                self.state[aircraft_idx, k + 2] = new_arr_time
                                break
                        # Update flights_dict for existing flight
                        self.update_flight_times(existing_flight_id, new_dep_time, new_arr_time)
                        # Recursively resolve conflicts for existing flight
                        self.schedule_flight_on_aircraft(aircraft_id, existing_flight_id, new_dep_time, new_arr_time, delayed_flights)

        # Now, update the flight's times in the state
        for j in range(4, self.columns_state_space - 2, 3):  # Adjusted starting index to 5
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

    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, action):
        """Calculates the reward based on the current state of the environment.

        This function evaluates the reward based on conflict resolutions, delays, cancelled flights, and inaction penalties.
        It returns the total reward for the action taken.

        Args:
            resolved_conflicts (set): The set of conflicts that were resolved during the action.
            remaining_conflicts (set): The set of conflicts that remain after the action.
            action (int): The action taken by the agent.

        Returns:
            float: The calculated reward for the action.
        """
        reward = 0

        if DEBUG_MODE_REWARD:
            print("Chosen action:", action)

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
        # print("delay_penalty", delay_penalty)

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
        if action == 0 and len(remaining_conflicts) > 0:
            inaction_penalty = NO_ACTION_PENALTY
            reward -= inaction_penalty
        if DEBUG_MODE_REWARD:
            print(f"  -{inaction_penalty} for inaction with conflicts")

        if DEBUG_MODE_REWARD:
            print("_______________")
            print(f"{reward} total reward for action: {action}")

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

        # Generate breakdowns for each aircraft
        for aircraft_id in self.aircraft_ids:
            if aircraft_id in self.alt_aircraft_dict:
                continue

            breakdown_probability = np.random.uniform(0, 1)  # Set realistic probability
            if breakdown_probability > MIN_BREAKDOWN_PROBABILITY:  # Set a minimum threshold if desired
                max_breakdown_start = total_simulation_minutes - BREAKDOWN_DURATION
                if max_breakdown_start > 0:
                    breakdown_start_minutes = np.random.uniform(0, max_breakdown_start)
                    breakdown_start = self.start_datetime + timedelta(minutes=breakdown_start_minutes)
                    
                    # Generate a random breakdown duration for this specific breakdown
                    breakdown_duration = np.random.uniform(60, 600)  # Random duration between 60 and 600 minutes
                    breakdown_end = breakdown_start + timedelta(minutes=breakdown_duration)

                    self.uncertain_breakdowns[aircraft_id] = [{
                        'StartTime': breakdown_start,
                        'EndTime': breakdown_end,
                        'StartDate': breakdown_start.date(),
                        'EndDate': breakdown_end.date(),
                        'Probability': breakdown_probability,
                        'Resolved': False  # Initially unresolved
                    }]

                    if DEBUG_MODE:
                        print(f"Aircraft {aircraft_id} has an uncertain breakdown scheduled at {breakdown_start} with probability {breakdown_probability:.2f}")

        self.state = self._get_initial_state()

        self.swapped_flights = []  # Reset the swapped flights list
        self.environment_delayed_flights = {}  # Reset the delayed flights list
        self.penalized_delays = set()  # Reset the penalized delays
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()  # Reset penalized cancelled flights

        self.cancelled_flights = set()

        valid_actions = self.get_valid_actions()
        self.action_space = spaces.Discrete(len(valid_actions))
        
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
            if breakdown_probability != 1.0:
                continue  # Only consider unavailabilities with probability 1

            unavail_start = self.state[idx + 1, 2]
            unavail_end = self.state[idx + 1, 3]

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):  # Added -2 to prevent out of bounds
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Check if the flight's departure is in the past (relative to current time)
                        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
                        if flight_dep < current_time_minutes:
                            # Check for conflicts before marking as cancelled
                            if flight_dep < unavail_end and flight_arr > unavail_start:
                                # This flight is in the past and in conflict, mark as cancelled
                                print(f"Flight {flight_id} of aircraft {aircraft_id} with time {flight_dep} to {flight_arr} is cancelled due to departure time in past while having a conflict with unavailability {unavail_start} to {unavail_end}.")
                                self.cancelled_flights.add(flight_id)
                            # If there is no conflict, do not cancel the flight
                            else:
                                print(f"Flight {flight_id} of aircraft {aircraft_id} with time {flight_dep} to {flight_arr} is in the past but has no conflict with unavailability {unavail_start} to {unavail_end}. Not cancelled.")
                            continue  # Skip this flight as it's already processed

                        if flight_id in self.cancelled_flights:
                            continue  # Skip cancelled flights

                        # Check for conflicts between flight and unavailability periods
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id, flight_dep, flight_arr)
                            current_conflicts.add(conflict_identifier)

        if DEBUG_MODE:
            print(f"Current conflicts: {current_conflicts}")
            print(f"Cancelled flights: {self.cancelled_flights}")

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




    def get_valid_actions(self):
        """Generates a list of valid actions for the agent.

        This function returns a list of valid actions, including the option to take no action and actions for each aircraft.

        Returns:
            list: A list of valid actions that the agent can take.
        """
        valid_actions = [0]  # No action is always valid
        # Add all aircraft actions
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break
            valid_actions.append(idx + 1)
        # print(f"Valid actions: {valid_actions}")
        return valid_actions
