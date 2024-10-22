import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config import *
from scripts.utils import *

MIN_TURN_TIME = 0  # Minimum turnaround time in minutes

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict):
        super(AircraftDisruptionEnv, self).__init__()

        # Constants for environment configuration
        self.max_aircraft = MAX_AIRCRAFT
        self.columns_state_space = COLUMNS_STATE_SPACE
        self.rows_state_space = ROWS_STATE_SPACE

        self.config_dict = config_dict

        # Define the recovery period based on provided configuration
        start_date = config_dict['RecoveryPeriod']['StartDate']
        start_time = config_dict['RecoveryPeriod']['StartTime']
        end_date = config_dict['RecoveryPeriod']['EndDate']
        end_time = config_dict['RecoveryPeriod']['EndTime']
        self.start_datetime = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
        self.end_datetime = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
        self.timestep = timedelta(hours=1)

        # Aircraft information and indexing
        self.aircraft_ids = list(aircraft_dict.keys())
        self.aircraft_id_to_idx = {aircraft_id: idx for idx, aircraft_id in enumerate(self.aircraft_ids)}

        # Filter out flights with '+' in DepTime (next day flights)
        # print(flights_dict)
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


        # Initialize the environment state
        self.reset()

    def _get_initial_state(self):
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

            # Store aircraft index starting from 1
            state[idx + 1, 0] = idx + 1

            # Add unavailability times
            if aircraft_id in self.alt_aircraft_dict:
                unavail_info = self.alt_aircraft_dict[aircraft_id]
                unavail_start_time = datetime.strptime(unavail_info['StartDate'] + ' ' + unavail_info['StartTime'], '%d/%m/%y %H:%M')
                unavail_end_time = datetime.strptime(unavail_info['EndDate'] + ' ' + unavail_info['EndTime'], '%d/%m/%y %H:%M')

                unavail_start_minutes = (unavail_start_time - self.earliest_datetime).total_seconds() / 60
                unavail_end_minutes = (unavail_end_time - self.earliest_datetime).total_seconds() / 60

                state[idx + 1, 1] = unavail_start_minutes
                state[idx + 1, 2] = unavail_end_minutes
            else:
                state[idx + 1, 1] = np.nan  # No unavailability

            # Gather flight times for this aircraft
            flight_times = []
            for flight_id, rotation_info in self.rotations_dict.items():
                if flight_id in self.flights_dict and rotation_info['Aircraft'] == aircraft_id:
                    # Use updated times from flights_dict (which already include delays)
                    flight_info = self.flights_dict[flight_id]
                    dep_time_str = flight_info['DepTime']
                    arr_time_str = flight_info['ArrTime']

                    # Parse the departure and arrival times
                    dep_time = parse_time_with_day_offset(dep_time_str, self.start_datetime)
                    arr_time = parse_time_with_day_offset(arr_time_str, self.start_datetime)

                    # Append flight id, departure time, and arrival time to the flight_times list
                    dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                    arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60
                    flight_times.append((flight_id, dep_time_minutes, arr_time_minutes))

            # Sort the flight_times by dep_time_minutes
            flight_times.sort(key=lambda x: x[1])

            # Flatten the flight times list and store in the state matrix
            flight_times_flat = [time for times in flight_times for time in times]
            state[idx + 1, 3:3 + len(flight_times_flat)] = flight_times_flat

        # Check the state space for conflicting flights (unavailability and scheduled flights)
        earliest_conf_flight_id = None
        earliest_conf_dep_time = None
        earliest_conf_aircraft_idx = None  # To store the index (idx) of the conflicting aircraft

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            unavail_start = state[idx + 1, 1]
            unavail_end = state[idx + 1, 2]

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for flight conflicts with unavailability
                for j in range(3, self.columns_state_space, 3):
                    flight_id = state[idx + 1, j]
                    flight_dep = state[idx + 1, j + 1]

                    if not np.isnan(flight_dep):
                        # Skip this flight if it is in the past (considered cancelled)
                        if flight_dep < current_time_minutes:
                            if DEBUG_MODE:
                                print(f"Flight {flight_id} is cancelled (dep time: {flight_dep}, current time: {current_time_minutes})")
                            self.cancelled_flights.add(flight_id)
                            continue  # Skip this flight
                        if unavail_start <= flight_dep and unavail_end >= flight_dep:
                            # Conflict detected, and flight is not cancelled
                            if flight_id not in self.cancelled_flights:
                                if earliest_conf_dep_time is None or flight_dep < earliest_conf_dep_time:
                                    earliest_conf_dep_time = flight_dep
                                    earliest_conf_flight_id = flight_id
                                    earliest_conf_aircraft_idx = idx + 1  # Store the idx (row index) of the conflicting aircraft

        # Store the earliest conflicting aircraft index and flight ID in the state matrix
        state[0, 0] = earliest_conf_aircraft_idx  # Store the index of the earliest conflicting aircraft (idx + 1 for row indexing)
        state[0, 1] = earliest_conf_flight_id     # Store the earliest conflicting flight ID

        if DEBUG_MODE:
            print(f"Earliest conflicting flight: {earliest_conf_flight_id} on aircraft idx {earliest_conf_aircraft_idx}")




        return state


    def process_observation(self, state):
        """Processes the observation: applies mask and flattens the state and mask."""
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

        agent_acted = False
        info = {} 

        while not agent_acted:
            pre_action_conflicts = self.get_current_conflicts()

            if len(pre_action_conflicts) == 0:
                next_datetime = self.current_datetime + self.timestep
                if next_datetime >= self.end_datetime:
                    terminated = True
                    truncated = False
                    processed_state = self.process_observation(self.state)
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

                unavail_start = self.state[idx + 1, 1]
                unavail_end = self.state[idx + 1, 2]

                if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                    for j in range(3, self.columns_state_space, 3):
                        flight_id = self.state[idx + 1, j]
                        flight_dep = self.state[idx + 1, j + 1]
                        flight_arr = self.state[idx + 1, j + 2]

                        if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                            if unavail_start <= flight_arr and unavail_end >= flight_dep:
                                if (earliest_conflict_time is None or flight_dep < earliest_conflict_time) and flight_id not in self.cancelled_flights:
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

                terminated = self._is_done()
                truncated = False

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

                terminated = self._is_done()
                truncated = False

                processed_state = self.process_observation(self.state)
                return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

            selected_aircraft_id = self.aircraft_ids[action_value - 1]
            selected_idx = self.aircraft_id_to_idx[selected_aircraft_id]

            for j in range(3, self.columns_state_space, 3):
                if self.state[conflicting_idx + 1, j] == conflicting_flight_id:
                    conf_dep_time = self.state[conflicting_idx + 1, j + 1]
                    conf_arr_time = self.state[conflicting_idx + 1, j + 2]
                    break

            if selected_aircraft_id == conflicting_aircraft:
                unavail_end = self.state[conflicting_idx + 1, 2]
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

                terminated = self._is_done()
                truncated = False

                processed_state = self.process_observation(self.state)
                return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}

            if conflicting_flight_id:
                if conflicting_flight_id in self.cancelled_flights:
                    if DEBUG_MODE:
                        print(f"Skipping tail swap for cancelled flight {conflicting_flight_id}")
                else:
                    self.swapped_flights.append((conflicting_flight_id, selected_aircraft_id))

                    self.rotations_dict[conflicting_flight_id]['Aircraft'] = selected_aircraft_id

                    for j in range(3, self.columns_state_space, 3):
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

            terminated = self._is_done()
            truncated = False

            processed_state = self.process_observation(self.state)
            return np.array(processed_state, dtype=np.float32), reward, terminated, truncated, {}




    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, arr_time, delayed_flights=None):
        if delayed_flights is None:
            delayed_flights = set()

        aircraft_idx = self.aircraft_id_to_idx[aircraft_id] + 1  # Adjust for state indexing
        scheduled_flights = []
        for j in range(3, self.columns_state_space, 3):
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
                        for k in range(3, self.columns_state_space, 3):
                            if self.state[aircraft_idx, k] == existing_flight_id:
                                self.state[aircraft_idx, k + 1] = new_dep_time
                                self.state[aircraft_idx, k + 2] = new_arr_time
                                break
                        # Update flights_dict for existing flight
                        self.update_flight_times(existing_flight_id, new_dep_time, new_arr_time)
                        # Recursively resolve conflicts for existing flight
                        self.schedule_flight_on_aircraft(aircraft_id, existing_flight_id, new_dep_time, new_arr_time, delayed_flights)

        # Now, update the flight's times in the state
        for j in range(3, self.columns_state_space, 3):
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
        if delay_penalty_total > RESOLVED_CONFLICT_REWARD:
            capped_delay_penalty = True
            delay_penalty_total = RESOLVED_CONFLICT_REWARD
        else:
            delay_penalty_total = delay_penalty

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


    def _is_done(self):
        return self.current_datetime >= self.end_datetime or len(self.get_current_conflicts()) == 0
    def reset(self, seed=None, options=None):
        self.current_datetime = self.start_datetime

        self.actions_taken = set()  # Reset the actions taken at the beginning of each episode

        # Deep copy the initial dictionaries to reset them
        self.aircraft_dict = copy.deepcopy(self.initial_aircraft_dict)
        self.flights_dict = copy.deepcopy(self.initial_flights_dict)
        self.rotations_dict = copy.deepcopy(self.initial_rotations_dict)
        self.alt_aircraft_dict = copy.deepcopy(self.initial_alt_aircraft_dict)

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
        
        return np.array(processed_state, dtype=np.float32), {}  # Return processed observation as a NumPy array

    
    def get_current_conflicts(self):
        """Returns the set of conflicts in the current state, excluding past flights (which are considered cancelled)."""
        current_conflicts = set()
        self.cancelled_flights = set()  # Track cancelled flights

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break
            if not np.isnan(self.state[idx + 1, 1]) and not np.isnan(self.state[idx + 1, 2]):
                # Check for conflicts between flights and unavailability periods
                for j in range(3, self.columns_state_space, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Check if the flight's departure is in the past (relative to current time)
                        if flight_dep < (self.current_datetime - self.earliest_datetime).total_seconds() / 60:
                            # This flight is in the past, mark as cancelled
                            if DEBUG_MODE:
                                print(f"Flight {flight_id} is cancelled (dep time: {flight_dep}, current time: {self.current_datetime})")
                            self.cancelled_flights.add(flight_id)
                            continue  # Skip this flight as it's already cancelled

                        # Check for conflicts between flight and unavailability periods
                        if flight_dep < self.state[idx + 1, 2] and flight_arr > self.state[idx + 1, 1]:
                            conflict_identifier = (aircraft_id, flight_dep, flight_arr)
                            current_conflicts.add(conflict_identifier)

        if DEBUG_MODE:
            print(f"Current conflicts: {current_conflicts}")
            print(f"Cancelled flights: {self.cancelled_flights}")

        return current_conflicts




    def get_valid_actions(self):
        valid_actions = [0]  # No action is always valid
        # Add all aircraft actions
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break
            valid_actions.append(idx + 1)
        # print(f"Valid actions: {valid_actions}")
        return valid_actions
