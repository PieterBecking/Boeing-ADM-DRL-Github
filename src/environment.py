import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config import *
from scripts.utils import *
import time
import random
from typing import Dict, List, Tuple
import os

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type):
        """Initializes the AircraftDisruptionEnv class.

        Args:
            aircraft_dict (dict): Dictionary containing aircraft information.
            flights_dict (dict): Dictionary containing flight information.
            rotations_dict (dict): Dictionary containing rotation information.
            alt_aircraft_dict (dict): Dictionary containing alternative aircraft information.
            config_dict (dict): Dictionary containing configuration information.
            env_type (str): Type of environment ('myopic' or 'proactive', 'reactive').
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
        self.timestep_minutes = TIMESTEP_HOURS * 60

        # Aircraft information and indexing
        self.aircraft_ids = list(aircraft_dict.keys())
        self.aircraft_id_to_idx = {aircraft_id: idx for idx, aircraft_id in enumerate(self.aircraft_ids)}

        self.conflicted_flights = {}  # Tracks flights in conflict due to past departure and prob == 1.0
    

        # Flight information and indexing
        # if flights_dict is empty, flights_dict is empty
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

        self.info_after_step = {}

        # Initialize a dictionary to store unavailabilities
        self.unavailabilities_dict = {}

        # Initialize the environment state without generating probabilities
        self.current_datetime = self.start_datetime
        self.state = self._get_initial_state()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()
        self.eligible_flights_for_not_being_cancelled_when_disruption_happens = self.get_initial_conflicts_with_deptime_before_unavail_start()

        self.scenario_wide_delay_minutes = 0
        self.scenario_wide_cancelled_flights = 0
        self.scenario_wide_steps = 0
        self.scenario_wide_resolved_conflicts = 0
        self.scenario_wide_solution_slack = 0
        self.scenario_wide_tail_swaps = 0
        self.scenario_wide_initial_disrupted_flights_list = self.get_current_conflicts()
        self.scenario_wide_actual_disrupted_flights = len(self.get_current_conflicts())
        # print(f"*********scenario_wide_actual_disrupted_flights: {self.scenario_wide_actual_disrupted_flights}")
        # print(f"*********scenario_wide_initial_disrupted_flights_list: {self.scenario_wide_initial_disrupted_flights_list}")

        self.scenario_wide_reward_components = {
            "delay_penalty_total": 0,
            "cancel_penalty": 0,
            "inaction_penalty": 0,
            "proactive_bonus": 0,
            "time_penalty": 0,
            "final_conflict_resolution_reward": 0,
        }

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
        for i in range(0, 2):  # Start at 0 and step by 2 for the half of the columns
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

            # # In the myopic env, the info for uncertain breakdowns is not shown
            # if breakdown_probability != 1.0 and self.env_type == 'myopic':
            #     breakdown_probability = np.nan  # Set to NaN if not 1.00
            #     unavail_start_minutes = np.nan
            #     unavail_end_minutes = np.nan

            # # In the proactive env, the info for unrealized breakdowns is also not shown
            # if np.isnan(breakdown_probability) or breakdown_probability == 0.00:  # Added check for 0.00
            #     breakdown_probability = np.nan  # Set to NaN if not 1.00
            #     unavail_start_minutes = np.nan
            #     unavail_end_minutes = np.nan


            # # In the reactive env, the info for unrealized breakdowns is also not shown
            # if self.env_type == 'reactive':  # Added check for 0.00
            #     breakdown_probability = np.nan  # Set to NaN if not 1.00
            #     unavail_start_minutes = np.nan
            #     unavail_end_minutes = np.nan

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
        """Processes the observation by applying env_type-specific masking.
        Does NOT modify internal state or unavailabilities_dict.
        Returns an observation suitable for the agent."""
        state_to_observe = state.copy()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            # Get the real, internal probability and times from the state
            breakdown_probability = state_to_observe[idx + 1, 1]
            unavail_start_minutes = state_to_observe[idx + 1, 2]
            unavail_end_minutes = state_to_observe[idx + 1, 3]

            # Make copies for observation only
            obs_breakdown_probability = breakdown_probability
            obs_unavail_start_minutes = unavail_start_minutes
            obs_unavail_end_minutes = unavail_end_minutes

            # Apply env_type logic to these observed values ONLY
            if self.env_type == 'reactive':
                # Reactive sees no info about disruptions
                obs_breakdown_probability = np.nan
                obs_unavail_start_minutes = np.nan
                obs_unavail_end_minutes = np.nan
            elif self.env_type == 'myopic':
                # Myopic only sees if probability == 1.00
                if obs_breakdown_probability != 1.0:
                    obs_breakdown_probability = np.nan
                    obs_unavail_start_minutes = np.nan
                    obs_unavail_end_minutes = np.nan
            elif self.env_type == 'proactive':
                # Proactive sees everything (no masking needed)
                pass

            # Assign the masked values back to the observation copy
            state_to_observe[idx + 1, 1] = obs_breakdown_probability
            state_to_observe[idx + 1, 2] = obs_unavail_start_minutes
            state_to_observe[idx + 1, 3] = obs_unavail_end_minutes

        # Create a mask where 1 indicates valid values, 0 indicates NaN
        mask = np.where(np.isnan(state_to_observe), 0, 1)
        # Replace NaN with a dummy value
        state_to_observe = np.nan_to_num(state_to_observe, nan=DUMMY_VALUE)

        # Flatten both state and mask
        state_flat = state_to_observe.flatten()
        mask_flat = mask.flatten()

        # Use get_action_mask to generate the action mask
        action_mask = self.get_action_mask()

        # Return the observation dictionary without modifying internal structures
        obs_with_mask = {
            'state': state_flat,
            'action_mask': action_mask
        }
        return obs_with_mask, state_to_observe

    
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
            print_state_nicely(self.state, self.env_type)
            print("")

        # Extract the action values from the action
        flight_action, aircraft_action = self.map_index_to_action(action_index)

        # Check if the flight action is valid
        if DEBUG_MODE_ACTION:
            pass
        
        if flight_action != 0:
            # Check if the flight_action exists in our valid flight IDs
            if flight_action not in self.flight_id_to_idx.keys():
                raise ValueError(f"Invalid flight action: {flight_action}")
            else:
                pass
        else:
            pass

        # Validate the action
        self.validate_action(flight_action, aircraft_action)

        # Print the processed action and chosen action
        if DEBUG_MODE:
            print(f"Processed action: {action_index} of type: {type(action_index)}")
            print(f"Chosen action: flight {flight_action}, aircraft {aircraft_action}")

        # Get pre-action conflicts
        pre_action_conflicts = self.get_current_conflicts()
        unresolved_uncertainties = self.get_unresolved_uncertainties()

        # print("-----1-----")
        # print(f"pre_action_conflicts: {pre_action_conflicts}")

        # Process uncertainties before handling flight operations
        self.process_uncertainties()
        self.scenario_wide_steps += 1

        if len(pre_action_conflicts) == 0 and len(unresolved_uncertainties) == 0:
            # Handle the case when there are no conflicts
            # print("-----2-----")
            processed_state, reward, terminated, truncated, info = self.handle_no_conflicts(flight_action, aircraft_action)
        else:
            # print("-----3-----")
            # Resolve the conflict based on the action
            processed_state, reward, terminated, truncated, info = self.handle_flight_operations(flight_action, aircraft_action, pre_action_conflicts)

        # Update the processed state after processing uncertainties
        processed_state, _ = self.process_observation(self.state)

        terminated = self.check_termination_criteria()
        if DEBUG_MODE_STOPPING_CRITERIA:
            print(f"checked and terminated: {terminated}")

        return processed_state, reward, terminated, truncated, info


    def check_termination_criteria(self):
        """
        Checks if the stopping criteria are met:

        Stopping criteria:
        1. There are no uncertainties in the system anymore.
           (All probabilities are either np.nan, 0.0, or 1.0.)
        2. There is no overlap of breakdowns (Probability == 1.0) and flights.

        Returns:
            bool: True if both criteria are met, False otherwise.
        """
        # Check that all probabilities are either nan, 0.0, or 1.0
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if not (np.isnan(prob) or prob == 0.0 or prob == 1.0):
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"    prob: {prob} is not nan, 0.0, or 1.0, for aircraft {aircraft_id}, so termination = False")
                return False

        # Check that there is no overlap between flights and breakdowns with prob == 1.0
        # get_current_conflicts() only returns conflicts for probability == 1.0 breakdowns.
        if len(self.get_current_conflicts_with_prob_1()) > 0:
            if DEBUG_MODE_STOPPING_CRITERIA:
                print(f"    get_current_conflicts_with_prob_1() returns {self.get_current_conflicts_with_prob_1()}, so termination = False")
            return False

        return True


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
                self.alt_aircraft_dict[aircraft_id]['Probability'] = new_prob
                # print(f"*** new_prob: {new_prob}")

                if self.env_type == "proactive":
                    self.state[idx + 1, 1] = new_prob
                    # print(f"*** updated proactive state: {self.state[idx + 1, 1]}")

                if DEBUG_MODE:
                    print(f"Aircraft {aircraft_id}: Probability updated from {prob:.2f} to {new_prob:.2f}")

                if self.current_datetime + self.timestep >= breakdown_start_time:
                    # print(f"Rolling the dice for breakdown with updated probability {new_prob} starting at {breakdown_start_time}")
                    
                    
                    # print("state before rolling the dice:")
                    # if self.env_type == "proactive":
                    #     print_state_nicely_proactive(self.state)
                    #     print("")
                    # else:
                    #     print_state_nicely_myopic(self.state)
                    #     print("")

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
                        
                        # Count affected flights for this aircraft in the initial disrupted flights list
                        affected_flights = 0
                        for conflict in self.scenario_wide_initial_disrupted_flights_list:
                            # Extract just the flight ID and aircraft ID from the conflict tuple
                            if isinstance(conflict, tuple):
                                conflict_aircraft_id = conflict[0]  # First element is aircraft ID
                                if conflict_aircraft_id == aircraft_id:
                                    affected_flights += 1
                        
                        # Subtract the count of affected flights from the total
                        self.scenario_wide_actual_disrupted_flights -= affected_flights

                        # make the prop, start, end all np.nan for both env types
                        self.state[idx + 1, 1] = np.nan
                        self.state[idx + 1, 2] = np.nan
                        self.state[idx + 1, 3] = np.nan
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 0.00

                    # Update alt_aircraft_dict if necessary
                    if aircraft_id in self.alt_aircraft_dict:
                        if isinstance(self.alt_aircraft_dict[aircraft_id], dict):
                            # print("*** alt_aircraft_dict[aircraft_id] is a dict")
                            self.alt_aircraft_dict[aircraft_id] = [self.alt_aircraft_dict[aircraft_id]]
                        elif isinstance(self.alt_aircraft_dict[aircraft_id], str):
                            # print("*** alt_aircraft_dict[aircraft_id] is a string")
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
                            # print(f"*** breakdown_info['Probability']: {breakdown_info['Probability']}")

                    # print("state after rolling the dice:")
                    # if self.env_type == "proactive":
                    #     print_state_nicely_proactive(self.state)
                    #     print("")
                    # else:
                    #     print_state_nicely_myopic(self.state)
                    #     print("")



    def handle_no_conflicts(self, flight_action, aircraft_action):
        """Handles the case when there are no conflicts in the current state.

        This function updates the current datetime, checks if the episode is terminated,
        updates the state, and returns the appropriate outputs.
        """
        # print("*** HANDLING NO CONFLICTS")

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
                processed_state, _ = self.process_observation(self.state)
                truncated = False
                # print("*** 1")
                reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, terminated)
                # print("*** doing this")
                return processed_state, reward, terminated, truncated, {}

        self.current_datetime = next_datetime
        self.state = self._get_initial_state()
        


        # Since there are no conflicts, return the new state with zero reward
        terminated, reason = self._is_done()
        truncated = False

        # Call _calculate_reward even when there are no conflicts
        done = terminated or truncated
        
        # print("*** 2")
        reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, done)


        processed_state, _ = self.process_observation(self.state)

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

        terminated = self.check_termination_criteria()

        if flight_action == 0:
            # No action taken
            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            # print("*** 3")
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state, _ = self.process_observation(self.state)
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

            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            # print("*** 4")
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state, _ = self.process_observation(self.state)
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
                terminated, reason = self._is_done()
                truncated = False

                done = terminated or truncated
                # print("*** 5")
                reward = self._calculate_reward(pre_action_conflicts, pre_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

                processed_state, _ = self.process_observation(self.state)
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
                self.scenario_wide_tail_swaps += 1  # Increment counter when flight is moved to new aircraft

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

            terminated, reason = self._is_done()
            terminated = self.check_termination_criteria()
            truncated = False
            done = terminated or truncated
            # print("*** done = ", done)
            # print("*** terminated = ", terminated)
            # print("*** 6")
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated=done)

            if terminated:
                # print("*** 6 - but with terminated = ", terminated)
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state, _ = self.process_observation(self.state)
            # print("*** returning: ", processed_state, reward, terminated, truncated, {})
            return processed_state, reward, terminated, truncated, {}

    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, current_aircraft_id, arr_time=None, delayed_flights=None, secondary=False):
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
            # A flight overlaps if it doesn't end before the disruption starts
            if flight_end > unavail_start_time:
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
            print(f"current_ac_is_same_as_target_ac: {current_ac_is_same_as_target_ac}") 

        # If flight is on current aircraft and completes before disruption, keep it there
        if current_ac_is_same_as_target_ac and not has_unavail_overlap:
            if DEBUG_MODE_SCHEDULING:
                print("Flight is on current aircraft and completes before disruption - keeping original schedule")
            self.something_happened = False
            return

        # Handle unavailability overlap
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

        # Get all flights on this aircraft sorted by departure time
        scheduled_flights = []
        for j in range(4, self.columns_state_space - 2, 3):
            existing_flight_id = self.state[aircraft_idx, j]
            existing_dep_time = self.state[aircraft_idx, j + 1]
            existing_arr_time = self.state[aircraft_idx, j + 2]
            if not np.isnan(existing_flight_id) and not np.isnan(existing_dep_time) and not np.isnan(existing_arr_time):
                if existing_flight_id != flight_id:  # Don't include the flight we're scheduling
                    scheduled_flights.append((existing_flight_id, existing_dep_time, existing_arr_time))
        
        # Sort flights by departure time
        scheduled_flights.sort(key=lambda x: x[1])

        # Find where our flight should be inserted
        insert_idx = 0
        for i, (_, existing_dep_time, _) in enumerate(scheduled_flights):
            if dep_time < existing_dep_time:
                break
            insert_idx = i + 1

        # Insert our flight at the correct position
        scheduled_flights.insert(insert_idx, (flight_id, dep_time, arr_time))

        # Now process all flights in sequence to ensure proper spacing
        for i in range(len(scheduled_flights)):
            current_flight = scheduled_flights[i]
            current_flight_id = current_flight[0]
            current_dep_time = current_flight[1]
            current_arr_time = current_flight[2]

            # Check for overlap with previous flight
            if i > 0:
                prev_flight = scheduled_flights[i-1]
                prev_arr_time = prev_flight[2]
                
                # If there's an overlap, delay the current flight
                if current_dep_time < prev_arr_time + MIN_TURN_TIME:
                    new_dep_time = prev_arr_time + MIN_TURN_TIME
                    new_arr_time = new_dep_time + (current_arr_time - current_dep_time)
                    
                    # Update the flight times
                    if current_flight_id == flight_id:
                        dep_time = new_dep_time
                        arr_time = new_arr_time
                    else:
                        # Update the other flight's times
                        self.update_flight_times(current_flight_id, new_dep_time, new_arr_time)
                        for k in range(4, self.columns_state_space - 2, 3):
                            if self.state[aircraft_idx, k] == current_flight_id:
                                self.state[aircraft_idx, k + 1] = new_dep_time
                                self.state[aircraft_idx, k + 2] = new_arr_time
                                break
                    
                    # Update the scheduled_flights list with new times
                    scheduled_flights[i] = (current_flight_id, new_dep_time, new_arr_time)
                    
                    # Track the delay
                    original_dep = parse_time_with_day_offset(
                        self.flights_dict[current_flight_id]['DepTime'], 
                        self.start_datetime
                    )
                    original_dep_minutes = (original_dep - self.earliest_datetime).total_seconds() / 60
                    delay = new_dep_time - original_dep_minutes
                    self.environment_delayed_flights[current_flight_id] = self.environment_delayed_flights.get(current_flight_id, 0) + delay

        # Finally, update the state for our flight
        flight_placed = False
        for j in range(4, self.columns_state_space - 2, 3):
            if self.state[aircraft_idx, j] == flight_id:
                self.state[aircraft_idx, j + 1] = dep_time
                self.state[aircraft_idx, j + 2] = arr_time
                flight_placed = True
                break
            elif np.isnan(self.state[aircraft_idx, j]) and not flight_placed:
                self.state[aircraft_idx, j] = flight_id
                self.state[aircraft_idx, j + 1] = dep_time
                self.state[aircraft_idx, j + 2] = arr_time
                flight_placed = True
                break

        # Update flights_dict
        self.update_flight_times(flight_id, dep_time, arr_time)

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

    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated):
        """Calculates the reward based on the current state of the environment.

        The reward consists of several components:
        1. Delay Penalty: Penalty for each minute of delay introduced
        2. Cancellation Penalty: Penalty for each newly cancelled flight
        3. Inaction Penalty: Penalty for taking no action when conflicts exist
        4. Proactive Bonus: Reward for taking actions well before flight departure
        5. Time Penalty: Small penalty for each minute of simulation time
        6. Final Resolution Reward: Bonus for resolving real conflicts at scenario end

        Args:
            resolved_conflicts (set): The set of conflicts that were resolved during the action
            remaining_conflicts (set): The set of conflicts that remain after the action
            flight_action (int): The flight action taken by the agent
            aircraft_action (int): The aircraft action taken by the agent
            original_flight_action_departure_time (str): The departure time of the flight being acted upon
            terminated (bool): Whether the episode has ended

        Returns:
            float: The calculated reward for the action
        """
        reward = 0

        if DEBUG_MODE_REWARD:
            print("")
            print(f"Calculating reward for action: flight {flight_action}, aircraft {aircraft_action}")

        # 1. Delay Penalty: Penalize additional minutes of delay
        delay_penalty_minutes = sum(
            self.environment_delayed_flights[flight_id] - self.penalized_delays.get(flight_id, 0)
            for flight_id in self.environment_delayed_flights
        )
        self.scenario_wide_delay_minutes += delay_penalty_minutes
        delay_penalty_total = min(delay_penalty_minutes * DELAY_MINUTE_PENALTY, MAX_DELAY_PENALTY)

        if DEBUG_MODE_REWARD:
            print(f"  -{delay_penalty_total} penalty for {delay_penalty_minutes} minutes of additional delay (capped at {MAX_DELAY_PENALTY})")

        # 2. Cancellation Penalty: Penalize newly cancelled flights
        new_cancellations = {
            flight_id for flight_id in self.cancelled_flights if flight_id not in self.penalized_cancelled_flights
        }
        cancellation_penalty_count = len(new_cancellations)
        cancel_penalty = cancellation_penalty_count * CANCELLED_FLIGHT_PENALTY
        self.scenario_wide_cancelled_flights += cancellation_penalty_count
        self.penalized_cancelled_flights.update(new_cancellations)

        if DEBUG_MODE_REWARD:
            print(f"  -{cancel_penalty} penalty for {cancellation_penalty_count} new cancelled flights: {new_cancellations}")

        # 3. Inaction Penalty: Penalize doing nothing when conflicts exist
        inaction_penalty = NO_ACTION_PENALTY if flight_action == 0 and remaining_conflicts else 0

        if DEBUG_MODE_REWARD:
            print(f"  -{inaction_penalty} penalty for inaction with remaining conflicts")

        # 4. Proactive Bonus: Reward for acting ahead of time
        proactive_bonus = 0
        time_to_departure = None
        if flight_action != 0 and self.something_happened:
            original_dep_time = parse_time_with_day_offset(original_flight_action_departure_time, self.start_datetime)
            current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
            action_time = current_time_minutes - TIMESTEP_HOURS * 60
            time_to_departure = (original_dep_time - self.earliest_datetime).total_seconds() / 60 - action_time
            proactive_bonus = max(0, time_to_departure * AHEAD_BONUS_PER_MINUTE)

        if DEBUG_MODE_REWARD and proactive_bonus > 0:
            print(f"  +{proactive_bonus} bonus for proactive action ({time_to_departure:.1f} minutes ahead)")

        # 5. Time Penalty: Small penalty for simulation progression
        time_penalty_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
        time_penalty = time_penalty_minutes * TIME_MINUTE_PENALTY

        if DEBUG_MODE_REWARD:
            print(f"  -{time_penalty} penalty for time progression")

        # 6. Final Resolution Reward: Bonus for resolving real conflicts at scenario end
        final_conflict_resolution_reward = 0
        if terminated:
            # Count resolved conflicts for non-cancelled flights with probability 1.00
            final_resolved_count = 0
            resolved_flights = []
            for (aircraft_id, flight_id) in self.initial_conflict_combinations:
                if self.unavailabilities_dict[aircraft_id]['Probability'] == 1.00 and flight_id not in self.cancelled_flights:
                    final_resolved_count += 1
                    resolved_flights.append(flight_id)

            final_conflict_resolution_reward = final_resolved_count * RESOLVED_CONFLICT_REWARD
            self.scenario_wide_resolved_conflicts += final_resolved_count

            if DEBUG_MODE_REWARD:
                print(f"  +{final_conflict_resolution_reward} final reward for resolving {final_resolved_count} real (non-cancelled) conflicts at scenario end: {resolved_flights}")

            # Calculate scenario-wide solution slack
            self._calculate_scenario_wide_solution_slack()

        # Update penalized delays for next iteration
        for flight_id, delay in self.environment_delayed_flights.items():
            self.penalized_delays[flight_id] = delay

        # Calculate total reward
        reward = (
            - delay_penalty_total
            - cancel_penalty
            - inaction_penalty
            + proactive_bonus
            - time_penalty
            + final_conflict_resolution_reward
        )

        # Update scenario-wide reward components
        self.scenario_wide_reward_components.update({
            "delay_penalty_total": self.scenario_wide_reward_components["delay_penalty_total"] - delay_penalty_total,
            "cancel_penalty": self.scenario_wide_reward_components["cancel_penalty"] - cancel_penalty,
            "inaction_penalty": self.scenario_wide_reward_components["inaction_penalty"] - inaction_penalty,
            "proactive_bonus": self.scenario_wide_reward_components["proactive_bonus"] + proactive_bonus,
            "time_penalty": self.scenario_wide_reward_components["time_penalty"] - time_penalty,
            "final_conflict_resolution_reward": self.scenario_wide_reward_components["final_conflict_resolution_reward"] + final_conflict_resolution_reward
        })

        # Store reward components in state
        self.state[0, 4] = reward
        self.state[0, 6] = delay_penalty_total
        self.state[0, 7] = cancel_penalty
        self.state[0, 8] = inaction_penalty
        self.state[0, 9] = proactive_bonus
        self.state[0, 10] = time_penalty

        # Round final reward
        reward = round(reward, 1)

        if DEBUG_MODE_REWARD:
            print("--------------------------------")
            print(f"Total reward: {reward}")
            print("--------------------------------")

        # Store step information
        self.info_after_step = {
            "total_reward": reward,
            "something_happened": self.something_happened,
            "current_time_minutes": time_penalty_minutes,
            "resolved_conflicts_count": len(resolved_conflicts),
            "remaining_conflicts_count": len(remaining_conflicts),
            "delay_penalty_minutes": delay_penalty_minutes,
            "delay_penalty_total": delay_penalty_total,
            "delay_penalty_capped": delay_penalty_total == MAX_DELAY_PENALTY,
            "cancelled_flights_count": cancellation_penalty_count,
            "cancellation_penalty": cancel_penalty,
            "inaction_penalty": inaction_penalty,
            "proactive_bonus": proactive_bonus,
            "time_to_departure_minutes": time_to_departure,
            "time_penalty": time_penalty,
            "flight_action": flight_action,
            "aircraft_action": aircraft_action,
            "original_departure_time": original_flight_action_departure_time,
        }

        return reward

    def _calculate_scenario_wide_solution_slack(self):
        """Calculate the scenario-wide solution slack based on flight schedules."""
        # Extract all flights and their departure/arrival times
        all_dep_times = []
        all_arr_times = []
        for f_id, f_info in self.flights_dict.items():
            dep_dt = parse_time_with_day_offset(f_info['DepTime'], self.start_datetime)
            arr_dt = parse_time_with_day_offset(f_info['ArrTime'], self.start_datetime)
            all_dep_times.append(dep_dt)
            all_arr_times.append(arr_dt)

        if not all_dep_times:
            self.scenario_wide_solution_slack = 0.0
            return

        # Calculate time horizon
        earliest_dep = min(all_dep_times)
        latest_arr = max(all_arr_times)
        horizon = max(int((latest_arr - earliest_dep).total_seconds() / 60), 1)

        # Organize flights by aircraft
        aircraft_flights = {ac: [] for ac in self.aircraft_ids}
        for f_id, f_info in self.flights_dict.items():
            if f_id not in self.rotations_dict:
                continue

            ac_id = self.rotations_dict[f_id]['Aircraft']
            if ac_id not in aircraft_flights:
                continue

            dep_dt = parse_time_with_day_offset(f_info['DepTime'], self.start_datetime)
            arr_dt = parse_time_with_day_offset(f_info['ArrTime'], self.start_datetime)
            flight_duration = int((arr_dt - dep_dt).total_seconds() / 60)
            aircraft_flights[ac_id].append(flight_duration)

        # Calculate slack per aircraft
        aircraft_slacks = []
        for ac_id in self.aircraft_ids:
            if not aircraft_flights[ac_id]:
                ac_slack = 0.0
            else:
                total_flight_time = sum(aircraft_flights[ac_id])
                ac_slack = total_flight_time / horizon
            aircraft_slacks.append(ac_slack)

        # Calculate average slack across all aircraft
        self.scenario_wide_solution_slack = sum(aircraft_slacks) / len(aircraft_slacks) if aircraft_slacks else 0.0

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

        self.initial_conflict_combinations = self.get_initial_conflicts()

        self.swapped_flights = []  # Reset the swapped flights list
        self.environment_delayed_flights = {}  # Reset the delayed flights list
        self.penalized_delays = {}  # Reset the penalized delays
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()  # Reset penalized cancelled flights

        self.cancelled_flights = set()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()

        # Process the state into an observation as a NumPy array
        processed_state, _ = self.process_observation(self.state)

        if DEBUG_MODE:
            print(f"State space shape: {self.state.shape}")
            print(f"Type of processed state: {type(processed_state)}")
            print(processed_state)
        return processed_state, {}

    def get_initial_conflicts(self):
        """Retrieves the initial conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability greater than 0.0.

        Returns:
            set: A set of conflicts currently present in the initial state of the environment.
        """
        initial_conflicts = set()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability <= 0.0 or np.isnan(breakdown_probability):
                continue  # Only consider unavailabilities with probability > 0.0

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Skip cancelled flights
                        if flight_id in self.cancelled_flights:
                            continue

                        # Check for overlaps with unavailability periods
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id)
                            initial_conflicts.add(conflict_identifier)

        return initial_conflicts


    def get_initial_conflicts_with_deptime_before_unavail_start(self):
        """Retrieves the initial conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability greater than 0.0.

        Returns:
            set: A set of conflicts currently present in the initial state of the environment.
        """
        initial_conflicts = set()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability <= 0.0 or np.isnan(breakdown_probability):
                continue  # Only consider unavailabilities with probability > 0.0

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Skip cancelled flights
                        if flight_id in self.cancelled_flights:
                            continue

                        # Check for overlaps with unavailability periods
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            if flight_dep < unavail_start:
                                conflict_identifier = (aircraft_id, flight_id)
                                initial_conflicts.add(conflict_identifier)

        return initial_conflicts


    def get_current_conflicts(self):
        """Retrieves the current conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering only unavailabilities with probability 
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts = set()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability == 0.0:  # Only consider unavailability with probability 
                continue  # Skip if probability is not 0.0

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

                        # Check for overlaps with unavailability periods with prob 
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id, flight_dep, flight_arr)
                            current_conflicts.add(conflict_identifier)

        return current_conflicts
    

    def get_current_conflicts_with_prob_1(self):
        """Retrieves the current conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering only unavailabilities with probability 1.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts_with_prob_1 = set()

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
                            current_conflicts_with_prob_1.add(conflict_identifier)

        return current_conflicts_with_prob_1

    def check_flight_disruption_overlaps(self):
        """Checks if there are any overlaps between flights and disruptions (certain or uncertain).
        
        Returns:
            bool: True if there are overlaps, False otherwise.
        """
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            # Get disruption info
            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
            prob = self.unavailabilities_dict[aircraft_id]['Probability']

            # Skip if no disruption period defined
            if np.isnan(unavail_start) or np.isnan(unavail_end):
                continue

            # Check flights on this aircraft
            for j in range(4, self.columns_state_space - 2, 3):
                flight_id = self.state[idx + 1, j]
                flight_dep = self.state[idx + 1, j + 1]
                flight_arr = self.state[idx + 1, j + 2]

                if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                    # Skip cancelled flights
                    if flight_id in self.cancelled_flights:
                        continue

                    # Skip past flights
                    current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
                    if flight_arr < current_time_minutes:
                        continue

                    # Check for overlap with disruption period
                    if flight_dep < unavail_end and flight_arr > unavail_start:
                        return True  # Found an overlap

        return False  # No overlaps found

    def termination_criteria_met(self):
        """
        Checks if the stopping criteria are met.

        Stopping criteria:
        - There are no uncertainties in the system anymore, i.e., for all aircraft:
          Probability is either NaN, exactly 0.0, or exactly 1.0.
        - AND there is no overlap of BREAKDOWNS (with Probability == 1.00) and flights.

        Returns:
            bool: True if the stopping criteria are met, False otherwise.
        """
        # Check if all probabilities are either nan, 0.0, or 1.0
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if not (np.isnan(prob) or prob == 0.0 or prob == 1.0):
                return False

        # Check if there is no overlap of breakdowns (prob == 1.00) and flights
        # If check_flight_disruption_overlaps() returns False, that means no overlap.
        if self.check_flight_disruption_overlaps():
            return False

        return True


    def _is_done(self):
        """Checks if the episode is finished.

        The episode is considered done if:
        1. Current time has reached or exceeded the end time, OR
        2. There are no overlaps between flights and disruptions (regardless of uncertainty status)

        Returns:
            tuple: (bool, str) indicating if the episode is done and the reason.
        """
        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        
        # Check for any overlaps between flights and disruptions
        if not self.check_flight_disruption_overlaps():
            return True, "No remaining overlaps between flights and disruptions."
        
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

        # For reactive environment, only allow 0,0 action if no current conflicts with prob==1.00
        if self.env_type == 'reactive':
            reactive_allowed_to_take_action = False
            current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

            # Find earliest disrupted flight departure time and earliest disruption start time
            earliest_disrupted_dep = float('inf')
            earliest_disruption_start = float('inf')
            earliest_disruption_end = float('inf')

            # Check each aircraft for disruptions with probability 1
            for idx, aircraft_id in enumerate(self.aircraft_ids):
                breakdown_prob = self.unavailabilities_dict[aircraft_id]['Probability']
                if breakdown_prob == 1.0:
                    unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
                    unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
                    # print(f"*** unavail_start: {unavail_start}")
                    # print(f"*** unavail_end: {unavail_end}")
                    if not np.isnan(unavail_start):
                        
                        earliest_disruption_start = min(earliest_disruption_start, unavail_start)
                        
                        # Check if current time is inside disruption period
                        if current_time_minutes >= unavail_start and current_time_minutes <= unavail_end:
                            reactive_allowed_to_take_action = True
                            break
                            
                        # Check flights assigned to this aircraft for departures during disruption
                        for j in range(4, self.columns_state_space - 2, 3):
                            flight_id = self.state[idx + 1, j]
                            dep_time = self.state[idx + 1, j + 1]
                            arr_time = self.state[idx + 1, j + 2]
                            if not np.isnan(flight_id) and not np.isnan(dep_time):
                                # print(f"*** flight_id: {flight_id}")
                                # print(f"*** dep_time: {dep_time}")
                                # print(f"*** arr_time: {arr_time}")
                                # there is any overlap (whichever starts first, check the deptime/startime)
                                if dep_time < unavail_end and arr_time > unavail_start:
                                    earliest_disrupted_dep = min(earliest_disrupted_dep, dep_time)
                                    # print(f"*** earliest disrupted dep: {earliest_disrupted_dep} of flight {flight_id}")

            # Allow reactive action if approaching either critical time
            current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
            earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
            
            if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
                reactive_allowed_to_take_action = True
            # print(f"**** allowed to take action: {reactive_allowed_to_take_action}")
            if not reactive_allowed_to_take_action:
                # Reset mask to all zeros except for 0,0 action
                action_mask[:] = 0
                action_mask[0] = 1  # Only allow 0,0 action

        # print(f"*** Action mask: {action_mask}")
        # # print the value of the action together with the mask
        # for i in range(len(action_mask)):
        #     print(f"Action {i}: {action_mask[i]}")
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

class AircraftDisruptionOptimizer(AircraftDisruptionEnv):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict):
        # Initialize the environment with 'myopic' type (since we want to see all conflicts)
        super().__init__(aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type='reactive')
        
        # Initialize solution tracking
        self.solution = {
            "objective_value": 0,
            "assignments": {},
            "cancellations": [],
            "delays": {},
            "total_delay_minutes": 0,
            "statistics": {
                "runtime": 0,
                "gap": 0,
                "node_count": 0,
                "status": "In Progress"
            }
        }
        
        # Track start time for runtime calculation
        self.start_time = None
        
        # Track action history
        self.action_history = []

        self.scenario_wide_reward_total = 0
        

    def solve(self):
        """Solve the problem step by step using the environment's mechanics"""
        self.start_time = time.time()
        observation, _ = self.reset()  # Reset environment to initial state
        
        terminated = False
        total_reward = 0
        step_count = 0
        
        while not terminated:
            step_count += 1
            # print(f"\nStep {step_count}:")
            
            # Get current conflicts
            conflicts = self.get_current_conflicts()
            # print(f"Current conflicts: {conflicts}")
            
            if not conflicts:
                # No conflicts - take no action
                action = self.map_action_to_index(0, 0)  # No-op action
                # print("No conflicts - taking no-op action (0, 0)")
            else:
                # Choose the best action for the current state using valid action mask
                action = self.select_best_action()
                flight_action, aircraft_action = self.map_index_to_action(action)
                # print(f"Selected action: index={action} (flight={flight_action}, aircraft={aircraft_action})")
            
            # Take the action in the environment
            observation, reward, terminated, truncated, info = self.step(action)
            total_reward += reward
            # print(f"Action result: reward={reward}, terminated={terminated}")
            
            # Record action history
            flight_action, aircraft_action = self.map_index_to_action(action)
            self.action_history.append({
                'step': step_count,
                'action_index': action,
                'flight': flight_action,
                'aircraft': aircraft_action,
                'reward': reward,
                'conflicts': len(conflicts)
            })
            
            # Update solution based on the action taken
            self.update_solution(info)
        
        # Print action history summary
        # print("\nAction History Summary:")
        # print("----------------------")
        # print(f"{'Step':>4} | {'Flight':>6} | {'Aircraft':>8} | {'Reward':>8} | {'Conflicts':>9}")
        # print("-" * 45)
        # for entry in self.action_history:
        #     print(f"{entry['step']:4d} | {entry['flight']:6} | {entry['aircraft']:8} | {entry['reward']:8.1f} | {entry['conflicts']:9d}")
        # print("-" * 45)
        # print(f"Total Reward: {total_reward:.1f}")

        self.scenario_wide_reward_total = total_reward
        
        # Finalize solution
        self.solution["objective_value"] = -total_reward  # Convert reward to cost
        self.solution["statistics"]["runtime"] = time.time() - self.start_time
        self.solution["statistics"]["status"] = "Complete"
        
        return self.solution

    def select_best_action(self):
        """Select the best action from the valid action space using a greedy heuristic"""
        best_action = None
        best_score = float('-inf')
        # For reactive environment, only allow 0,0 action if no current conflicts with prob==1.00
        reactive_allowed_to_take_action = False
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        # Find earliest disrupted flight departure time and earliest disruption start time 
        earliest_disrupted_dep = float('inf')
        earliest_disruption_start = float('inf')
        earliest_disruption_end = float('inf')

        # Check each aircraft for disruptions with probability 1
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            breakdown_prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_prob == 1.0:
                unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
                unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
                if not np.isnan(unavail_start):
                    earliest_disruption_start = min(earliest_disruption_start, unavail_start)
                    
                    # Check if current time is inside disruption period
                    if current_time_minutes >= unavail_start and current_time_minutes <= unavail_end:
                        reactive_allowed_to_take_action = True
                        break
                        
                    # Check flights assigned to this aircraft for departures during disruption
                    for j in range(4, self.columns_state_space - 2, 3):
                        flight_id = self.state[idx + 1, j]
                        dep_time = self.state[idx + 1, j + 1]
                        arr_time = self.state[idx + 1, j + 2]
                        if not np.isnan(flight_id) and not np.isnan(dep_time):
                            if dep_time < unavail_end and arr_time > unavail_start:
                                earliest_disrupted_dep = min(earliest_disrupted_dep, dep_time)

        # Allow reactive action if approaching either critical time
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
        
        if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
            reactive_allowed_to_take_action = True

        if not reactive_allowed_to_take_action:
            return self.map_action_to_index(0, 0)  # Only allow 0,0 action
        
        # Get valid actions using the environment's action mask
        action_mask = self.get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]
        # print(f"\nEvaluating {len(valid_actions)} valid actions:")
        
        # Try each valid action and evaluate its impact
        for action in valid_actions:
            flight_action, aircraft_action = self.map_index_to_action(action)
            # Create a copy of the environment to simulate the action
            env_copy = copy.deepcopy(self)
            
            # Take the action in the copied environment
            _, reward, _, _, _ = env_copy.step(action)
            
            # print(f"  Action {action} (flight={flight_action}, aircraft={aircraft_action}): reward={reward}")
            
            # Update best action if this one has better reward
            if reward > best_score:
                best_score = reward
                best_action = action
                print(f"    -> New best action (reward={reward})")
        
        # If no good action found, take no action (should be included in valid_actions)
        if best_action is None:
            best_action = self.map_action_to_index(0, 0)
            # print("No good action found, defaulting to no-op action (0, 0)")
        else:
            flight_action, aircraft_action = self.map_index_to_action(best_action)
            # print(f"\nChosen best action: index={best_action} (flight={flight_action}, aircraft={aircraft_action}) with reward={best_score}")
        
        return best_action

    def update_solution(self, info):
        """Update the solution dictionary based on the action taken"""
        # Update cancellations
        if 'cancelled_flights_count' in info and info['cancelled_flights_count'] > 0:
            self.solution['cancellations'] = list(self.cancelled_flights)
        
        # Update assignments
        if 'flight_action' in info and 'aircraft_action' in info:
            flight_id = info['flight_action']
            aircraft_idx = info['aircraft_action']
            if flight_id != 0 and aircraft_idx != 0:
                aircraft_id = self.aircraft_ids[aircraft_idx - 1]
                if aircraft_id != self.rotations_dict[flight_id]['Aircraft']:
                    self.solution['assignments'][flight_id] = aircraft_id
        
        # Update delays
        if self.environment_delayed_flights:
            self.solution['delays'] = {k: v for k, v in self.environment_delayed_flights.items()}
            self.solution['total_delay_minutes'] = sum(self.environment_delayed_flights.values())