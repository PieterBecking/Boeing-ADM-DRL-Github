import os
import copy
import numpy as np
import time
from scripts.utils import load_scenario_data
from src.environment import AircraftDisruptionEnv
from scripts.visualizations import StatePlotter
from datetime import datetime


class AircraftDisruptionExactInference(AircraftDisruptionEnv):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict):
        # Initialize the environment with 'myopic' type (since we want to see all conflicts)
        super().__init__(aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type='proactive')
        
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
        
    def select_best_action(self):
        """Select the best action using beam search with width 3.
        
        This method looks ahead until terminal states, keeping only the top 3 paths at each step.
        Returns the first action of the best-performing path.
        """
        BEAM_WIDTH = 10
        
        # Get valid actions using the environment's action mask
        action_mask = self.get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]

        # For first step only, print rewards for all actions
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        self.step_counter += 1
        
        # Initialize beam with empty paths
        # Each path is (sequence_of_actions, total_reward, env_state, terminated)
        current_beam = [([], 0, copy.deepcopy(self), False)]
        
        # Keep track of best complete path
        best_complete_path = None
        best_complete_reward = float('-inf')
        
        # Track total sequences considered
        total_sequences_considered = 0
        
        # For first step only, evaluate and print top 10 actions
        if self.step_counter == 1:
            print("\nTop 10 actions being considered in first step:")
            action_rewards = []
            for action in valid_actions:
                env_copy = copy.deepcopy(self)
                flight_action, ac_action = self.map_index_to_action(action)
                _, reward, _, _, _ = env_copy.step(action)
                action_rewards.append((action, flight_action, ac_action, reward))
            
            # Sort by reward and print top 10
            action_rewards.sort(key=lambda x: x[3], reverse=True)
            for i, (action, flight, ac, reward) in enumerate(action_rewards[:10]):
                print(f"{i+1}. Flight {flight}, Aircraft {ac}, Expected Reward: {reward}")
            print()
        
        while current_beam:
            next_beam = []
            
            # Try extending each path in current beam
            for action_sequence, total_reward, env_state, terminated in current_beam:
                if terminated:
                    # If this is a complete path, update best if it's better
                    if total_reward > best_complete_reward:
                        best_complete_path = action_sequence
                        best_complete_reward = total_reward
                    continue
                
                # Get valid actions for this state
                action_mask = env_state.get_action_mask()
                valid_actions = np.where(action_mask == 1)[0]
                
                # Try each valid action
                candidates = []
                for action in valid_actions:
                    total_sequences_considered += 1
                    env_copy = copy.deepcopy(env_state)
                    _, reward, terminated, _, _ = env_copy.step(action)
                    new_sequence = action_sequence + [action]
                    new_total_reward = total_reward + reward
                    candidates.append((new_sequence, new_total_reward, env_copy, terminated))
                
                # Sort candidates by reward and keep top BEAM_WIDTH
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_beam.extend(candidates[:BEAM_WIDTH])
            
            # Sort all new paths by reward and keep top BEAM_WIDTH
            next_beam.sort(key=lambda x: x[1], reverse=True)
            current_beam = next_beam[:BEAM_WIDTH]
            
            # Check if all paths in beam are terminated
            if all(terminated for _, _, _, terminated in current_beam):
                # Update best complete path if any path in beam is better
                best_in_beam = max(current_beam, key=lambda x: x[1])
                if best_in_beam[1] > best_complete_reward:
                    best_complete_path = best_in_beam[0]
                    best_complete_reward = best_in_beam[1]
                break
        
        print(f"Total action sequences considered: {total_sequences_considered}")
        
        # If we found a complete path, return its first action
        if best_complete_path and best_complete_path:
            return best_complete_path[0]
        
        # If no complete path found (shouldn't happen), return no-op action
        return self.map_action_to_index(0, 0)

    def solve(self):
        """Solve the problem step by step using the environment's mechanics"""
        self.start_time = time.time()
        observation, _ = self.reset()  # Reset environment to initial state
        
        terminated = False
        total_reward = 0
        step_count = 0
        
        while not terminated:
            step_count += 1
            
            # Choose the best action for the current state using beam search
            action = self.select_best_action()
            flight_action, aircraft_action = self.map_index_to_action(action)
            print(f"Selected action: (flight={flight_action}, aircraft={aircraft_action})")
            
            # Take the action in the environment
            observation, reward, terminated, truncated, info = self.step(action)
            print(f"Action result: reward={reward}, terminated={terminated}, something_happened={self.something_happened}, something_happened_testing={self.something_happened_testing}")
            total_reward += reward
            
            # Record action history
            self.action_history.append({
                'step': step_count,
                'action_index': action,
                'flight': flight_action,
                'aircraft': aircraft_action,
                'reward': reward,
                'conflicts': len(self.get_current_conflicts())
            })
            
            # Update solution based on the action taken
            self.update_solution(info)
        
        self.scenario_wide_reward_total = total_reward
        
        # Finalize solution
        self.solution["objective_value"] = total_reward 
        self.solution["statistics"]["runtime"] = time.time() - self.start_time
        self.solution["statistics"]["status"] = "Complete"
        
        return self.solution

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


def main():
    scenario_folder = "data/Training/6ac-10-deterministic/Scenario_00009"
    
    # Load scenario data first
    data_dict = load_scenario_data(scenario_folder)
    aircraft_dict = data_dict['aircraft']
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_aircraft_dict = data_dict['alt_aircraft']
    config_dict = data_dict['config']
    
    # Parse recovery period times
    recovery_period = config_dict['RecoveryPeriod']
    start_datetime = datetime.strptime(f"{recovery_period['StartDate']} {recovery_period['StartTime']}", '%d/%m/%y %H:%M')
    end_datetime = datetime.strptime(f"{recovery_period['EndDate']} {recovery_period['EndTime']}", '%d/%m/%y %H:%M')
    
    # Initialize optimizer with loaded data
    optimizer = AircraftDisruptionExactInference(
        aircraft_dict=aircraft_dict,
        flights_dict=flights_dict,
        rotations_dict=rotations_dict,
        alt_aircraft_dict=alt_aircraft_dict,
        config_dict=config_dict
    )
    
    # Initialize state plotter
    state_plotter = StatePlotter(
        aircraft_dict=aircraft_dict,
        flights_dict=flights_dict,
        rotations_dict=rotations_dict,
        alt_aircraft_dict=alt_aircraft_dict,
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )

    
    # Solve the problem step by step
    print("Solving optimization problem step by step...")
    solution = optimizer.solve()
    
    # Print detailed results
    print("\nOptimization Results:")
    print(f"Objective value: {solution['objective_value']:.2f}")
    print(f"\nSolution statistics:")
    print(f"  Runtime: {solution['statistics']['runtime']:.2f} seconds")
    print(f"  Status: {solution['statistics']['status']}")
    
    print(f"\nSolution summary:")
    print(f"  Cancelled flights: {len(solution['cancellations'])}")
    print(f"  Total delay minutes: {solution['total_delay_minutes']}")
    print(f"  Number of reassignments: {len(solution['assignments'])}")
    
    if solution['cancellations']:
        print("\nCancelled flights:")
        for flight_id in solution['cancellations']:
            print(f"  - Flight {flight_id}")
    
    if solution['delays']:
        print("\nDelayed flights:")
        for flight_id, delay in solution['delays'].items():
            print(f"  - Flight {flight_id}: {delay} minutes")
    
    if solution['assignments']:
        print("\nFlight reassignments:")
        for flight_id, new_aircraft in solution['assignments'].items():
            original_aircraft = rotations_dict[flight_id]['Aircraft']
            if new_aircraft != original_aircraft:
                print(f"  - Flight {flight_id}: {original_aircraft} -> {new_aircraft}")
    
    # Convert optimizer solution to state plotter format
    swapped_flights = [(flight_id, new_aircraft) for flight_id, new_aircraft in solution['assignments'].items()]
    environment_delayed_flights = set(solution['delays'].keys())
    cancelled_flights = set(solution['cancellations'])
    
    # Update flights dict with delays
    updated_flights_dict = flights_dict.copy()
    for flight_id, delay in solution['delays'].items():
        if flight_id in updated_flights_dict:
            flight_info = updated_flights_dict[flight_id]
            # Note: The actual delay application is handled by the plot_state function
            flight_info['Delay'] = delay

    scenario_wide_delay_minutes = optimizer.scenario_wide_delay_minutes
    print(f"Scenario wide delay minutes: {scenario_wide_delay_minutes}")

    scenario_wide_resolved_conflicts = optimizer.scenario_wide_resolved_conflicts
    print(f"Scenario wide resolved conflicts: {scenario_wide_resolved_conflicts}")

    scenario_wide_cancelled_flights = optimizer.scenario_wide_cancelled_flights
    print(f"Scenario wide cancelled flights: {scenario_wide_cancelled_flights}")


if __name__ == "__main__":
    main()
