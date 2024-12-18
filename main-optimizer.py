import os
from scripts.utils import load_scenario_data
from src.environment import AircraftDisruptionOptimizer
from scripts.visualizations import StatePlotter
from datetime import datetime

def main():
    scenario_folder = "data/Training/6ac-700-diverse/mixed_high_Scenario_004"
    
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
    optimizer = AircraftDisruptionOptimizer(
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
