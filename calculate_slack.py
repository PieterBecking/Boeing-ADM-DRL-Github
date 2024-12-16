import pandas as pd
import os
import numpy as np

def calculate_scenario_slack(scenario_path):
    # Read the necessary files
    flights_df = pd.read_csv(os.path.join(scenario_path, 'flights.csv'))
    rotations_df = pd.read_csv(os.path.join(scenario_path, 'rotations.csv'))
    
    # Merge flights with rotations to get aircraft assignments
    merged_df = pd.merge(flights_df, rotations_df, on='flight_number')
    
    # Calculate horizon H (in minutes)
    earliest_departure = merged_df['departure_time'].min()
    latest_arrival = merged_df['arrival_time'].max()
    H = latest_arrival - earliest_departure
    
    # Calculate T_ac for each aircraft
    aircraft_times = {}
    for ac in merged_df['aircraft_id'].unique():
        ac_flights = merged_df[merged_df['aircraft_id'] == ac]
        T_ac = sum(ac_flights['arrival_time'] - ac_flights['departure_time'])
        slack_ac = T_ac / H if H > 0 else 0
        aircraft_times[ac] = slack_ac
    
    # Calculate overall scenario slack
    scenario_slack = np.mean(list(aircraft_times.values()))
    
    return scenario_slack, aircraft_times

def main():
    base_path = 'data/Training/6ac-700-diverse'
    results = {}
    
    # Process all scenario folders
    for scenario_folder in sorted(os.listdir(base_path)):
        if scenario_folder.startswith('deterministic_na_Scenario_'):
            scenario_path = os.path.join(base_path, scenario_folder)
            try:
                scenario_slack, aircraft_slacks = calculate_scenario_slack(scenario_path)
                results[scenario_folder] = {
                    'overall_slack': scenario_slack,
                    'aircraft_slacks': aircraft_slacks
                }
                print(f"{scenario_folder}: Overall Slack = {scenario_slack:.3f}")
            except Exception as e:
                print(f"Error processing {scenario_folder}: {str(e)}")
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'scenario': scenario,
            'overall_slack': data['overall_slack'],
            **{f'aircraft_{ac}_slack': slack for ac, slack in data['aircraft_slacks'].items()}
        }
        for scenario, data in results.items()
    ])
    
    results_df.to_csv('scenario_slack_values.csv', index=False)
    print("\nResults saved to scenario_slack_values.csv")

if __name__ == "__main__":
    main() 