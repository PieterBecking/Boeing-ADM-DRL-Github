import os
import json
from datetime import datetime

def generate_config_file(output_path, config_dict):
    """Generates the config.csv file."""
    with open(output_path, 'w') as file:
        # Write recovery period
        file.write('%RecoveryPeriod\n')
        file.write(f"{config_dict['recovery_start_date']} {config_dict['recovery_start_time']} "
                  f"{config_dict['recovery_end_date']} {config_dict['recovery_end_time']}\n")

        # Write delay costs
        file.write('%DelayCosts\n')
        for cost in config_dict['config_dict']['DelayCosts']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write cancellation costs outbound
        file.write('%CancellationCostsOutbound\n')
        for cost in config_dict['config_dict']['CancellationCostsOutbound']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write cancellation costs inbound
        file.write('%CancellationCostsInbound\n')
        for cost in config_dict['config_dict']['CancellationCostsInbound']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write downgrading costs
        file.write('%DowngradingCosts\n')
        for cost in config_dict['config_dict']['DowngradingCosts']:
            file.write(f"{cost['FromCabin']} {cost['ToCabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write penalty costs
        file.write('%PenaltyCosts\n')
        for cost in config_dict['config_dict']['PenaltyCosts']:
            file.write(f"{cost} ")
        file.write('\n')

        # Write weights
        file.write('%Weights\n')
        for weight in config_dict['config_dict']['Weights']:
            file.write(f"{weight} ")
        file.write('\n')

        file.write('#')

def generate_aircraft_file(output_path, aircraft_data):
    """Generates the aircraft.csv file."""
    with open(output_path, 'w') as file:
        file.write('%Aircraft Model Family Config Dist Cost/h TurnRound Transit Orig Maint\n')
        for aircraft_id in sorted(aircraft_data['aircraft_ids']):
            # Since we don't have the full aircraft data in the JSON, we'll use default values
            # These should be adjusted based on your specific needs
            model = aircraft_id.split('#')[0]  # e.g., A320 from A320#1
            file.write(f"{aircraft_id} {model} {model}Family Standard 6000 5000 30 60 AMS AMS\n")
        file.write('#')

def generate_alt_aircraft_file(output_path, disruptions_data):
    """Generates the alt_aircraft.csv file."""
    with open(output_path, 'w') as file:
        file.write('%Aircraft StartDate StartTime EndDate EndTime Probability\n')
        for disruption in disruptions_data['disruptions']:
            aircraft_id = disruption['aircraft_id']
            file.write(f"{aircraft_id} {disruption['start_date']} {disruption['start_time']} "
                      f"{disruption['end_date']} {disruption['end_time']} {disruption['probability']:.2f}\n")
        file.write('#')

def generate_flights_file(output_path, flights_data):
    """Generates the flights.csv file."""
    with open(output_path, 'w') as file:
        file.write('%Flight Orig Dest DepTime ArrTime PrevFlight\n')
        for flight_id, flight_info in sorted(flights_data['flights'].items(), key=lambda x: int(x[0])):
            file.write(f"{flight_id} {flight_info['Orig']} {flight_info['Dest']} "
                      f"{flight_info['DepTime']} {flight_info['ArrTime']} {flight_info['PrevFlight']}\n")
        file.write('#')

def generate_rotations_file(output_path, flights_data, recovery_start_date):
    """Generates the rotations.csv file."""
    with open(output_path, 'w') as file:
        file.write('%Flight DepDate Aircraft\n')
        for flight_id, flight_info in sorted(flights_data['flights'].items(), key=lambda x: int(x[0])):
            file.write(f"{flight_id} {recovery_start_date} {flight_info['Aircraft']}\n")
        file.write('#')

def reverse_engineer_scenario(json_path, output_dir):
    """
    Takes a scenario JSON file and recreates the CSV files that were originally used to create it.
    
    Args:
        json_path (str): Path to the scenario JSON file
        output_dir (str): Directory where the CSV files should be created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the first scenario from outputs (they should all have the same config)
    first_scenario = next(iter(data['outputs'].values()))
    
    # Generate each CSV file
    generate_config_file(
        os.path.join(output_dir, 'config.csv'),
        {
            'recovery_start_date': first_scenario['recovery_start_date'],
            'recovery_start_time': first_scenario['recovery_start_time'],
            'recovery_end_date': first_scenario['recovery_end_date'],
            'recovery_end_time': first_scenario['recovery_end_time'],
            'config_dict': data['inputs']['config_dict']
        }
    )
    
    generate_aircraft_file(
        os.path.join(output_dir, 'aircraft.csv'),
        first_scenario
    )
    
    generate_alt_aircraft_file(
        os.path.join(output_dir, 'alt_aircraft.csv'),
        first_scenario['disruptions']
    )
    
    generate_flights_file(
        os.path.join(output_dir, 'flights.csv'),
        first_scenario
    )
    
    generate_rotations_file(
        os.path.join(output_dir, 'rotations.csv'),
        first_scenario,
        first_scenario['recovery_start_date']
    )
if __name__ == "__main__":
    import argparse
    import os
    
    # Default paths if no args provided
    default_json_path = os.path.join("logs", "scenarios", "scenario_folder_scenario_4-2.json")
    default_output_dir = os.path.join("data", "REVERSED", "scenario_4-2")
    
    parser = argparse.ArgumentParser(description='Reverse engineer scenario CSV files from JSON')
    parser.add_argument('--json_path', default=default_json_path,
                      help='Path to the scenario JSON file (default: %(default)s)')
    parser.add_argument('--output_dir', default=default_output_dir,
                      help='Directory where to create the CSV files (default: %(default)s)')
    
    args = parser.parse_args()
    reverse_engineer_scenario(args.json_path, args.output_dir)