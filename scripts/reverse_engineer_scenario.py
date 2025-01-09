import os
import json
from datetime import datetime

def get_default_config_dict():
    """Returns a default configuration dictionary."""
    return {
        'DelayCosts': [
            {'Cabin': 'F', 'Type': 'OB', 'Cost': 100},
            {'Cabin': 'C', 'Type': 'OB', 'Cost': 60},
            {'Cabin': 'Y', 'Type': 'OB', 'Cost': 30},
            {'Cabin': 'F', 'Type': 'IB', 'Cost': 300},
            {'Cabin': 'C', 'Type': 'IB', 'Cost': 180},
            {'Cabin': 'Y', 'Type': 'IB', 'Cost': 90}
        ],
        'CancellationCostsOutbound': [
            {'Cabin': 'F', 'Type': 'OB', 'Cost': 1000},
            {'Cabin': 'C', 'Type': 'OB', 'Cost': 600},
            {'Cabin': 'Y', 'Type': 'OB', 'Cost': 300}
        ],
        'CancellationCostsInbound': [
            {'Cabin': 'F', 'Type': 'IB', 'Cost': 3000},
            {'Cabin': 'C', 'Type': 'IB', 'Cost': 1800},
            {'Cabin': 'Y', 'Type': 'IB', 'Cost': 900}
        ],
        'DowngradingCosts': [
            {'FromCabin': 'F', 'ToCabin': 'C', 'Type': 'OB', 'Cost': 300},
            {'FromCabin': 'F', 'ToCabin': 'Y', 'Type': 'OB', 'Cost': 500},
            {'FromCabin': 'C', 'ToCabin': 'Y', 'Type': 'OB', 'Cost': 200},
            {'FromCabin': 'F', 'ToCabin': 'C', 'Type': 'IB', 'Cost': 900},
            {'FromCabin': 'F', 'ToCabin': 'Y', 'Type': 'IB', 'Cost': 1500},
            {'FromCabin': 'C', 'ToCabin': 'Y', 'Type': 'IB', 'Cost': 600}
        ],
        'PenaltyCosts': [1000],
        'Weights': [1, 1, 1]
    }

def generate_empty_file_with_header(output_path, header):
    """Creates an empty CSV file with just a header and footer."""
    with open(output_path, 'w') as file:
        file.write(header + '\n#')

def generate_empty_files(output_dir):
    """Generates all the empty CSV files needed."""
    # Define headers for each file type
    empty_files = {
        'airports.csv': '%Airport',
        'alt_airports.csv': '%Airport',
        'alt_flights.csv': '%Flight Orig Dest DepTime ArrTime PrevFlight',
        'dist.csv': '%Airport1 Airport2 Distance',
        'itineraries.csv': '%Itinerary Cabin Type Flight1 Flight2 Flight3 Flight4',
        'position.csv': '%Aircraft Airport'
    }
    
    # Create each empty file
    for filename, header in empty_files.items():
        generate_empty_file_with_header(os.path.join(output_dir, filename), header)

def generate_config_file(output_path, config_dict):
    """Generates the config.csv file."""
    with open(output_path, 'w') as file:
        # Write recovery period
        file.write('%RecoveryPeriod\n')
        file.write(f"{config_dict['recovery_start_date']} {config_dict['recovery_start_time']} "
                  f"{config_dict['recovery_end_date']} {config_dict['recovery_end_time']}\n")

        default_config = get_default_config_dict()

        # Write delay costs
        file.write('%DelayCosts\n')
        for cost in default_config['DelayCosts']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write cancellation costs outbound
        file.write('%CancellationCostsOutbound\n')
        for cost in default_config['CancellationCostsOutbound']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write cancellation costs inbound
        file.write('%CancellationCostsInbound\n')
        for cost in default_config['CancellationCostsInbound']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write downgrading costs
        file.write('%DowngradingCosts\n')
        for cost in default_config['DowngradingCosts']:
            file.write(f"{cost['FromCabin']} {cost['ToCabin']} {cost['Type']} {cost['Cost']} ")
        file.write('\n')

        # Write penalty costs
        file.write('%PenaltyCosts\n')
        for cost in default_config['PenaltyCosts']:
            file.write(f"{cost} ")
        file.write('\n')

        # Write weights
        file.write('%Weights\n')
        for weight in default_config['Weights']:
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
            model = aircraft_id.split('#')[0]  # e.g., B737 from B737#1
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

def generate_scenario_files(scenario_data, output_dir):
    """Generates all CSV files for a single scenario."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each CSV file
    generate_config_file(
        os.path.join(output_dir, 'config.csv'),
        {
            'recovery_start_date': scenario_data['recovery_start_date'],
            'recovery_start_time': scenario_data['recovery_start_time'],
            'recovery_end_date': scenario_data['recovery_end_date'],
            'recovery_end_time': scenario_data['recovery_end_time']
        }
    )
    
    generate_aircraft_file(
        os.path.join(output_dir, 'aircraft.csv'),
        scenario_data
    )
    
    generate_alt_aircraft_file(
        os.path.join(output_dir, 'alt_aircraft.csv'),
        scenario_data['disruptions']
    )
    
    generate_flights_file(
        os.path.join(output_dir, 'flights.csv'),
        scenario_data
    )
    
    generate_rotations_file(
        os.path.join(output_dir, 'rotations.csv'),
        scenario_data,
        scenario_data['recovery_start_date']
    )
    
    # Generate empty files
    generate_empty_files(output_dir)

def reverse_engineer_scenario(json_path, output_dir):
    """
    Takes a scenario JSON file and recreates the CSV files that were originally used to create it.
    Creates a separate folder for each scenario.
    
    Args:
        json_path (str): Path to the scenario JSON file
        output_dir (str): Directory where to create the scenario folders
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process each scenario
    for scenario_name, scenario_data in data['outputs'].items():
        scenario_dir = os.path.join(output_dir, scenario_name)
        generate_scenario_files(scenario_data, scenario_dir)
        print(f"Generated files for scenario: {scenario_name}")

if __name__ == "__main__":
    import argparse
    
    # Default paths if no args provided
    default_json_path = os.path.join("logs", "scenarios", "scenario_folder_scenario_4-2.json")
    default_output_dir = os.path.join("data", "REVERSED", "scenario_4-2")
    
    parser = argparse.ArgumentParser(description='Reverse engineer scenario CSV files from JSON')
    parser.add_argument('--json_path', default=default_json_path,
                      help='Path to the scenario JSON file (default: %(default)s)')
    parser.add_argument('--output_dir', default=default_output_dir,
                      help='Directory where to create the scenario folders (default: %(default)s)')
    
    args = parser.parse_args()
    reverse_engineer_scenario(args.json_path, args.output_dir)