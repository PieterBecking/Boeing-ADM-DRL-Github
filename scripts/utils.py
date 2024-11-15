import os
import numpy as np
from datetime import datetime, timedelta
import re
from src.config import *
import csv


# File reader with comment filtering
def read_csv_with_comments(file_path):
    """Reads a CSV file and skips comment lines (lines starting with '%') and stops at '#'."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_lines = []
        for line in lines:
            if line.startswith('#'):
                break
            if not line.startswith('%'):
                data_lines.append(line.strip())

        # Return an empty list if no data lines were found
        if not data_lines:
            return []
        
        return data_lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    

def load_scenario_data(scenario_folder):
    file_keys = ['aircraft', 'airports', 'alt_aircraft', 'alt_airports', 'alt_flights', 'config', 'dist', 'flights', 'itineraries', 'position', 'rotations']
    file_paths = {key: os.path.join(scenario_folder, f"{key}.csv") for key in file_keys}

    data_dict = {}
    file_parsing_functions = {
        'config': FileParsers.parse_config,
        'airports': FileParsers.parse_airports,
        'dist': FileParsers.parse_dist,
        'flights': FileParsers.parse_flights,
        'aircraft': FileParsers.parse_aircraft,
        'rotations': FileParsers.parse_rotations,
        'itineraries': FileParsers.parse_itineraries,
        'position': FileParsers.parse_position,
        'alt_flights': FileParsers.parse_alt_flights,
        'alt_aircraft': FileParsers.parse_alt_aircraft,
        'alt_airports': FileParsers.parse_alt_airports
    }

    # Iterate over each file and process it using the correct parsing function
    for file_type, file_path in file_paths.items():
        file_lines = read_csv_with_comments(file_path)
        if file_lines:
            parse_function = file_parsing_functions.get(file_type)
            if parse_function:
                parsed_data = parse_function(file_lines)
                data_dict[file_type] = parsed_data
            else:
                print(f"No parser available for file type: {file_type}")
        else:
            data_dict[file_type] = None

    return data_dict

# Clear file content
def clear_file(file_name):
    """Clears the content of a file."""
    with open(file_name, 'w') as file:
        file.write('')

# Convert time to string
def convert_time_to_str(current_datetime, time_obj):
    time_str = time_obj.strftime('%H:%M')
    if time_obj.date() > current_datetime.date():
        time_str += ' +1'
    return time_str

def parse_time_with_day_offset(time_str, reference_date):
    """
    Parses time and adds a day offset if '+1' is present, or if the arrival time 
    is earlier than the departure time (indicating a flight crosses midnight).
    """
    # Check if '+1' exists in the time string
    if '+1' in time_str:
        # Remove the '+1' and strip any whitespace
        time_str = time_str.replace('+1', '').strip()
        time_obj = datetime.strptime(time_str, '%H:%M')
        # Add 1 day to the time
        return datetime.combine(reference_date, time_obj.time()) + timedelta(days=1)
    else:
        # No '+1', parse the time normally
        time_obj = datetime.strptime(time_str, '%H:%M')
        parsed_time = datetime.combine(reference_date, time_obj.time())
        
        # If the parsed time is earlier than the reference time, it's the next day
        if parsed_time < reference_date:
            parsed_time += timedelta(days=1)
            
        return parsed_time



# Print state
def print_state_nicely_myopic(state):
    # First print the information row in tabular form
    info_row = state[0]
    # print("\nSimulation Info:")
    # print("┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐")
    print("│ Target Aircraft    │ Target Flight      │ Current Time       │ Time Until End     │")
    # print("├────────────────────┼────────────────────┼────────────────────┼────────────────────┤")
    print(f"│ {int(info_row[0]) if not np.isnan(info_row[0]) else '-':^19}│ "
          f"{int(info_row[1]) if not np.isnan(info_row[1]) else '-':^19}│ "
          f"{int(info_row[2]) if not np.isnan(info_row[2]) else '-':^19}│ "
          f"{int(info_row[3]) if not np.isnan(info_row[3]) else '-':^19}│")
    # print("└────────────────────┴────────────────────┴────────────────────┴────────────────────┘")
    print("")  # Empty line for separation
    
    # Define column widths with extra space for non-flight headers
    ac_width = 4
    prob_width = 6
    start_width = 6
    end_width = 5
    flight_width = 5
    time_width = 5
    
    # Generate headers dynamically with proper spacing
    headers = [
        f"{'AC':>{ac_width}}", 
        f"{'Prob':>{prob_width}}", 
        f"{'Start':>{start_width}}", 
        f"{'End':>{end_width}}"
    ]
    
    # Add flight headers with proper spacing
    for i in range(1, MAX_FLIGHTS_PER_AIRCRAFT + 1):
        headers.extend([
            f"| {'F'+str(i):>{flight_width}}", 
            f"{'Dep'+str(i):>{time_width}}", 
            f"{'Arr'+str(i):>{time_width}}"
        ])
    
    # Print headers
    print(" ".join(headers))
    
    # Print state rows with matching alignment
    formatted_rows = []
    for row in state[1:]:
        formatted_values = []
        for i, x in enumerate(row):
            # Add vertical line before flight groups
            if i >= 4 and (i - 4) % 3 == 0:
                formatted_values.append("|")
                
            if np.isnan(x):
                formatted_values.append(f"{'-':>{time_width}}" if i >= 4 else 
                                     f"{'-':>{ac_width}}" if i == 0 else
                                     f"{'-':>{prob_width}}" if i == 1 else
                                     f"{'-':>{start_width}}" if i == 2 else
                                     f"{'-':>{end_width}}")
            else:
                if i == 0:  # Aircraft index
                    formatted_values.append(f"{float(x):>{ac_width}.0f}")
                elif i == 1:  # Probability
                    formatted_values.append(f"{float(x):>{prob_width}.2f}")
                elif i == 2:  # Start time
                    formatted_values.append(f"{float(x):>{start_width}.0f}")
                elif i == 3:  # End time
                    formatted_values.append(f"{float(x):>{end_width}.0f}")
                else:  # Flight numbers and times
                    formatted_values.append(f"{float(x):>{time_width}.0f}")
        formatted_rows.append(" ".join(formatted_values))
    
    print('\n'.join(formatted_rows))

# Print state
def print_state_nicely_proactive(state):
    # First print the information row in tabular form
    info_row = state[0]
    # print("\nSimulation Info:")
    # print("┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐")
    print("│ Current Time       │ Time Until End     │   ")
    # print("├────────────────────┼────────────────────┼────────────────────┼────────────────────┤")
    print(f"│ {int(info_row[0]) if not np.isnan(info_row[0]) else '-':^19}│ "
          f"{int(info_row[1]) if not np.isnan(info_row[1]) else '-':^19}│")
    # print("└────────────────────┴────────────────────┴────────────────────┴────────────────────┘")
    print("")  # Empty line for separation
    
    # Define column widths with extra space for non-flight headers
    ac_width = 4
    prob_width = 6
    start_width = 6
    end_width = 5
    flight_width = 5
    time_width = 5
    
    # Generate headers dynamically with proper spacing
    headers = [
        f"{'AC':>{ac_width}}", 
        f"{'Prob':>{prob_width}}", 
        f"{'Start':>{start_width}}", 
        f"{'End':>{end_width}}"
    ]
    
    # Add flight headers with proper spacing
    for i in range(1, MAX_FLIGHTS_PER_AIRCRAFT + 1):
        headers.extend([
            f"| {'F'+str(i):>{flight_width}}", 
            f"{'Dep'+str(i):>{time_width}}", 
            f"{'Arr'+str(i):>{time_width}}"
        ])
    
    # Print headers
    print(" ".join(headers))
    
    # Print state rows with matching alignment
    formatted_rows = []
    for row in state[1:]:
        formatted_values = []
        for i, x in enumerate(row):
            # Add vertical line before flight groups
            if i >= 4 and (i - 4) % 3 == 0:
                formatted_values.append("|")
                
            if np.isnan(x):
                formatted_values.append(f"{'-':>{time_width}}" if i >= 4 else 
                                     f"{'-':>{ac_width}}" if i == 0 else
                                     f"{'-':>{prob_width}}" if i == 1 else
                                     f"{'-':>{start_width}}" if i == 2 else
                                     f"{'-':>{end_width}}")
            else:
                if i == 0:  # Aircraft index
                    formatted_values.append(f"{float(x):>{ac_width}.0f}")
                elif i == 1:  # Probability
                    formatted_values.append(f"{float(x):>{prob_width}.2f}")
                elif i == 2:  # Start time
                    formatted_values.append(f"{float(x):>{start_width}.0f}")
                elif i == 3:  # End time
                    formatted_values.append(f"{float(x):>{end_width}.0f}")
                else:  # Flight numbers and times
                    formatted_values.append(f"{float(x):>{time_width}.0f}")
        formatted_rows.append(" ".join(formatted_values))
    
    print('\n'.join(formatted_rows))

def print_state_raw(state):
    print(state)

# Parsing all the data files
class FileParsers:
    
    @staticmethod
    def parse_config(data_lines):
        config_dict = {}
        config_dict['RecoveryPeriod'] = {
            'StartDate': data_lines[0].split()[0],
            'StartTime': data_lines[0].split()[1],
            'EndDate': data_lines[0].split()[2],
            'EndTime': data_lines[0].split()[3]
        }

        def parse_costs(line):
            parts = re.split(r'\s+', line)
            costs = [{'Cabin': parts[i], 'Type': parts[i+1], 'Cost': float(parts[i+2])} for i in range(0, len(parts), 3)]
            return costs

        config_dict['DelayCosts'] = parse_costs(data_lines[1])
        config_dict['CancellationCostsOutbound'] = parse_costs(data_lines[2])
        config_dict['CancellationCostsInbound'] = parse_costs(data_lines[3])

        def parse_downgrading_costs(line):
            parts = re.split(r'\s+', line)
            costs = [{'FromCabin': parts[i], 'ToCabin': parts[i+1], 'Type': parts[i+2], 'Cost': float(parts[i+3])} for i in range(0, len(parts), 4)]
            return costs

        config_dict['DowngradingCosts'] = parse_downgrading_costs(data_lines[4])
        config_dict['PenaltyCosts'] = [float(x) for x in re.split(r'\s+', data_lines[5])]
        config_dict['Weights'] = [float(x) for x in re.split(r'\s+', data_lines[6])]
        return config_dict

    @staticmethod
    def parse_airports(data_lines):
        airports_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            airport = parts[0]
            capacities = [{'Dep/h': int(parts[i]), 'Arr/h': int(parts[i+1]), 'StartTime': parts[i+2], 'EndTime': parts[i+3]} for i in range(1, len(parts), 4)]
            airports_dict[airport] = capacities
        return airports_dict

    @staticmethod
    def parse_dist(data_lines):
        dist_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            dist_dict[(parts[0], parts[1])] = {'Dist': int(parts[2]), 'Type': parts[3]}
        return dist_dict

    @staticmethod
    def parse_flights(data_lines):
        flights_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            flights_dict[int(parts[0])] = {'Orig': parts[1], 'Dest': parts[2], 'DepTime': parts[3], 'ArrTime': parts[4], 'PrevFlight': int(parts[5])}
        return flights_dict

    @staticmethod
    def parse_aircraft(data_lines):
        aircraft_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            aircraft_dict[parts[0]] = {'Model': parts[1], 'Family': parts[2], 'Config': parts[3], 'Dist': int(parts[4]), 'Cost/h': float(parts[5]),
                                       'TurnRound': int(parts[6]), 'Transit': int(parts[7]), 'Orig': parts[8], 'Maint': parts[9] if len(parts) > 9 else None}
        return aircraft_dict

    @staticmethod
    def parse_rotations(data_lines):
        rotations_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            rotations_dict[int(parts[0])] = {'DepDate': parts[1], 'Aircraft': parts[2]}
        return rotations_dict

    @staticmethod
    def parse_itineraries(data_lines):
        itineraries_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            itineraries_dict[int(parts[0])] = {'Type': parts[1], 'Price': float(parts[2]), 'Count': int(parts[3]), 'Flights': parts[4:]}
        return itineraries_dict

    @staticmethod
    def parse_position(data_lines):
        positions_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            if parts[0] not in positions_dict:
                positions_dict[parts[0]] = []
            positions_dict[parts[0]].append({'Model': parts[1], 'Config': parts[2], 'Count': int(parts[3])})
        return positions_dict

    @staticmethod
    def parse_alt_flights(data_lines):
        """Parses the alt_flights file into a dictionary."""
        if not data_lines:
            return {}

        alt_flights_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            alt_flights_dict[int(parts[0])] = {'DepDate': parts[1], 'Delay': int(parts[2])}
        return alt_flights_dict

    @staticmethod
    def parse_alt_aircraft(data_lines):
        """Parses the alt_aircraft file into a dictionary."""
        if data_lines is None:
            return {}

        alt_aircraft_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            alt_aircraft_dict[parts[0]] = {
                'StartDate': parts[1],
                'StartTime': parts[2],
                'EndDate': parts[3],
                'EndTime': parts[4],
                'Probability': float(parts[5])
            }
        return alt_aircraft_dict

    @staticmethod
    def parse_alt_airports(data_lines):
        """Parses the alt_airports file into a dictionary."""
        if data_lines is None:
            return {}

        alt_airports_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            airport = parts[0]
            if airport not in alt_airports_dict:
                alt_airports_dict[airport] = []
            alt_airports_dict[airport].append({
                'StartDate': parts[1],
                'StartTime': parts[2],
                'EndDate': parts[3],
                'EndTime': parts[4],
                'Dep/h': int(parts[5]),
                'Arr/h': int(parts[6])
            })
        return alt_airports_dict



# Data Processing Function
def load_data(data_folder):
    """Loads all the CSV files and returns a dictionary with parsed data."""
    
    # File paths
    aircraft_file = os.path.join(data_folder, 'aircraft.csv')
    airports_file = os.path.join(data_folder, 'airports.csv')
    alt_aircraft_file = os.path.join(data_folder, 'alt_aircraft.csv')
    alt_airports_file = os.path.join(data_folder, 'alt_airports.csv')
    alt_flights_file = os.path.join(data_folder, 'alt_flights.csv')
    config_file = os.path.join(data_folder, 'config.csv')
    dist_file = os.path.join(data_folder, 'dist.csv')
    flights_file = os.path.join(data_folder, 'flights.csv')
    itineraries_file = os.path.join(data_folder, 'itineraries.csv')
    positions_file = os.path.join(data_folder, 'position.csv')
    rotations_file = os.path.join(data_folder, 'rotations.csv')

    data_dict = {
        'config': FileParsers.parse_config(read_csv_with_comments(config_file)) if read_csv_with_comments(config_file) else {},
        'aircraft': FileParsers.parse_aircraft(read_csv_with_comments(aircraft_file)) if read_csv_with_comments(aircraft_file) else {},
        'airports': FileParsers.parse_airports(read_csv_with_comments(airports_file)) if read_csv_with_comments(airports_file) else {},
        'dist': FileParsers.parse_dist(read_csv_with_comments(dist_file)) if read_csv_with_comments(dist_file) else {},
        'flights': FileParsers.parse_flights(read_csv_with_comments(flights_file)) if read_csv_with_comments(flights_file) else {},
        'rotations': FileParsers.parse_rotations(read_csv_with_comments(rotations_file)) if read_csv_with_comments(rotations_file) else {},
        'itineraries': FileParsers.parse_itineraries(read_csv_with_comments(itineraries_file)) if read_csv_with_comments(itineraries_file) else {},
        'position': FileParsers.parse_position(read_csv_with_comments(positions_file)) if read_csv_with_comments(positions_file) else {},
        'alt_flights': FileParsers.parse_alt_flights(read_csv_with_comments(alt_flights_file)),
        'alt_aircraft': FileParsers.parse_alt_aircraft(read_csv_with_comments(alt_aircraft_file)),
        'alt_airports': FileParsers.parse_alt_airports(read_csv_with_comments(alt_airports_file))
    }
    
    return data_dict


# The name of the model is the current date in the format YYYYMMDD-X where X is the subsequent number of the model, based on the number of models already saved for the current day
def get_model_version(model_name):
    model_number = 1
    for file in os.listdir('../trained_models'):
        # print(f"checking file: {file}")
        if file.startswith(model_name):
            # print(f" - file starts with {model_name}")
            model_number += 1
    return str(model_number)





def format_time(time_dt, start_datetime):
    # Calculate the number of days difference from the start date
    delta_days = (time_dt.date() - start_datetime.date()).days
    time_str = time_dt.strftime('%H:%M')
    if delta_days >= 1:
        time_str += f'+{delta_days}'
    return time_str


def print_train_hyperparams():
    hyperparams = {
        "LEARNING_RATE": LEARNING_RATE,
        "GAMMA": GAMMA,
        "BUFFER_SIZE": BUFFER_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "TARGET_UPDATE_INTERVAL": TARGET_UPDATE_INTERVAL,
        "EPSILON_START": EPSILON_START,
        "EPSILON_MIN": EPSILON_MIN,
        "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE,
        "MAX_TIMESTEPS": MAX_TIMESTEPS,
    }
    
    for param, value in hyperparams.items():
        print(f"{param}: {value}")
    print("")


def save_best_and_worst_to_csv(scenario_folder, model_name, worst_actions, best_actions, worst_reward, best_reward):
    """Save the worst and best action sequences to a CSV file."""
    csv_file = os.path.join(scenario_folder, 'action_sequences.csv')
    
    # Check if the file exists; if not, create and write headers
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Write headers if file doesn't exist
            writer.writerow(['model_name', 'sequence_type', 'actions', 'reward'])
        
        # Append worst and best action sequences
        writer.writerow([model_name, "worst action sequence", worst_actions, worst_reward])
        writer.writerow([model_name, "best action sequence", best_actions, best_reward])
