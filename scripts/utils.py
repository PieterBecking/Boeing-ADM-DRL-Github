import os
import numpy as np
from datetime import datetime, timedelta
import re
from src.config import *
import csv
import torch as th
import math

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

    # # Fix the alt_aircraft file: if the arrival time is before the departure time, add 24 hours to the arrival time
    # if data_dict['alt_aircraft']:
    #     print(f"Fixing alt_aircraft file")
    #     print(data_dict['alt_aircraft'])
    #     """
    #     data_dict['alt_aircraft'] is of this form:
    #     {'A320#1': {'StartDate': '14/09/24', 'StartTime': '08:02', 'EndDate': '14/09/24', 'EndTime': '23:29', 'Probability': 0.17}, 'A320#2': {'StartDate': '14/09/24', 'StartTime': '08:15', 'EndDate': '14/09/24', 'EndTime': '13:13', 'Probability': 0.31}, 'A320#3': {'StartDate': '14/09/24', 'StartTime': '08:12', 'EndDate': '14/09/24', 'EndTime': '15:54', 'Probability': 0.0}}
    #     """
    #     for aircraft in data_dict['alt_aircraft']:
    #         for flight in data_dict['alt_aircraft'][aircraft]:
    #             deptime = data_dict['alt_aircraft'][aircraft]['StartTime']
    #             print(f"deptime: {deptime}")
    #             arrtime = data_dict['alt_aircraft'][aircraft]['EndTime']
    #             print(f"arrtime: {arrtime}")
    #             if arrtime < deptime:
    #                 # add one day to the end date
    #                 data_dict['alt_aircraft'][aircraft]['EndDate'] = (datetime.strptime(data_dict['alt_aircraft'][aircraft]['EndDate'], '%d/%m/%y') + timedelta(days=1)).strftime('%d/%m/%y')
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


# Check the trained_models folder, split each name by "-" and see the last part, which is the model version. add one to it and return it as a string
def get_model_version(model_name, myopic_proactive, drl_type):
    print(f"Getting model version for {model_name}")
    
    model_number = 1
    for file in os.listdir(f'../trained_models/{drl_type}'):

        # drop the -x in the end
        file_model_name = file.split('-')[0]

        if file_model_name == model_name:
            model_number += 1
    return str(model_number)

# Check the trained_models folder, split each name by "-" and see the last part, which is the model version. add one to it and return it as a string
def get_model_version_from_root(model_name, model_type):
    print(f"Getting model version for {model_name}")
    
    model_number = 1
    for file in os.listdir(f'trained_models/dqn/{model_type}'):

        # drop the -x in the end
        file_model_name = file.split('-')[0]

        if file_model_name == model_name:
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


import subprocess
import platform


def get_macbook_info():
    model_info = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
    output = model_info.stdout
    
    # Extract relevant information
    model_name = re.search(r"Model Name: (.*)", output)
    chip = re.search(r"Chip: (.*)", output)
    total_cores = re.search(r"Total Number of Cores: (.*)", output)
    memory = re.search(r"Memory: (.*)", output)
    
    info = {
        "Model Name": model_name.group(1) if model_name else "Unknown",
        "Chip": chip.group(1) if chip else "Unknown",
        "Total Cores": total_cores.group(1) if total_cores else "Unknown",
        "Memory": memory.group(1) if memory else "Unknown"
    }
    
    return info



def get_gpu_info():
    if platform.system() == "Darwin":  # macOS
        gpu_info = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
        output = gpu_info.stdout
        chipset_model = re.search(r"Chipset Model: (.*)", output)
        type = re.search(r"Type: (.*)", output)
        return {
            "Chipset Model": chipset_model.group(1) if chipset_model else "Unknown",
            "Type": type.group(1) if type else "Unknown"
        }
    elif platform.system() == "Windows" or platform.system() == "Linux":
        gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        output = gpu_info.stdout.strip()
        if output:
            name, memory = output.split(', ')
            return {
                "GPU Name": name,
                "Memory Total": memory
            }
        else:
            return "No NVIDIA GPU found or `nvidia-smi` not available."
    else:
        return "Unsupported OS"



def get_l40s_info():
    # Querying memory usage, GPU utilization, temperature, etc.
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    output = gpu_info.stdout.strip()
    if output:
        fields = output.split(', ')
        return {
            "GPU Name": fields[0],
            "Driver Version": fields[1],
            "Total Memory": fields[2],
            "Used Memory": fields[3],
            "Free Memory": fields[4],
            "GPU Utilization": fields[5],
            "Temperature": fields[6]
        }
    else:
        return "No detailed information available or `nvidia-smi` not available."



# device_info = get_l40s_info()
# print("Detailed GPU Info:", device_info)


def initialize_device():
    """Initialize and return the computation device."""
    if th.cuda.is_available():
        device = th.device('cuda')
    elif th.backends.mps.is_available():
        device = th.device('mps')
    else:
        device = th.device('cpu')

    print(f"Using device: {device}")
    return device


def get_device_info(device):
    """Get detailed information about the device."""
    device_str = str(device).lower()
    print("Device:", device_str)

    if device_str == 'mps':
        print("Using MacBook M1")
        device_info = {"device_type": "MacBook M1"}
    elif device_str == 'cuda':
        print("Using GPU")
        device_info = {"device_type": "GPU"}
    else:
        print("Using CPU")
        device_info = {"device_type": "CPU"}

    return device_info


def check_device_capabilities():
    """Check and print device capabilities."""
    print("CUDA available:", th.cuda.is_available())
    print("Number of GPUs available:", th.cuda.device_count())
    if th.cuda.is_available():
        print("Current GPU name:", th.cuda.get_device_name())
    print("cuDNN enabled:", th.backends.cudnn.enabled)


def verify_training_folders(path):
    """Verify if the training folders exist and return folder names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Training folder not found at {path}')

    training_folders = [
        folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))
    ]
    return training_folders


def calculate_training_days(n_episodes, folders):
    """Calculate total training days."""
    return n_episodes * len(folders)


def format_days(days):
    """Format the number of days for better readability."""
    if days >= 1000000:
        return f"{math.floor(days / 1000000)}M"
    elif days >= 1000:
        return f"{math.floor(days / 1000)}k"
    return str(days)


def create_results_directory(base_dir='../results', append_to_name=''):
    """Create a results directory with the current datetime."""
    base_dir = os.path.join(base_dir, append_to_name)
    now = datetime.now()
    folder_name = now.strftime('%Y%m%d-%H-%M')
    results_dir = os.path.join(base_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    return results_dir

def calculate_epsilon_decay_rate(total_timesteps, epsilon_start, epsilon_min, percentage_min=95, EPSILON_TYPE="exponential"):
    """
    Calculates the decay rate for epsilon such that it reaches epsilon_min after the specified percentage of total timesteps.

    Args:
        total_timesteps (int): Total number of timesteps in training.
        epsilon_start (float): Initial epsilon value for exploration.
        epsilon_min (float): Minimum epsilon value for exploration.
        percentage_min (float): Percentage of timesteps at which epsilon should reach epsilon_min (default is 95%).

    Returns:
        float: Calculated epsilon decay rate.
    """
    # Calculate the timesteps at which epsilon should reach epsilon_min
    target_timesteps = total_timesteps * (percentage_min / 100)

    if EPSILON_TYPE == "exponential":
        # Solve for the decay rate using the exponential decay formula
        decay_rate = -np.log(epsilon_min / epsilon_start) / target_timesteps
    elif EPSILON_TYPE == "linear":
        decay_rate = (epsilon_start - epsilon_min) / target_timesteps

    print(f"Calculated EPSILON_DECAY_RATE: {decay_rate}")
    return decay_rate
