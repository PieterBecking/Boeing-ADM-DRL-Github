import pandas as pd
import os
import re
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

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
            print(f"File contains no valid data: {file_path}")
            return []
        
        return data_lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


# File Parsers

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

def parse_airports(data_lines):
    airports_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        airport = parts[0]
        capacities = [{'Dep/h': int(parts[i]), 'Arr/h': int(parts[i+1]), 'StartTime': parts[i+2], 'EndTime': parts[i+3]} for i in range(1, len(parts), 4)]
        airports_dict[airport] = capacities
    return airports_dict

def parse_dist(data_lines):
    dist_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        dist_dict[(parts[0], parts[1])] = {'Dist': int(parts[2]), 'Type': parts[3]}
    return dist_dict

def parse_flights(data_lines):
    flights_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        flights_dict[int(parts[0])] = {'Orig': parts[1], 'Dest': parts[2], 'DepTime': parts[3], 'ArrTime': parts[4], 'PrevFlight': int(parts[5])}
    return flights_dict

def parse_aircraft(data_lines):
    aircraft_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        aircraft_dict[parts[0]] = {'Model': parts[1], 'Family': parts[2], 'Config': parts[3], 'Dist': int(parts[4]), 'Cost/h': float(parts[5]),
                                   'TurnRound': int(parts[6]), 'Transit': int(parts[7]), 'Orig': parts[8], 'Maint': parts[9] if len(parts) > 9 else None}
    return aircraft_dict

def parse_rotations(data_lines):
    rotations_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        rotations_dict[int(parts[0])] = {'DepDate': parts[1], 'Aircraft': parts[2]}
    return rotations_dict

def parse_itineraries(data_lines):
    itineraries_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        itineraries_dict[int(parts[0])] = {'Type': parts[1], 'Price': float(parts[2]), 'Count': int(parts[3]), 'Flights': parts[4:]}
    return itineraries_dict

def parse_positions(data_lines):
    positions_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        if parts[0] not in positions_dict:
            positions_dict[parts[0]] = []
        positions_dict[parts[0]].append({'Model': parts[1], 'Config': parts[2], 'Count': int(parts[3])})
    return positions_dict

def parse_alt_flights(data_lines):
    """Parses the alt_flights file into a dictionary."""
    # Return an empty dictionary if data_lines is empty
    if not data_lines:
        return {}

    alt_flights_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        alt_flights_dict[int(parts[0])] = {'DepDate': parts[1], 'Delay': int(parts[2])}
    return alt_flights_dict



def parse_alt_aircraft(data_lines):
    """Parses the alt_aircraft file into a dictionary."""
    # Return an empty dictionary if data_lines is None (file is empty)
    if data_lines is None:
        return {}

    alt_aircraft_dict = {}
    for line in data_lines:
        parts = re.split(r'\s+', line)
        alt_aircraft_dict[parts[0]] = {
            'StartDate': parts[1],
            'StartTime': parts[2],
            'EndDate': parts[3],
            'EndTime': parts[4]
        }
    return alt_aircraft_dict


def parse_alt_airports(data_lines):
    """Parses the alt_airports file into a dictionary."""
    # Return an empty dictionary if data_lines is None (file is empty)
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

    print(f"Loading data from folder: {data_folder}")
    
    # Read and parse data
    alt_flights_data = read_csv_with_comments(alt_flights_file)
    if alt_flights_data is None:
        print(f"read_csv_with_comments returned None for alt_flights_file: {alt_flights_file}")
    else:
        print(f"read_csv_with_comments returned {len(alt_flights_data)} lines for alt_flights_file")

    data_dict = {
        'config': parse_config(read_csv_with_comments(config_file)) if read_csv_with_comments(config_file) else {},
        'aircraft': parse_aircraft(read_csv_with_comments(aircraft_file)) if read_csv_with_comments(aircraft_file) else {},
        'airports': parse_airports(read_csv_with_comments(airports_file)) if read_csv_with_comments(airports_file) else {},
        'dist': parse_dist(read_csv_with_comments(dist_file)) if read_csv_with_comments(dist_file) else {},
        'flights': parse_flights(read_csv_with_comments(flights_file)) if read_csv_with_comments(flights_file) else {},
        'rotations': parse_rotations(read_csv_with_comments(rotations_file)) if read_csv_with_comments(rotations_file) else {},
        'itineraries': parse_itineraries(read_csv_with_comments(itineraries_file)) if read_csv_with_comments(itineraries_file) else {},
        'positions': parse_positions(read_csv_with_comments(positions_file)) if read_csv_with_comments(positions_file) else {},
        'alt_flights': parse_alt_flights(alt_flights_data),
        'alt_aircraft': parse_alt_aircraft(read_csv_with_comments(alt_aircraft_file)),
        'alt_airports': parse_alt_airports(read_csv_with_comments(alt_airports_file))
    }
    
    return data_dict







# Visualization Functions

def visualize_aircraft_rotations(data_dict):
    """Visualizes aircraft rotations and delays."""
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_flights_dict = data_dict['alt_flights']
    alt_aircraft_dict = data_dict['alt_aircraft']
    config_dict = data_dict['config']

    # Time parsing
    start_datetime = datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + config_dict['RecoveryPeriod']['StartTime'], '%d/%m/%y %H:%M')
    end_datetime = datetime.strptime(config_dict['RecoveryPeriod']['EndDate'] + ' ' + config_dict['RecoveryPeriod']['EndTime'], '%d/%m/%y %H:%M')

    # Determine the earliest and latest times from the flight data
    earliest_datetime = start_datetime
    latest_datetime = end_datetime

    for flight_info in flights_dict.values():
        dep_datetime = parse_time_with_day_offset(flight_info['DepTime'], start_datetime)
        arr_datetime = parse_time_with_day_offset(flight_info['ArrTime'], dep_datetime)

        if dep_datetime < earliest_datetime:
            earliest_datetime = dep_datetime
        if arr_datetime > latest_datetime:
            latest_datetime = arr_datetime

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot logic (from the notebook)
    aircraft_ids = sorted(list(set([rotation_info['Aircraft'] for rotation_info in rotations_dict.values()])), reverse=False)
    aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(aircraft_ids)}

    # Plot each flight's schedule
    for rotation_id, rotation_info in rotations_dict.items():
        flight_id = rotation_id
        aircraft_id = rotation_info['Aircraft']

        if flight_id in flights_dict:
            flight_info = flights_dict[flight_id]
            dep_datetime = parse_time_with_day_offset(flight_info['DepTime'], start_datetime)
            arr_datetime = parse_time_with_day_offset(flight_info['ArrTime'], dep_datetime)

            ax.plot([dep_datetime, arr_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='blue', marker='o')

    # Plot disruptions (from the notebook)
    if alt_flights_dict:
        for flight_id, disruption_info in alt_flights_dict.items():
            dep_datetime = parse_time_with_day_offset(flights_dict[flight_id]['DepTime'], start_datetime)
            delay_duration = timedelta(minutes=disruption_info['Delay'])
            delayed_datetime = dep_datetime + delay_duration
            aircraft_id = rotations_dict[flight_id]['Aircraft']

            ax.plot([dep_datetime, delayed_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='red', linestyle='-')

    # Customize the plot
    ax.set_xlim(earliest_datetime - timedelta(hours=1), latest_datetime + timedelta(hours=1))
    plt.xlabel('Time')
    plt.ylabel('Aircraft')
    plt.title('Aircraft Rotations, AC Unavailability and Flight Delays')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()


def visualize_flight_airport_unavailability(data_dict):
    """Visualizes flight and airport unavailability."""
    flights_dict = data_dict['flights']
    alt_airports_dict = data_dict['alt_airports']
    config_dict = data_dict['config']

    start_datetime = datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + config_dict['RecoveryPeriod']['StartTime'], '%d/%m/%y %H:%M')
    end_datetime = datetime.strptime(config_dict['RecoveryPeriod']['EndDate'] + ' ' + config_dict['RecoveryPeriod']['EndTime'], '%d/%m/%y %H:%M')

    # Plot logic (from the notebook)
    airports = list(set([flight_info['Orig'] for flight_info in flights_dict.values()] + [flight_info['Dest'] for flight_info in flights_dict.values()]))
    airport_indices = {airport: index + 1 for index, airport in enumerate(airports)}

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each flight's schedule based on airports in blue
    for flight_id, flight_info in flights_dict.items():
        dep_datetime = parse_time_with_day_offset(flight_info['DepTime'], start_datetime)
        arr_datetime = parse_time_with_day_offset(flight_info['ArrTime'], dep_datetime)
        ax.plot([dep_datetime, arr_datetime], [airport_indices[flight_info['Orig']], airport_indices[flight_info['Dest']]], color='blue', marker='o')

    # Plot airport disruptions (from the notebook)
    if alt_airports_dict:
        for airport, disruptions in alt_airports_dict.items():
            for disruption_info in disruptions:
                unavail_start_datetime = datetime.strptime(disruption_info['StartDate'] + ' ' + disruption_info['StartTime'], '%d/%m/%y %H:%M')
                unavail_end_datetime = datetime.strptime(disruption_info['EndDate'] + ' ' + disruption_info['EndTime'], '%d/%m/%y %H:%M')

                ax.plot([unavail_start_datetime, unavail_end_datetime], [airport_indices[airport], airport_indices[airport]], color='red', linestyle='solid')

    # Customize the plot
    ax.set_xlim(start_datetime - timedelta(hours=1), end_datetime + timedelta(hours=1))
    plt.xlabel('Time')
    plt.ylabel('Airports')
    plt.title('Flights and Airport Unavailability')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()


# Function to parse time strings with day offset handling
def parse_time_with_day_offset(time_str, reference_date):
    """Parses time and handles '+1' for next day times."""
    if '+1' in time_str:
        time_str = time_str.replace('+1', '').strip()
        time_obj = datetime.strptime(time_str, '%H:%M')
        return datetime.combine(reference_date, time_obj.time()) + timedelta(days=1)
    else:
        return datetime.strptime(time_str, '%H:%M').replace(year=reference_date.year, month=reference_date.month, day=reference_date.day)


# Callable entry point for visualization process

def run_visualization(scenario_name, data_root_folder):
    data_folder = os.path.join(data_root_folder, scenario_name)
    
    # Load data from CSV files
    data_dict = load_data(data_folder)

    # Visualize aircraft rotations
    visualize_aircraft_rotations(data_dict)

    # Visualize flight and airport unavailability
    visualize_flight_airport_unavailability(data_dict)

    print(f"Visualization for scenario {scenario_name} completed.")

