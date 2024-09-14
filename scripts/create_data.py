import os
import csv
import random
import shutil
import re
from datetime import datetime, timedelta

# Utility functions

def read_csv_with_comments(file_path):
    """
    Reads a CSV file, skipping comment lines that start with '%' and stopping at lines that start with '#'.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_lines = []
    for line in lines:
        if line.startswith('#'):
            break
        if not line.startswith('%'):
            data_lines.append(line.strip())

    return data_lines if data_lines else None


def clear_file(file_name):
    """Clears the content of a file."""
    with open(file_name, 'w') as file:
        file.write('')


def parse_time_with_day_offset(time_str, reference_date):
    """Parses time and adds a day offset if '+1' is present."""
    if '+1' in time_str:
        time_str = time_str.replace('+1', '').strip()
        time_obj = datetime.strptime(time_str, '%H:%M')
        return datetime.combine(reference_date, time_obj.time()) + timedelta(days=1)
    else:
        return datetime.strptime(time_str, '%H:%M').replace(year=reference_date.year, month=reference_date.month, day=reference_date.day)


# Function to generate the config file
def parse_config(data_lines):
    """Parses the configuration file data lines."""
    config_dict = {}
    config_dict['RecoveryPeriod'] = {
        'StartDate': data_lines[0].split()[0],
        'StartTime': data_lines[0].split()[1],
        'EndDate': data_lines[0].split()[2],
        'EndTime': data_lines[0].split()[3]
    }

    def parse_costs(line):
        parts = re.split(r'\s+', line)
        costs = []
        for i in range(0, len(parts), 3):
            costs.append({'Cabin': parts[i], 'Type': parts[i+1], 'Cost': float(parts[i+2])})
        return costs

    config_dict['DelayCosts'] = parse_costs(data_lines[1])
    config_dict['CancellationCostsOutbound'] = parse_costs(data_lines[2])
    config_dict['CancellationCostsInbound'] = parse_costs(data_lines[3])

    def parse_downgrading_costs(line):
        parts = re.split(r'\s+', line)
        costs = []
        for i in range(0, len(parts), 4):
            costs.append({'FromCabin': parts[i], 'ToCabin': parts[i+1], 'Type': parts[i+2], 'Cost': float(parts[i+3])})
        return costs

    config_dict['DowngradingCosts'] = parse_downgrading_costs(data_lines[4])
    config_dict['PenaltyCosts'] = [float(x) for x in re.split(r'\s+', data_lines[5])]
    config_dict['Weights'] = [float(x) for x in re.split(r'\s+', data_lines[6])]
    return config_dict


def generate_config_file(file_name, config_dict, recovery_start_date, recovery_start_time, recovery_end_date, recovery_end_time):
    """Generates the config file."""
    clear_file(file_name)
    with open(file_name, 'w') as file:
        file.write('%RecoveryPeriod\n')
        file.write(f"{recovery_start_date} {recovery_start_time} {recovery_end_date} {recovery_end_time}\n")

        config_dict['RecoveryPeriod'] = {
            'StartDate': recovery_start_date,
            'StartTime': recovery_start_time,
            'EndDate': recovery_end_date,
            'EndTime': recovery_end_time
        }

        file.write('%DelayCosts\n')
        for cost in config_dict['DelayCosts']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")

        file.write('\n%CancellationCostsOutbound\n')
        for cost in config_dict['CancellationCostsOutbound']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")

        file.write('\n%CancellationCostsInbound\n')
        for cost in config_dict['CancellationCostsInbound']:
            file.write(f"{cost['Cabin']} {cost['Type']} {cost['Cost']} ")

        file.write('\n%DowngradingCosts\n')
        for cost in config_dict['DowngradingCosts']:
            file.write(f"{cost['FromCabin']} {cost['ToCabin']} {cost['Type']} {cost['Cost']} ")

        file.write('\n%PenaltyCosts\n')
        for cost in config_dict['PenaltyCosts']:
            file.write(f"{cost} ")
        file.write('\n')

        file.write('%Weights\n')
        for weight in config_dict['Weights']:
            file.write(f"{weight} ")
        file.write('\n')

        file.write('#')


# Function to generate aircraft file
def generate_aircraft_file(file_name, aircraft_types, total_aircraft_range):
    """Generates the aircraft.csv file."""
    clear_file(file_name)
    total_aircraft = random.randint(*total_aircraft_range)
    
    aircraft_data = []
    aircraft_counter = {aircraft['Model']: 0 for aircraft in aircraft_types}
    aircraft_ids = []

    for _ in range(total_aircraft):
        aircraft_type = random.choice(aircraft_types)
        model = aircraft_type['Model']
        aircraft_counter[model] += 1
        aircraft_id = f"{model}#{aircraft_counter[model]}"
        aircraft_ids.append(aircraft_id)

        aircraft_data.append(f"{aircraft_id} {model} {aircraft_type['Family']} {aircraft_type['Config']} {aircraft_type['Dist']} {aircraft_type['Cost/h']} "
                             f"{aircraft_type['TurnRound']} {aircraft_type['Transit']} {random.choice(aircraft_type['Orig'])} {random.choice(aircraft_type['Maint'])}")
    
    aircraft_data.sort()

    with open(file_name, 'w') as file:
        file.write('%Aircraft Model Family Config Dist Cost/h TurnRound Transit Orig Maint\n')
        for aircraft in aircraft_data:
            file.write(f"{aircraft}\n")
        file.write('#')

    return aircraft_ids


# Function to generate alt_aircraft.csv
def generate_alt_aircraft_file(file_name, aircraft_ids, amount_aircraft_disrupted, config_dict, min_delta_start_unavailability, max_delta_start_unavailability, min_period_unavailability, max_period_unavailability):
    """Generates the alt_aircraft.csv file."""
    clear_file(file_name)

    disrupted_aircraft_ids = random.sample(aircraft_ids, amount_aircraft_disrupted)
    disrupted_aircraft_data = []

    for aircraft_id in disrupted_aircraft_ids:
        start_date = config_dict['RecoveryPeriod']['StartDate']
        start_time_recovery = config_dict['RecoveryPeriod']['StartTime']
        start_unavail = (datetime.strptime(start_time_recovery, '%H:%M') +
                         timedelta(minutes=random.randint(min_delta_start_unavailability, max_delta_start_unavailability))).strftime('%H:%M')
        
        end_date = config_dict['RecoveryPeriod']['EndDate']
        start_unavail_obj = datetime.strptime(start_unavail, '%H:%M')
        end_unavail = (start_unavail_obj + timedelta(minutes=random.randint(min_period_unavailability, max_period_unavailability))).strftime('%H:%M')

        disrupted_aircraft_data.append(f"{aircraft_id} {start_date} {start_unavail} {end_date} {end_unavail}")

    with open(file_name, 'w') as file:
        file.write('%Aircraft StartDate StartTime EndDate EndTime\n')
        for aircraft in disrupted_aircraft_data:
            file.write(f"{aircraft}\n")
        file.write('#')


# Function to generate flights.csv
def generate_flights_file(file_name, aircraft_ids, average_flights_per_aircraft, std_dev_flights_per_aircraft, airports, config_dict, start_datetime, end_datetime):
    """Generates the flights.csv file."""
    clear_file(file_name)

    flights_dict = {}
    flight_rotation_data = {}

    total_flights = len(aircraft_ids) * average_flights_per_aircraft
    amount_flights_per_aircraft = {}
    flights_left_to_generate = total_flights

    for aircraft_id in aircraft_ids:
        amount_flights_per_aircraft[aircraft_id] = min(
            random.randint(average_flights_per_aircraft - std_dev_flights_per_aircraft, 
                           average_flights_per_aircraft + std_dev_flights_per_aircraft),
            flights_left_to_generate
        )
        flights_left_to_generate -= amount_flights_per_aircraft[aircraft_id]

    flight_id = 0
    for aircraft_id in aircraft_ids:
        flight_id_ac_specific = 0

        for _ in range(amount_flights_per_aircraft[aircraft_id]):
            flight_id += 1
            flight_id_ac_specific += 1
            orig, dest = random.choice(airports), random.choice(airports)
            while orig == dest:
                dest = random.choice(airports)

            if flight_id_ac_specific == 1:
                dep_time = f"{random.randint(6, 12)}:{random.choice(['00', '15', '30', '45'])}"
                dep_time_obj = parse_time_with_day_offset(dep_time, start_datetime)
                arr_time_obj = dep_time_obj + timedelta(hours=random.randint(1, 4), minutes=random.randint(0, 59))
            else:
                arr_time_prev = flights_dict[flight_id - 1]['ArrTime']
                dep_time_obj = parse_time_with_day_offset(arr_time_prev, start_datetime) + timedelta(hours=random.randint(0, 2), minutes=random.randint(0, 59))
                arr_time_obj = dep_time_obj + timedelta(hours=random.randint(1, 4), minutes=random.randint(0, 59))

            if arr_time_obj.day > start_datetime.day:
                arr_time = f"{arr_time_obj.strftime('%H:%M')}+1"
            else:
                arr_time = arr_time_obj.strftime('%H:%M')

            dep_time = dep_time_obj.strftime('%H:%M')

            flights_dict[flight_id] = {'Orig': orig, 'Dest': dest, 'DepTime': dep_time, 'ArrTime': arr_time, 'PrevFlight': 0, 'Aircraft': aircraft_id}
            flight_rotation_data[flight_id] = {'Aircraft': aircraft_id}

            if arr_time_obj > end_datetime:
                break

    with open(file_name, 'w') as file:
        file.write('%Flight Orig Dest DepTime ArrTime PrevFlight\n')
        for flight_id, flight_data in flights_dict.items():
            file.write(f"{flight_id} {flight_data['Orig']} {flight_data['Dest']} {flight_data['DepTime']} {flight_data['ArrTime']} {flight_data['PrevFlight']}\n")
        file.write('#')

    return flights_dict, flight_rotation_data


# Function to generate rotations.csv
def generate_rotations_file(file_name, flight_rotation_data, start_datetime):
    """Generates the rotations.csv file."""
    clear_file(file_name)
    rotations_data = []

    for flight_id, flight_data in flight_rotation_data.items():
        dep_date = start_datetime.strftime('%d/%m/%y')
        rotations_data.append(f"{flight_id} {dep_date} {flight_data['Aircraft']}")

    with open(file_name, 'w') as file:
        file.write('%Flight DepDate Aircraft\n')
        for rotation in rotations_data:
            file.write(f"{rotation}\n")
        file.write('#')


# Main function to run the whole process (but you can call this whatever you like)

def create_data_scenario(
    scenario_name, template_folder, data_root_folder, aircraft_types, total_aircraft_range,
    amount_aircraft_disrupted, min_delta_start_unavailability, max_delta_start_unavailability,
    min_period_unavailability, max_period_unavailability, average_flights_per_aircraft,
    std_dev_flights_per_aircraft, airports, config_dict, recovery_start_date,
    recovery_start_time, recovery_end_date, recovery_end_time
):
    data_folder = os.path.join(data_root_folder, scenario_name)
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    shutil.copytree(template_folder, data_folder)

    # Generate config file
    config_file = os.path.join(data_folder, 'config.csv')
    generate_config_file(config_file, config_dict, recovery_start_date, recovery_start_time, recovery_end_date, recovery_end_time)

    # Generate aircraft data
    aircraft_file = os.path.join(data_folder, 'aircraft.csv')
    aircraft_ids = generate_aircraft_file(aircraft_file, aircraft_types, total_aircraft_range)

    # Generate alt aircraft (disrupted aircraft)
    alt_aircraft_file = os.path.join(data_folder, 'alt_aircraft.csv')
    generate_alt_aircraft_file(alt_aircraft_file, aircraft_ids, amount_aircraft_disrupted, config_dict, min_delta_start_unavailability, max_delta_start_unavailability, min_period_unavailability, max_period_unavailability)

    # Generate flights data
    flights_file = os.path.join(data_folder, 'flights.csv')
    start_datetime = datetime.strptime(f"{recovery_start_date} {recovery_start_time}", '%d/%m/%y %H:%M')
    end_datetime = datetime.strptime(f"{recovery_end_date} {recovery_end_time}", '%d/%m/%y %H:%M')
    flights_dict, flight_rotation_data = generate_flights_file(flights_file, aircraft_ids, average_flights_per_aircraft, std_dev_flights_per_aircraft, airports, config_dict, start_datetime, end_datetime)

    # Generate rotations data
    rotations_file = os.path.join(data_folder, 'rotations.csv')
    generate_rotations_file(rotations_file, flight_rotation_data, start_datetime)

    print(f"Data creation for scenario {scenario_name} completed with {len(aircraft_ids)} aircraft and {len(flights_dict)} flights.")
