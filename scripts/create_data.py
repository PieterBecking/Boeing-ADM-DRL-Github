import os
import csv
import random
import shutil
import re
from datetime import datetime, timedelta
from scripts.utils import *
from src.config import *


# Function to generate the config file
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

    total_flights = max(1, len(aircraft_ids) * average_flights_per_aircraft)  # Ensure at least 1 flight
    amount_flights_per_aircraft = {}
    flights_left_to_generate = total_flights

    for aircraft_id in aircraft_ids:
        amount_flights_per_aircraft[aircraft_id] = max(1, min(
            random.randint(average_flights_per_aircraft - std_dev_flights_per_aircraft, 
                           average_flights_per_aircraft + std_dev_flights_per_aircraft),
            flights_left_to_generate
        ))
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

            # Check if the departure time exceeds the end_datetime with a reasonable buffer
            if dep_time_obj > end_datetime + timedelta(hours=DEPARTURE_AFTER_END_RECOVERY):
                break  # Stop generating flights if departure time exceeds the limit
            
            # Add day offset to arrival and departure times when necessary
            if arr_time_obj.day > start_datetime.day:
                arr_time = f"{arr_time_obj.strftime('%H:%M')}+1"
            else:
                arr_time = arr_time_obj.strftime('%H:%M')

            # Check if departure time crosses into the next day (after midnight)
            if dep_time_obj.day > start_datetime.day:
                dep_time = f"{dep_time_obj.strftime('%H:%M')}+1"
            else:
                dep_time = dep_time_obj.strftime('%H:%M')

            flights_dict[flight_id] = {'Orig': orig, 'Dest': dest, 'DepTime': dep_time, 'ArrTime': arr_time, 'PrevFlight': 0, 'Aircraft': aircraft_id}
            flight_rotation_data[flight_id] = {'Aircraft': aircraft_id}

            # Also, break if arrival time exceeds the end of the recovery period
            if arr_time_obj > end_datetime:
                break

    # Ensure at least one flight is generated
    if not flights_dict:
        flight_id = 1
        aircraft_id = aircraft_ids[0]
        orig, dest = random.choice(airports), random.choice(airports)
        while orig == dest:
            dest = random.choice(airports)
        
        dep_time = f"{random.randint(6, 12)}:{random.choice(['00', '15', '30', '45'])}"
        dep_time_obj = parse_time_with_day_offset(dep_time, start_datetime)
        arr_time_obj = dep_time_obj + timedelta(hours=random.randint(1, 4), minutes=random.randint(0, 59))

        # Add day offset to arrival and departure times when necessary
        if arr_time_obj.day > start_datetime.day:
            arr_time = f"{arr_time_obj.strftime('%H:%M')}+1"
        else:
            arr_time = arr_time_obj.strftime('%H:%M')

        # Check if departure time crosses into the next day (after midnight)
        if dep_time_obj.day > start_datetime.day:
            dep_time = f"{dep_time_obj.strftime('%H:%M')}+1"
        else:
            dep_time = dep_time_obj.strftime('%H:%M')

        flights_dict[flight_id] = {'Orig': orig, 'Dest': dest, 'DepTime': dep_time, 'ArrTime': arr_time, 'PrevFlight': 0, 'Aircraft': aircraft_id}
        flight_rotation_data[flight_id] = {'Aircraft': aircraft_id}

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
    recovery_start_time, recovery_end_date, recovery_end_time, clear_one_random_aircraft
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

    # Clear one random aircraft such that its flights are removed
    if clear_one_random_aircraft:
        aircraft_id = random.choice(aircraft_ids)
        
        # get its flights from the rotations file
        rotations_data = []
        with open(rotations_file, 'r') as file:
            for line in file:
                if line.startswith('%') or line.startswith('#'):
                    continue
                flight_id, dep_date, aircraft = line.strip().split(' ')
                if aircraft == aircraft_id:
                    rotations_data.append(flight_id)

        # remove the flights from the flights file
        flights_data = []
        with open(flights_file, 'r') as file:
            for line in file:
                if line.startswith('%') or line.startswith('#'):
                    continue
                flight_id, orig, dest, dep_time, arr_time, prev_flight = line.strip().split(' ')
                if flight_id not in rotations_data:
                    flights_data.append(line)
        with open(flights_file, 'w') as file:
            file.write('%Flight Orig Dest DepTime ArrTime PrevFlight\n')
            for flight in flights_data:
                file.write(f"{flight}")
            file.write('#')



    print(f"Data creation for scenario {scenario_name} completed with {len(aircraft_ids)} aircraft and {len(flights_dict)} flights.")


