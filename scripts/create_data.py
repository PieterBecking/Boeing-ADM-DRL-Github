import os
import csv
import random
import shutil
import re
from datetime import datetime, timedelta
from scripts.utils import *
from scripts.logger import *
from src.config import *
import numpy as np


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

# generate_alt_aircraft_file(alt_aircraft_file, aircraft_ids, amount_aircraft_disrupted, config_dict, min_delta_start_unavailability, max_delta_start_unavailability, min_period_unavailability, max_period_unavailability, probability_range, probability_distribution)

def generate_alt_aircraft_file(file_name, aircraft_ids, amount_aircraft_disrupted, amount_aircraft_uncertain, config_dict, min_delta_start_unavailability, max_delta_start_unavailability, min_period_unavailability, max_period_unavailability, probability_range, probability_distribution):
    """Generates the alt_aircraft.csv file with additional probability information."""
    clear_file(file_name)

    # Ensure we don't request more aircraft than available
    total_affected = amount_aircraft_disrupted + amount_aircraft_uncertain
    if total_affected > len(aircraft_ids):
        raise ValueError(f"Total affected aircraft ({total_affected}) exceeds available aircraft ({len(aircraft_ids)})")

    # Randomly select aircraft for both categories
    all_affected_aircraft = random.sample(aircraft_ids, total_affected)
    disrupted_aircraft_ids = all_affected_aircraft[:amount_aircraft_disrupted]
    uncertain_aircraft_ids = all_affected_aircraft[amount_aircraft_disrupted:]

    all_aircraft_data = []
    disruption_log = {
        "total_aircraft": len(aircraft_ids),
        "disrupted_count": amount_aircraft_disrupted,
        "uncertain_count": amount_aircraft_uncertain,
        "probability_range": probability_range,
        "disruptions": []
    }

    # Only process aircraft that are either disrupted or uncertain
    for aircraft_id in all_affected_aircraft:
        disruption_info = {
            "aircraft_id": aircraft_id,
            "is_disrupted": aircraft_id in disrupted_aircraft_ids,
            "is_uncertain": aircraft_id in uncertain_aircraft_ids
        }

        # Common time calculations for both cases
        start_date = config_dict['RecoveryPeriod']['StartDate']
        start_time_recovery = config_dict['RecoveryPeriod']['StartTime']
        delta_start = random.randint(min_delta_start_unavailability, max_delta_start_unavailability)
        start_unavail = (datetime.strptime(start_time_recovery, '%H:%M') +
                         timedelta(minutes=delta_start)).strftime('%H:%M')

        end_date = config_dict['RecoveryPeriod']['EndDate']
        start_unavail_obj = datetime.strptime(start_unavail, '%H:%M')
        unavail_period = random.randint(min_period_unavailability, max_period_unavailability)
        end_unavail = (start_unavail_obj + timedelta(minutes=unavail_period)).strftime('%H:%M')

        # Adjust end_date if end_unavail is earlier than start_unavail
        crosses_midnight = False
        if datetime.strptime(end_unavail, '%H:%M') < start_unavail_obj:
            end_date_obj = datetime.strptime(end_date, '%d/%m/%y')
            end_date = (end_date_obj + timedelta(days=1)).strftime('%d/%m/%y')
            crosses_midnight = True

        # Set probability based on aircraft category
        if aircraft_id in disrupted_aircraft_ids:
            probability = 1.00
        else:  # aircraft is in uncertain_aircraft_ids
            if probability_distribution.lower() == 'uniform':
                probability = random.uniform(probability_range[0], probability_range[1])
            elif probability_distribution.lower() == 'normal':
                mu = (probability_range[1] + probability_range[0]) / 2
                sigma = (probability_range[1] - probability_range[0]) / 6
                probability = np.clip(random.gauss(mu, sigma), probability_range[0], probability_range[1])
            elif probability_distribution.lower() == 'exponential':
                lambda_param = 1 / ((probability_range[1] - probability_range[0]) / 3)
                probability = probability_range[0] + random.expovariate(lambda_param)
                probability = min(probability, probability_range[1])
            else:
                probability = random.uniform(probability_range[0], probability_range[1])

        all_aircraft_data.append(f"{aircraft_id} {start_date} {start_unavail} {end_date} {end_unavail} {probability:.2f}")
        
        # Log disruption details
        disruption_info.update({
            "start_date": start_date,
            "start_time": start_unavail,
            "end_date": end_date,
            "end_time": end_unavail,
            "probability": probability,
            "delta_start_minutes": delta_start,
            "unavailability_period_minutes": unavail_period,
            "crosses_midnight": crosses_midnight
        })
        disruption_log["disruptions"].append(disruption_info)

    # Write to file
    with open(file_name, 'w') as file:
        file.write('%Aircraft StartDate StartTime EndDate EndTime Probability\n')
        for aircraft in all_aircraft_data:
            file.write(f"{aircraft}\n")
        file.write('#')

    return disruption_log


# Function to generate flights.csv
def generate_flights_file(file_name, aircraft_ids, average_flights_per_aircraft, std_dev_flights_per_aircraft, airports, config_dict, 
                          start_datetime, end_datetime, first_flight_dep_time_range, flight_length_range, time_between_flights_range):
    """Generates the flights.csv file."""
    clear_file(file_name)

    flights_dict = {}
    flight_rotation_data = {}

    total_flights = max(1, len(aircraft_ids) * average_flights_per_aircraft)  # Ensure at least 1 flight

    amount_flights_per_aircraft = {}
    flights_left_to_generate = total_flights
    min_flights_per_aircraft = max(1, average_flights_per_aircraft - std_dev_flights_per_aircraft)

    for aircraft_id in aircraft_ids:
        # Calculate maximum allowed flights for this aircraft
        remaining_aircraft = len(aircraft_ids) - len(amount_flights_per_aircraft)
        max_allowed = flights_left_to_generate - (remaining_aircraft - 1) * min_flights_per_aircraft
        
        amount_flights_per_aircraft[aircraft_id] = max(min_flights_per_aircraft, min(
            random.randint(average_flights_per_aircraft - std_dev_flights_per_aircraft, 
                          average_flights_per_aircraft + std_dev_flights_per_aircraft),
            max_allowed
        ))
        flights_left_to_generate -= amount_flights_per_aircraft[aircraft_id]

    current_flight_id = 1  # Keep track of the current flight ID

    for aircraft_id in aircraft_ids:
        last_arr_time = None  # Track the last arrival time for this aircraft

        for _ in range(amount_flights_per_aircraft[aircraft_id]):
            orig, dest = random.choice(airports), random.choice(airports)
            while orig == dest:
                dest = random.choice(airports)

            if last_arr_time is None:  # First flight for this aircraft
                # Ensure first flight starts on the same day by comparing with start_datetime's hour
                start_hour = start_datetime.hour
                min_dep_hour = max(start_hour, first_flight_dep_time_range[0])
                max_dep_hour = min(23, first_flight_dep_time_range[1])
                
                if min_dep_hour > max_dep_hour:  # If no valid same-day time slot exists
                    min_dep_hour = start_hour  # Default to start hour
                    max_dep_hour = start_hour + 1  # And the next hour
                
                dep_time = f"{random.randint(min_dep_hour, max_dep_hour)}:{random.choice(['00', '15', '30', '45'])}"
                dep_time_obj = datetime.strptime(f"{start_datetime.strftime('%d/%m/%y')} {dep_time}", '%d/%m/%y %H:%M')
            else:
                # Use this aircraft's last arrival time
                dep_time_obj = parse_time_with_day_offset(last_arr_time, start_datetime) + timedelta(
                    hours=random.randint(time_between_flights_range[0], time_between_flights_range[1] - 1),
                    minutes=random.randint(0, 59)
                )

            arr_time_obj = dep_time_obj + timedelta(
                hours=random.randint(flight_length_range[0], flight_length_range[1] - 1),
                minutes=random.randint(0, 59)
            )

            # Check time constraints
            if dep_time_obj > end_datetime + timedelta(hours=DEPARTURE_AFTER_END_RECOVERY):
                break
            if arr_time_obj > end_datetime:
                break

            # Format times with day offset when necessary
            if arr_time_obj.day > start_datetime.day:
                arr_time = f"{arr_time_obj.strftime('%H:%M')}+1"
            else:
                arr_time = arr_time_obj.strftime('%H:%M')

            if dep_time_obj.day > start_datetime.day:
                dep_time = f"{dep_time_obj.strftime('%H:%M')}+1"
            else:
                dep_time = dep_time_obj.strftime('%H:%M')

            # Add the flight
            flights_dict[current_flight_id] = {
                'Orig': orig,
                'Dest': dest,
                'DepTime': dep_time,
                'ArrTime': arr_time,
                'PrevFlight': 0,
                'Aircraft': aircraft_id
            }
            flight_rotation_data[current_flight_id] = {'Aircraft': aircraft_id}
            
            last_arr_time = arr_time
            current_flight_id += 1

    # Ensure at least one flight is generated
    if not flights_dict:
        flight_id = 1
        aircraft_id = aircraft_ids[0]
        orig, dest = random.choice(airports), random.choice(airports)
        while orig == dest:
            dest = random.choice(airports)
        
        dep_time = f"{random.randint(first_flight_dep_time_range[0], first_flight_dep_time_range[1])}:{random.choice(['00', '15', '30', '45'])}"
        dep_time_obj = parse_time_with_day_offset(dep_time, start_datetime)
        arr_time_obj = dep_time_obj + timedelta(hours=random.randint(flight_length_range[0], flight_length_range[1]), minutes=random.randint(0, 59))

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
            line = f"{flight_id} {flight_data['Orig']} {flight_data['Dest']} {flight_data['DepTime']} {flight_data['ArrTime']} {flight_data['PrevFlight']}\n"
            file.write(line)
        file.write('#')

    return flights_dict, flight_rotation_data



import random

# Function to generate rotations.csv
def generate_rotations_file(file_name, flight_rotation_data, start_datetime, clear_one_random_aircraft):
    """Generates the rotations.csv file."""
    # Clear the file before writing
    clear_file(file_name)

    # If clear_one_random_aircraft is True, randomly select one aircraft to clear its flights
    if clear_one_random_aircraft:
        all_aircraft = list(set(flight_data['Aircraft'] for flight_data in flight_rotation_data.values()))
        if all_aircraft:  # Ensure there are aircraft available to choose from
            aircraft_to_clear = random.choice(all_aircraft)
            # Remove flights assigned to the chosen aircraft
            flight_rotation_data = {
                flight_id: flight_data
                for flight_id, flight_data in flight_rotation_data.items()
                if flight_data['Aircraft'] != aircraft_to_clear
            }

    # Prepare rotations data
    rotations_data = []
    for flight_id, flight_data in flight_rotation_data.items():
        dep_date = start_datetime.strftime('%d/%m/%y')
        rotations_data.append(f"{flight_id} {dep_date} {flight_data['Aircraft']}")

    # Write the rotations data to the file
    with open(file_name, 'w') as file:
        file.write('%Flight DepDate Aircraft\n')
        for rotation in rotations_data:
            file.write(f"{rotation}\n")
        file.write('#')


"""


    # Call the function for each scenario
    create_data_scenario(
        scenario_name=scenario_name,
        template_folder=template_folder,
        data_root_folder=data_root_folder,
        aircraft_types=aircraft_types,
        total_aircraft_range=aircraft_range,  # Use the defined aircraft range
        amount_aircraft_disrupted=amount_aircraft_disrupted,  # Use the defined disrupted amount
        min_delta_start_unavailability=0,
        max_delta_start_unavailability=120,
        min_period_unavailability=120,
        max_period_unavailability=1020,
        average_flights_per_aircraft=average_flights_per_aircraft,  # Use the defined average flights per aircraft
        std_dev_flights_per_aircraft=1,  # Set a constant standard deviation
        airports=airports,
        config_dict=config_dict,
        recovery_start_date=recovery_start_date,
        recovery_start_time=recovery_start_time,
        recovery_end_date=recovery_end_date,
        recovery_end_time=recovery_end_time,
        clear_one_random_aircraft=False,
        probability_range=probability_range,
        probability_distribution=probability_distribution
    )

"""

def create_data_scenario(
    scenario_name, template_folder, data_root_folder, aircraft_types, total_aircraft_range,
    amount_aircraft_disrupted, amount_aircraft_uncertain, min_delta_start_unavailability, max_delta_start_unavailability,
    min_period_unavailability, max_period_unavailability, average_flights_per_aircraft,
    std_dev_flights_per_aircraft, airports, config_dict, recovery_start_date,
    recovery_start_time, recovery_end_date, recovery_end_time, clear_one_random_aircraft, 
    clear_random_flights, probability_range, probability_distribution, first_flight_dep_time_range, 
    flight_length_range, time_between_flights_range):
    """Creates a data scenario and returns the outputs."""

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
    disruptions = generate_alt_aircraft_file(
        alt_aircraft_file, aircraft_ids, amount_aircraft_disrupted, amount_aircraft_uncertain, config_dict, 
        min_delta_start_unavailability, max_delta_start_unavailability, 
        min_period_unavailability, max_period_unavailability, 
        probability_range, probability_distribution
    )

    # Generate flights data
    flights_file = os.path.join(data_folder, 'flights.csv')
    start_datetime = datetime.strptime(f"{recovery_start_date} {recovery_start_time}", '%d/%m/%y %H:%M')
    end_datetime = datetime.strptime(f"{recovery_end_date} {recovery_end_time}", '%d/%m/%y %H:%M')
    flights_dict, flight_rotation_data = generate_flights_file(
        flights_file, aircraft_ids, average_flights_per_aircraft, std_dev_flights_per_aircraft, 
        airports, config_dict, start_datetime, end_datetime, 
        first_flight_dep_time_range, flight_length_range, time_between_flights_range
    )

    # Generate rotations data
    rotations_file = os.path.join(data_folder, 'rotations.csv')
    generate_rotations_file(rotations_file, flight_rotation_data, start_datetime, clear_one_random_aircraft)

  
    # Collect inputs and outputs for logging
    inputs = {
        "scenario_name": scenario_name,
        "template_folder": template_folder,
        "data_root_folder": data_root_folder,
        "aircraft_types": aircraft_types,
        "aircraft_range": total_aircraft_range,
        "amount_aircraft_disrupted": amount_aircraft_disrupted,
        "unavailability_start_delta": {
            "min": min_delta_start_unavailability,
            "max": max_delta_start_unavailability,
        },
        "unavailability_duration": {
            "min": min_period_unavailability,
            "max": max_period_unavailability,
        },
        "average_flights_per_aircraft": average_flights_per_aircraft,
        "std_dev_flights_per_aircraft": std_dev_flights_per_aircraft,
        "airports": airports,
        "config_dict": config_dict,
        "recovery_period": {
            "start_date": recovery_start_date,
            "start_time": recovery_start_time,
            "end_date": recovery_end_date,
            "end_time": recovery_end_time,
        },
        "clear_one_random_aircraft": clear_one_random_aircraft,
        "clear_random_flights": clear_random_flights,
        "probability_range": probability_range,
        "probability_distribution": probability_distribution,
        "first_flight_dep_time_range": first_flight_dep_time_range,
        "flight_length_range": flight_length_range,
        "time_between_flights_range": time_between_flights_range,
    }

    outputs = {
        "total_aircraft": len(aircraft_ids),
        "total_flights": len(flights_dict),
        "disrupted_aircraft": amount_aircraft_disrupted,
        "disruption_probabilities": {
            "range": probability_range,
            "distribution": probability_distribution,
        },
        "aircraft_ids": aircraft_ids,
        "flights_per_aircraft": {
            aircraft: sum(1 for flight in flights_dict.values() if flight['Aircraft'] == aircraft) 
            for aircraft in aircraft_ids
        },
        "flights": flights_dict,
        "disruptions": disruptions,
    }

    print(f"Data creation for scenario {scenario_name} completed with {len(aircraft_ids)} aircraft and {len(flights_dict)} flights.")

    return data_folder, inputs, outputs