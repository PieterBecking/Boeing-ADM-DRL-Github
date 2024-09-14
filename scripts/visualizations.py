import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scripts.utils import *


# Visualization Functions
def visualize_aircraft_rotations(data_dict):
    """Visualizes aircraft rotations, delays, and unavailability."""
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_flights_dict = data_dict.get('alt_flights', {})
    alt_aircraft_dict = data_dict.get('alt_aircraft', {})
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

    # Plot logic
    aircraft_ids = sorted(list(set([rotation_info['Aircraft'] for rotation_info in rotations_dict.values()])), reverse=False)
    aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(aircraft_ids)}

    # Plot each flight's schedule in blue
    for rotation_id, rotation_info in rotations_dict.items():
        flight_id = rotation_id
        aircraft_id = rotation_info['Aircraft']

        if flight_id in flights_dict:
            flight_info = flights_dict[flight_id]
            dep_datetime = parse_time_with_day_offset(flight_info['DepTime'], start_datetime)
            arr_datetime = parse_time_with_day_offset(flight_info['ArrTime'], dep_datetime)

            ax.plot([dep_datetime, arr_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='blue', marker='o', label='Scheduled Flight' if rotation_id == 1 else "")

    # Plot flight disruptions (delays) in red
    if alt_flights_dict:
        for flight_id, disruption_info in alt_flights_dict.items():
            if flight_id in flights_dict:
                dep_datetime = parse_time_with_day_offset(flights_dict[flight_id]['DepTime'], start_datetime)
                delay_duration = timedelta(minutes=disruption_info['Delay'])
                delayed_datetime = dep_datetime + delay_duration

                aircraft_id = rotations_dict[flight_id]['Aircraft']
                ax.plot(dep_datetime, aircraft_indices[aircraft_id], 'X', color='red', label='Flight Delay Start' if flight_id == list(alt_flights_dict.keys())[0] else "")
                ax.plot([dep_datetime, delayed_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='red', linestyle='-', label='Flight Delay' if flight_id == list(alt_flights_dict.keys())[0] else "")
                ax.plot(delayed_datetime, aircraft_indices[aircraft_id], '>', color='red', label='Flight Delay End' if flight_id == list(alt_flights_dict.keys())[0] else "")

    # Plot aircraft unavailability in red
    if alt_aircraft_dict:
        for aircraft_id, unavailability_info in alt_aircraft_dict.items():
            unavail_start_datetime = datetime.strptime(unavailability_info['StartDate'] + ' ' + unavailability_info['StartTime'], '%d/%m/%y %H:%M')
            unavail_end_datetime = datetime.strptime(unavailability_info['EndDate'] + ' ' + unavailability_info['EndTime'], '%d/%m/%y %H:%M')

            ax.plot(unavail_start_datetime, aircraft_indices[aircraft_id], 'rx', label='Unavailability Start' if aircraft_id == list(alt_aircraft_dict.keys())[0] else "")
            ax.plot([unavail_start_datetime, unavail_end_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='red', linestyle='--', label='Aircraft Unavailable' if aircraft_id == list(alt_aircraft_dict.keys())[0] else "")
            ax.plot(unavail_end_datetime, aircraft_indices[aircraft_id], 'rx', label='Unavailability End' if aircraft_id == list(alt_aircraft_dict.keys())[0] else "")

    # Plot the recovery period
    ax.axvline(start_datetime, color='green', linestyle='--', label='Start Recovery Period')
    ax.axvline(end_datetime, color='green', linestyle='-', label='End Recovery Period')

    # Grey out periods outside recovery time
    ax.axvspan(end_datetime, latest_datetime + timedelta(hours=1), color='lightgrey', alpha=0.3)
    ax.axvspan(earliest_datetime - timedelta(hours=1), start_datetime, color='lightgrey', alpha=0.3)

    # Formatting the plot
    ax.set_xlim(earliest_datetime - timedelta(hours=1), latest_datetime + timedelta(hours=1))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Set y-ticks for aircraft indices
    plt.yticks(range(1, len(aircraft_ids) + 1), aircraft_ids)
    plt.xlabel('Time')
    plt.ylabel('Aircraft')
    plt.title('Aircraft Rotations, AC Unavailability and Flight Delays')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create legend on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def visualize_flight_airport_unavailability(data_dict):
    """Visualizes flight schedules and airport unavailability."""
    flights_dict = data_dict['flights']
    alt_airports_dict = data_dict.get('alt_airports', {})
    config_dict = data_dict['config']

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

    # Collect airports from both origins and destinations and sort them alphabetically
    airports = sorted(list(set([flight_info['Orig'] for flight_info in flights_dict.values()] + [flight_info['Dest'] for flight_info in flights_dict.values()])))
    airport_indices = {airport: index + 1 for index, airport in enumerate(airports)}

    # Plot each flight's schedule based on airports in blue
    for flight_id, flight_info in flights_dict.items():
        dep_datetime = parse_time_with_day_offset(flight_info['DepTime'], start_datetime)
        arr_datetime = parse_time_with_day_offset(flight_info['ArrTime'], dep_datetime)
        ax.plot([dep_datetime, arr_datetime], [airport_indices[flight_info['Orig']], airport_indices[flight_info['Dest']]], color='blue', marker='o', label='Scheduled Flight' if flight_id == 1 else "")

    # Track which labels have been added to the legend
    labels_added = set()

    # Plot airport disruptions with different styles
    if alt_airports_dict:
        for airport, disruptions in alt_airports_dict.items():
            for disruption_info in disruptions:
                unavail_start_datetime = datetime.strptime(disruption_info['StartDate'] + ' ' + disruption_info['StartTime'], '%d/%m/%y %H:%M')
                unavail_end_datetime = datetime.strptime(disruption_info['EndDate'] + ' ' + disruption_info['EndTime'], '%d/%m/%y %H:%M')

                dep_h = disruption_info['Dep/h']
                arr_h = disruption_info['Arr/h']
                
                if dep_h == 0 and arr_h == 0:
                    linestyle = 'solid'
                    linewidth = 3
                    label = 'Completely Closed'
                elif dep_h == 0 or arr_h == 0:
                    linestyle = 'solid'
                    linewidth = 1
                    label = 'Partially Closed (Dep/Arr)'
                else:
                    linestyle = 'dashed'
                    linewidth = 1
                    label = 'Constrained'

                # Only add each label once
                if label not in labels_added:
                    ax.plot([unavail_start_datetime, unavail_end_datetime], [airport_indices[airport], airport_indices[airport]], color='red', linestyle=linestyle, linewidth=linewidth, label=label)
                    labels_added.add(label)
                else:
                    ax.plot([unavail_start_datetime, unavail_end_datetime], [airport_indices[airport], airport_indices[airport]], color='red', linestyle=linestyle, linewidth=linewidth)
                ax.plot(unavail_start_datetime, airport_indices[airport], 'rx')
                ax.plot(unavail_end_datetime, airport_indices[airport], 'rx')

    # Formatting the plot
    ax.set_xlim(earliest_datetime - timedelta(hours=1), latest_datetime + timedelta(hours=1))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.axvline(start_datetime, color='green', linestyle='--', label='Start Recovery Period')
    ax.axvline(end_datetime, color='green', linestyle='-', label='End Recovery Period')

    # Grey out periods outside recovery time
    ax.axvspan(end_datetime, latest_datetime + timedelta(hours=1), color='lightgrey', alpha=0.3)
    ax.axvspan(earliest_datetime - timedelta(hours=1), start_datetime, color='lightgrey', alpha=0.3)

    # Set y-ticks for airport indices
    plt.yticks(range(1, len(airport_indices) + 1), airport_indices.keys())
    plt.xlabel('Time')
    plt.ylabel('Airports')
    plt.title('Flights and Airport Unavailability')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create legend on the right
    ax.legend()

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

