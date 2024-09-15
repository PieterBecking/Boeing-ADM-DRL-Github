import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scripts.utils import *
from stable_baselines3.common.evaluation import evaluate_policy
from src.config import *


# Callable entry point for visualization process
def run_visualization(scenario_name, data_root_folder, aircraft_rotations, airport_rotations):
    data_folder = os.path.join(data_root_folder, scenario_name)
    
    # Load data from CSV files
    data_dict = load_data(data_folder)

    # Visualize aircraft rotations
    if aircraft_rotations:
        print(f"Aircraft Rotations for {data_folder}")
        visualize_aircraft_rotations(data_dict)

    # Visualize flight and airport unavailability
    if airport_rotations:
        print(f"Flight and Airport Unavailability for {data_folder}")
        visualize_flight_airport_unavailability(data_dict)


    
# Visualization Functions
def visualize_aircraft_rotations(data_dict):
    """Visualizes aircraft rotations, delays, and unavailability in a state-plotter style."""
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_flights_dict = data_dict.get('alt_flights', {})
    alt_aircraft_dict = data_dict.get('alt_aircraft', {})
    config_dict = data_dict['config']

    # Time parsing
    start_datetime = datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + config_dict['RecoveryPeriod']['StartTime'], '%d/%m/%y %H:%M')
    end_datetime = datetime.strptime(config_dict['RecoveryPeriod']['EndDate'] + ' ' + config_dict['RecoveryPeriod']['EndTime'], '%d/%m/%y %H:%M')

    # Determine the earliest and latest times from the flight data
    earliest_datetime = min(
        min(parse_time_with_day_offset(flight_info['DepTime'], start_datetime) for flight_info in flights_dict.values()),
        start_datetime
    )
    latest_datetime = max(
        max(parse_time_with_day_offset(flight_info['ArrTime'], start_datetime) for flight_info in flights_dict.values()),
        end_datetime
    )

    # Aircraft IDs
    all_aircraft_ids = sorted(list(set([rotation_info['Aircraft'] for rotation_info in rotations_dict.values()])))
    aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(all_aircraft_ids)}

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    labels = {
        'Scheduled Flight': False,
        'Swapped Flight': False,
        'Environment Delayed Flight': False,
        'Aircraft Unavailable': False,
        'Disruption Start': False,
        'Disruption End': False,
        'Delay of Flight': False
    }

    # Plot each flight's schedule in blue
    for rotation_id, rotation_info in rotations_dict.items():
        flight_id = rotation_id
        aircraft_id = rotation_info['Aircraft']

        if flight_id in flights_dict:
            flight_info = flights_dict[flight_id]
            dep_datetime = parse_time_with_day_offset(flight_info['DepTime'], start_datetime)
            arr_datetime = parse_time_with_day_offset(flight_info['ArrTime'], dep_datetime)

            # Standard flight plot
            plot_color = 'blue'
            plot_label = 'Scheduled Flight'

            y_offset = aircraft_indices[aircraft_id]

            ax.plot([dep_datetime, arr_datetime], [y_offset, y_offset], color=plot_color, label=plot_label if not labels['Scheduled Flight'] else "")
            labels['Scheduled Flight'] = True

            # Departure and arrival markers
            marker_offset = timedelta(minutes=4)  # Slight offset for markers
            ax.plot(dep_datetime + marker_offset, y_offset, color=plot_color, marker='>', markersize=6)
            ax.plot(arr_datetime - marker_offset, y_offset, color=plot_color, marker='<', markersize=6)

            # Flight ID
            mid_datetime = dep_datetime + (arr_datetime - dep_datetime) / 2
            ax.text(mid_datetime, y_offset - 0.05, flight_id, ha='center', va='bottom', fontsize=10, color='black')

    # Plot flight delays in red (if any)
    for flight_id, disruption_info in alt_flights_dict.items() if alt_flights_dict else []:
        if flight_id in flights_dict:
            dep_datetime = parse_time_with_day_offset(flights_dict[flight_id]['DepTime'], start_datetime)
            delay_duration = timedelta(minutes=disruption_info['Delay'])
            delayed_datetime = dep_datetime + delay_duration

            aircraft_id = rotations_dict[flight_id]['Aircraft']
            y_offset = aircraft_indices[aircraft_id]

            # Flight delay markers and lines
            ax.plot(dep_datetime, y_offset, 'X', color='red', label='Flight Delay Start' if not labels['Delay of Flight'] else "")
            ax.plot([dep_datetime, delayed_datetime], [y_offset, y_offset], color='red', linestyle='-', label='Flight Delay' if not labels['Delay of Flight'] else "")
            ax.plot(delayed_datetime, y_offset, '>', color='red', label='Flight Delay End' if not labels['Delay of Flight'] else "")
            labels['Delay of Flight'] = True

    # Plot aircraft unavailability
    for aircraft_id, unavailability_info in alt_aircraft_dict.items():
        unavail_start_datetime = datetime.strptime(unavailability_info['StartDate'] + ' ' + unavailability_info['StartTime'], '%d/%m/%y %H:%M')
        unavail_end_datetime = datetime.strptime(unavailability_info['EndDate'] + ' ' + unavailability_info['EndTime'], '%d/%m/%y %H:%M')

        if aircraft_id in aircraft_indices:
            y_offset = aircraft_indices[aircraft_id]

            ax.plot([unavail_start_datetime, unavail_end_datetime], [y_offset, y_offset], color='red', linestyle='--', label='Aircraft Unavailable' if not labels['Aircraft Unavailable'] else "")
            labels['Aircraft Unavailable'] = True
            ax.plot(unavail_start_datetime, y_offset, 'rx', label='Disruption Start' if not labels['Disruption Start'] else "")
            ax.plot(unavail_end_datetime, y_offset, 'rx', label='Disruption End' if not labels['Disruption End'] else "")

    # Plot recovery period
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
    ytick_labels = [f"{index}: {aircraft_id}" for index, aircraft_id in enumerate(all_aircraft_ids, start=1)]
    plt.yticks(range(1, len(all_aircraft_ids) + 1), ytick_labels)
    plt.xlabel('Time')
    plt.ylabel('Aircraft')
    plt.title('Aircraft Rotations, AC Unavailability, and Flight Delays')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create legend on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Reverse y-axis to match the state plotter
    ax.invert_yaxis()

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







# StatePlotter class for visualizing the state of the environment
class StatePlotter:
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, start_datetime, end_datetime, 
                 offset_baseline=0, offset_id_number=-0.05, offset_delayed_flight=0, offset_marker_minutes=4):
        self.aircraft_dict = aircraft_dict
        self.initial_flights_dict = flights_dict
        self.rotations_dict = rotations_dict
        self.alt_aircraft_dict = alt_aircraft_dict
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        
        # Offsets as inputs
        self.offset_baseline = offset_baseline
        self.offset_id_number = offset_id_number
        self.offset_delayed_flight = offset_delayed_flight
        self.offset_marker_minutes = offset_marker_minutes

        aircraft_id_to_idx = {aircraft_id: idx + 1 for idx, aircraft_id in enumerate(aircraft_dict.keys())}
        self.aircraft_id_to_idx = aircraft_id_to_idx
        
        # Calculate the earliest and latest datetimes
        self.earliest_datetime = min(
            min(parse_time_with_day_offset(flight_info['DepTime'], start_datetime) for flight_info in flights_dict.values()),
            start_datetime
        )
        self.latest_datetime = max(
            max(parse_time_with_day_offset(flight_info['ArrTime'], start_datetime) for flight_info in flights_dict.values()),
            end_datetime
        )

    def plot_state(self, flights_dict, swapped_flights, environment_delayed_flights, current_datetime):
        if DEBUG_MODE:
            print(f"Plotting state with following flights: {flights_dict}")

        updated_rotations_dict = self.rotations_dict.copy()
        for swap in swapped_flights:
            flight_id, new_aircraft_id = swap
            updated_rotations_dict[flight_id]['Aircraft'] = new_aircraft_id

        all_aircraft_ids = set([rotation_info['Aircraft'] for rotation_info in updated_rotations_dict.values()]).union(set(self.aircraft_dict.keys()))
        aircraft_ids = sorted(list(all_aircraft_ids), reverse=False)
        aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(aircraft_ids)}

        fig, ax = plt.subplots(figsize=(14, 8))

        labels = {
            'Scheduled Flight': False,
            'Swapped Flight': False,
            'Environment Delayed Flight': False,
            'Aircraft Unavailable': False,
            'Disruption Start': False,
            'Disruption End': False,
            'Delay of Flight': False
        }

        for rotation_id, rotation_info in updated_rotations_dict.items():
            flight_id = rotation_id
            aircraft_id = rotation_info['Aircraft']
            
            if flight_id in flights_dict:
                flight_info = flights_dict[flight_id]
                dep_datetime_str = flight_info['DepTime']
                arr_datetime_str = flight_info['ArrTime']
                
                dep_datetime = parse_time_with_day_offset(dep_datetime_str, self.start_datetime)
                arr_datetime = parse_time_with_day_offset(arr_datetime_str, dep_datetime)
                
                swapped = any(flight_id == swap[0] for swap in swapped_flights)
                delayed = flight_id in environment_delayed_flights
                plot_color = 'green' if swapped else 'orange' if delayed else 'blue'
                plot_label = 'Swapped Flight' if swapped else 'Environment Delayed Flight' if delayed else 'Scheduled Flight'
                
                y_offset = aircraft_indices[aircraft_id] + self.offset_baseline
                if delayed:
                    y_offset += self.offset_delayed_flight

                # Plot the line representing the flight without markers
                ax.plot([dep_datetime, arr_datetime], [y_offset, y_offset], color=plot_color, label=plot_label if (swapped and not labels['Swapped Flight']) or (delayed and not labels['Environment Delayed Flight']) or (not swapped and not delayed and not labels['Scheduled Flight']) else None)
                # Offset the markers slightly by reducing/increasing time
                marker_offset = timedelta(minutes=self.offset_marker_minutes)

                # Plot departure marker with slight offset
                ax.plot(dep_datetime + marker_offset, y_offset, color=plot_color, marker='>', markersize=6, markeredgewidth=0)

                # Plot arrival marker with slight offset
                ax.plot(arr_datetime - marker_offset, y_offset, color=plot_color, marker='<', markersize=6, markeredgewidth=0)

                if delayed:
                    # Draw a vertical line at the departure and arrival times to indicate delay
                    ax.vlines([dep_datetime, arr_datetime], y_offset - self.offset_delayed_flight, y_offset, color='orange', linestyle='-', linewidth=2)

                labels[plot_label] = True
                
                # Calculate the midpoint between departure and arrival for placing the flight ID
                mid_datetime = dep_datetime + (arr_datetime - dep_datetime) / 2

                # flight id above line
                ax.text(mid_datetime, y_offset + self.offset_id_number, flight_id, 
                        ha='center', va='bottom', fontsize=10, color='black')

        if self.alt_aircraft_dict:
            for aircraft_id, unavailability_info in self.alt_aircraft_dict.items():
                start_date = unavailability_info['StartDate']
                start_time = unavailability_info['StartTime']
                end_date = unavailability_info['EndDate']
                end_time = unavailability_info['EndTime']
                
                unavail_start_datetime = datetime.strptime(start_date + ' ' + start_time, '%d/%m/%y %H:%M')
                unavail_end_datetime = datetime.strptime(end_date + ' ' + end_time, '%d/%m/%y %H:%M')
                
                if aircraft_id in aircraft_indices:
                    if not labels['Aircraft Unavailable']:
                        ax.plot([unavail_start_datetime, unavail_end_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='red', linestyle='--', label='Aircraft Unavailable')
                        labels['Aircraft Unavailable'] = True
                    else:
                        ax.plot([unavail_start_datetime, unavail_end_datetime], [aircraft_indices[aircraft_id], aircraft_indices[aircraft_id]], color='red', linestyle='--')
                    if not labels['Disruption Start']:
                        ax.plot(unavail_start_datetime, aircraft_indices[aircraft_id], 'rx', label='Disruption Start')
                        labels['Disruption Start'] = True
                    else:
                        ax.plot(unavail_start_datetime, aircraft_indices[aircraft_id], 'rx')
                    if not labels['Disruption End']:
                        ax.plot(unavail_end_datetime, aircraft_indices[aircraft_id], 'rx', label='Disruption End')
                        labels['Disruption End'] = True
                    else:
                        ax.plot(unavail_end_datetime, aircraft_indices[aircraft_id], 'rx')

        ax.set_xlim(self.earliest_datetime - timedelta(hours=1), self.latest_datetime + timedelta(hours=1))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.axvline(self.start_datetime, color='green', linestyle='--', label='Start Recovery Period')
        ax.axvline(self.end_datetime, color='green', linestyle='-', label='End Recovery Period')
        ax.axvline(current_datetime, color='black', linestyle='-', label='Current Time')

        ax.axvspan(self.end_datetime, self.latest_datetime + timedelta(hours=1), color='lightgrey', alpha=0.3)
        ax.axvspan(self.earliest_datetime - timedelta(hours=1), self.start_datetime, color='lightgrey', alpha=0.3)
        
        # Reverse the y-axis
        ax.invert_yaxis()

        # Generate y-ticks with correct aircraft string
        ytick_labels = [f"{index + 1}: {aircraft_id}" for index, aircraft_id in enumerate(aircraft_ids)]

        plt.yticks(range(1, len(aircraft_ids) + 1), ytick_labels)

        plt.xlabel('Time')
        plt.ylabel('Aircraft')
        plt.title('Aircraft Rotations and Unavailability')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()



def plot_dqn_performance(rewards, epsilon_values, model, env, window=100, n_eval_episodes=10):
    """
    Plots DQN performance metrics such as reward progression and epsilon decay, and evaluates the model.

    Parameters:
    - rewards: List or np.array of rewards for each episode.
    - epsilon_values: List or np.array of epsilon values over the episodes.
    - model: The trained model to evaluate.
    - env: The environment to evaluate the model on.
    - window: The window size for the trailing average calculation (default is 100).
    - n_eval_episodes: Number of episodes to use for evaluating the model (default is 10).
    
    Returns:
    - mean_reward: The mean reward obtained from evaluating the policy.
    - std_reward: The standard deviation of the rewards obtained during evaluation.
    """
    
    # Flatten rewards if necessary
    rewards = np.array(rewards).flatten()

    # Calculate the trailing average of the rewards using np.convolve
    trailing_average = np.convolve(rewards, np.ones(window), 'valid') / window

    # Plotting the rewards to visualize learning
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(trailing_average, label='Trailing Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Performance')
    plt.legend()
    plt.show()

    # Suppress prints temporarily (because evaluate_policy prints debug info by calling step function)
    sys.stdout = open(os.devnull, 'w')

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    # Re-enable printing
    sys.stdout = sys.__stdout__

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Plot the epsilon values over the episodes
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Value Decay')
    plt.show()

    return mean_reward, std_reward




def plot_epsilon_decay(n_episodes, epsilon_start, epsilon_min, decay_rate):
    """ Plots the epsilon decay over a number of episodes using an exponential decay formula """
    epsilon_values = []

    for episode in range(n_episodes):
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(-decay_rate * episode)
        epsilon_values.append(epsilon)

    # Plot the epsilon decay curve
    plt.plot(epsilon_values)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Exponential Decay Curve")
    plt.show()
