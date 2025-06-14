import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scripts.utils import parse_time_with_day_offset, load_data
from stable_baselines3.common.evaluation import evaluate_policy
from src.config import *
import matplotlib.patches as patches
from scripts.utils import *
import re  # Make sure to import re at the top

# StatePlotter class for visualizing the state of the environment
class StatePlotter:
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, start_datetime, end_datetime, 
                 uncertain_breakdowns=None, offset_baseline=0, offset_id_number=-0.05, offset_delayed_flight=0, offset_marker_minutes=4, plot_title=None, last_flight_action=None):
        self.aircraft_dict = aircraft_dict
        self.initial_flights_dict = flights_dict
        self.rotations_dict = rotations_dict
        self.alt_aircraft_dict = alt_aircraft_dict
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.last_flight_action = last_flight_action
        # Initialize uncertain_breakdowns
        self.uncertain_breakdowns = uncertain_breakdowns if uncertain_breakdowns is not None else {}
        
        # Offsets as inputs
        self.offset_baseline = offset_baseline
        self.offset_id_number = offset_id_number
        self.offset_delayed_flight = offset_delayed_flight
        self.offset_marker_minutes = offset_marker_minutes
        self.plot_title = plot_title
        # Define the sorting key function
        def extract_sort_key(aircraft_id):
            letters = ''.join(re.findall(r'[A-Za-z]+', aircraft_id))
            numbers = tuple(int(num) for num in re.findall(r'\d+', aircraft_id))
            return (letters, ) + numbers

        # Sort aircraft IDs using the custom sort key
        sorted_aircraft_ids = sorted(self.aircraft_dict.keys(), key=extract_sort_key)
        self.aircraft_id_to_idx = {aircraft_id: idx + 1 for idx, aircraft_id in enumerate(sorted_aircraft_ids)}
        # Calculate the earliest and latest datetimes
        self.earliest_datetime = min(
            min(parse_time_with_day_offset(flight_info['DepTime'], start_datetime) for flight_info in flights_dict.values()),
            start_datetime
        )
        self.latest_datetime = max(
            max(parse_time_with_day_offset(flight_info['ArrTime'], start_datetime) for flight_info in flights_dict.values()),
            end_datetime
        )

    def plot_state(self, flights_dict, swapped_flights, environment_delayed_flights, cancelled_flights, current_datetime, 
                  title_appendix="", debug_print=False, show_plot=True, reward_and_action=None, last_flight_action=None,
                  show_flights=True, show_certain_disruptions=True, show_uncertain_disruptions=True,
                  show_current_time=True, show_recovery_window=True):
        if debug_print:
            print(f"Plotting state with following flights: {flights_dict}")

        # Set up the figure first
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up date formatter and locator before any plotting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
        
        updated_rotations_dict = self.rotations_dict.copy()
        for swap in swapped_flights:
            flight_id, new_aircraft_id = swap
            updated_rotations_dict[flight_id]['Aircraft'] = new_aircraft_id

        # Define the sorting key function
        def extract_sort_key(aircraft_id):
            letters = ''.join(re.findall(r'[A-Za-z]+', aircraft_id))
            numbers = tuple(int(num) for num in re.findall(r'\d+', aircraft_id))
            return (letters, ) + numbers

        # Collect and sort aircraft IDs using the custom sort key
        all_aircraft_ids = sorted(list(set([rotation_info['Aircraft'] for rotation_info in updated_rotations_dict.values()]).union(set(self.aircraft_dict.keys()))))
        aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(all_aircraft_ids)}

        # Add permanent invisible flights for first and last aircraft to maintain proper spacing
        if all_aircraft_ids:
            # Add invisible flight for first aircraft
            first_aircraft_id = all_aircraft_ids[0]
            first_y_offset = aircraft_indices[first_aircraft_id]
            dummy_start = self.start_datetime
            dummy_end = dummy_start + timedelta(minutes=30)
            ax.plot([dummy_start, dummy_end], [first_y_offset, first_y_offset], color='blue', alpha=0)

            # Add invisible flight for last aircraft
            last_aircraft_id = all_aircraft_ids[-1]
            last_y_offset = aircraft_indices[last_aircraft_id]
            ax.plot([dummy_start, dummy_end], [last_y_offset, last_y_offset], color='blue', alpha=0)

        labels = {
            'Scheduled Flight': False,
            'Swapped Flight': False,
            'Environment Delayed Flight': False,
            'Cancelled Flight': False,
            'Aircraft Unavailable': False,
            'Disruption Start': False,
            'Disruption End': False,
            'Delay of Flight': False,
            'Uncertain Breakdown': False,
            'Zero Probability': False,
            'Current Action Flight': False,
            'Last Action Flight': False
        }

        earliest_time = self.earliest_datetime
        latest_time = self.latest_datetime

        # Only plot flights if show_flights is True
        if show_flights:
            for rotation_id, rotation_info in updated_rotations_dict.items():
                flight_id = rotation_id
                aircraft_id = rotation_info['Aircraft']
                
                if flight_id in flights_dict:
                    flight_info = flights_dict[flight_id]
                    dep_datetime_str = flight_info['DepTime']
                    arr_datetime_str = flight_info['ArrTime']
                    
                    dep_datetime = parse_time_with_day_offset(dep_datetime_str, self.start_datetime)
                    arr_datetime = parse_time_with_day_offset(arr_datetime_str, dep_datetime)
                    
                    # Handle flights crossing midnight properly
                    if '+1' in dep_datetime_str and '+1' in arr_datetime_str:
                        # Both departure and arrival are on next day
                        dep_datetime = self.start_datetime + timedelta(days=1)
                        dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                         minute=int(dep_datetime_str.split(':')[1].split('+')[0]))
                        arr_datetime = dep_datetime + timedelta(days=0)  # Same day as departure
                        arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                         minute=int(arr_datetime_str.split(':')[1].split('+')[0]))
                    elif '+1' in dep_datetime_str:
                        # Only departure is next day
                        dep_datetime = self.start_datetime + timedelta(days=1)
                        dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                         minute=int(dep_datetime_str.split(':')[1].split('+')[0]))
                        arr_datetime = dep_datetime
                        arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                         minute=int(arr_datetime_str.split(':')[1]))
                    elif '+1' in arr_datetime_str:
                        # Only arrival is next day relative to departure
                        dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                         minute=int(dep_datetime_str.split(':')[1]))
                        arr_datetime = dep_datetime + timedelta(days=1)
                        arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                         minute=int(arr_datetime_str.split(':')[1].split('+')[0]))
                    
                    # Recalculate earliest and latest times after fixing arr_datetime
                    earliest_time = min(earliest_time, dep_datetime)
                    latest_time = max(latest_time, arr_datetime)
                    
                    swapped = any(flight_id == swap[0] for swap in swapped_flights)
                    delayed = flight_id in environment_delayed_flights
                    cancelled = flight_id in cancelled_flights
                    
                    # Debug prints for last flight action
                    print(f"Current flight_id: {flight_id}, type: {type(flight_id)}")
                    print(f"Last flight action: {last_flight_action}, type: {type(last_flight_action) if last_flight_action else None}")
                    if last_flight_action:
                        print(f"Last flight action[0]: {last_flight_action[0]}, type: {type(last_flight_action[0])}")
                    
                    # Check if this is the last flight action using the parameter - compare as integers
                    is_last_action = last_flight_action and flight_id == last_flight_action[0]
                    print(f"is_last_action: {is_last_action}")
                    
                    if cancelled:
                        plot_color = 'red'
                        plot_label = 'Cancelled Flight'
                    elif is_last_action:
                        print("Setting color to green for last action!")
                        plot_color = 'green'
                        plot_label = 'Last Action Flight'
                    elif swapped:
                        plot_color = 'blue'
                        plot_label = 'Swapped Flight'
                    elif delayed:
                        plot_color = 'orange'
                        plot_label = 'Environment Delayed Flight'
                    else:
                        plot_color = 'blue'
                        plot_label = 'Scheduled Flight'
                    
                    y_offset = aircraft_indices[aircraft_id] + self.offset_baseline
                    if delayed:
                        y_offset += self.offset_delayed_flight

                    ax.plot([dep_datetime, arr_datetime], [y_offset, y_offset], color=plot_color, label=plot_label if not labels[plot_label] else None)
                    
                    marker_offset = timedelta(minutes=self.offset_marker_minutes)
                    dep_marker = dep_datetime + marker_offset
                    arr_marker = arr_datetime - marker_offset

                    ax.plot(dep_marker, y_offset, color=plot_color, marker='>', markersize=6, markeredgewidth=0)
                    ax.plot(arr_marker, y_offset, color=plot_color, marker='<', markersize=6, markeredgewidth=0)

                    if delayed:
                        ax.vlines([dep_datetime, arr_datetime], y_offset - self.offset_delayed_flight, y_offset, color='orange', linestyle='-', linewidth=2)

                    labels[plot_label] = True
                    
                    mid_datetime = dep_datetime + (arr_datetime - dep_datetime) / 2
                    ax.text(mid_datetime, y_offset + self.offset_id_number, flight_id, 
                            ha='center', va='bottom', fontsize=10, color='black')

        # Function to compute the data height that corresponds to a given number of pixels
        def get_height_in_data_units(ax, pixels):
            # Transform data coordinates (0, y) to display coordinates (pixels)
            y0_display = ax.transData.transform((0, 0))[1]
            y1_display = ax.transData.transform((0, 1))[1]
            pixels_per_data_unit = abs(y1_display - y0_display)
            data_units_per_pixel = 1 / pixels_per_data_unit
            return data_units_per_pixel * pixels

        # Compute rectangle height in data units corresponding to 60 pixels (doubled from 30)
        rect_height = get_height_in_data_units(ax, 150)
        # Handle alt_aircraft_dict unavailabilities, including uncertain ones with probability < 1.0
        if self.alt_aircraft_dict:
            for aircraft_id, unavailability_info in self.alt_aircraft_dict.items():
                if not isinstance(unavailability_info, list):
                    unavailability_info = [unavailability_info]
                
                for unavail in unavailability_info:
                    start_date = unavail['StartDate']
                    start_time = unavail['StartTime']
                    end_date = unavail['EndDate']
                    end_time = unavail['EndTime']
                    probability = unavail.get('Probability', 1.0)  # Default to 1.0 if Probability is not given
                    
                    # Convert to datetime objects
                    unavail_start = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
                    unavail_end = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
                    y_offset = aircraft_indices[aircraft_id]

                    if np.isnan(probability):
                        probability = 0.0
                    
                    # Set color based on probability
                    if probability == 0.0:
                        if not show_certain_disruptions:
                            continue
                        rect_color = 'lightgrey'  # Very light grey for zero probability
                        plot_label = 'Zero Probability'
                    elif probability < 1.0:
                        if not show_uncertain_disruptions:
                            continue
                        rect_color = 'orange'  # Uncertain breakdown
                        plot_label = 'Uncertain Breakdown'
                    else:
                        if not show_certain_disruptions:
                            continue
                        rect_color = 'red'  # Certain unavailability
                        plot_label = 'Aircraft Unavailable'
                    
                    # Plot the unavailability period as a rectangle
                    rect = patches.Rectangle((unavail_start, y_offset - rect_height / 2),
                                            unavail_end - unavail_start,
                                            rect_height,
                                            linewidth=0,
                                            color=rect_color,
                                            alpha=0.3,
                                            zorder=0,
                                            label=plot_label if not labels[plot_label] else None)
                    ax.add_patch(rect)
                    labels[plot_label] = True

                    # Plot the probability below the rectangle
                    if probability < 1.0 and show_uncertain_disruptions:
                        x_position = unavail_start + (unavail_end - unavail_start) / 2
                        y_position = y_offset - rect_height / 2 - get_height_in_data_units(ax, 10) +0.2  # Adjust offset as needed
                        ax.text(x_position, y_position + 0.1, f"{probability:.2f}", ha='center', va='top', fontsize=9)

        # Add transparent rectangles for each aircraft at start of recovery period
        for aircraft_id in all_aircraft_ids:
            y_offset = aircraft_indices[aircraft_id]
            rect_start = self.start_datetime - timedelta(hours=1)
            rect = patches.Rectangle(
                (rect_start, y_offset - rect_height / 2),
                timedelta(minutes=1),  # Very thin rectangle
                rect_height,
                linewidth=0,
                color='white',  # Color doesn't matter since alpha is 0
                alpha=0,  # Fully transparent
                zorder=0
            )
            ax.add_patch(rect)

        # After all plotting is done, set the x-axis limits and format
        buffer_time = timedelta(hours=1)
        ax.set_xlim(earliest_time - buffer_time, latest_time + buffer_time)
        
        # Add day markers if the time range spans multiple days
        if (latest_time.date() - earliest_time.date()).days > 0:
            current_date = earliest_time.date()
            while current_date <= latest_time.date():
                next_day = datetime.combine(current_date + timedelta(days=1), datetime.min.time())
                if next_day >= earliest_time and next_day <= latest_time:
                    ax.axvline(x=next_day, color='gray', linestyle='--', alpha=0.5)
                    ax.text(next_day, ax.get_ylim()[1], f' Day +{(current_date - self.start_datetime.date()).days + 1}',
                           rotation=90, va='bottom', ha='right')
                current_date += timedelta(days=1)
        
        plt.xticks(rotation=45)

        # Plot recovery window and current time based on boolean flags
        if show_recovery_window:
            ax.axvline(self.start_datetime, color='green', linestyle='--', linewidth=2, label='Start/End Recovery Period')
            ax.axvline(self.end_datetime, color='green', linestyle='--', linewidth=2)  # Same style, no label
            ax.axvspan(self.end_datetime, latest_time + timedelta(hours=1), color='lightgrey', alpha=0.3)
            ax.axvspan(earliest_time - timedelta(hours=1), self.start_datetime, color='lightgrey', alpha=0.3)

        if show_current_time:
            ax.axvline(current_datetime, color='black', linestyle='-', label='Current Time')

        ax.invert_yaxis()

        ytick_labels = [f"{index + 1}: {aircraft_id}" for index, aircraft_id in enumerate(all_aircraft_ids)]
        plt.yticks(range(1, len(all_aircraft_ids) + 1), ytick_labels)

        plt.xlabel('Time')
        plt.ylabel('Aircraft')
        if self.plot_title:
            title = f"Aircraft rotations and unavailability {self.plot_title}"  
        else:
            title = f'Aircraft Rotations and Unavailability {title_appendix.title()}'
        if reward_and_action:
            title += f' (Reward: {reward_and_action[0]:.2f}, Action: (flight: {reward_and_action[1][0]}, aircraft: {reward_and_action[1][1]}), Total Reward: {reward_and_action[2]:.2f})'
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
        if show_plot:
            plt.show()
        else:
            plt.close()
            return fig



# StatePlotter class for visualizing the state of the environment
class StatePlotterDemo:
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, start_datetime, end_datetime, 
                 uncertain_breakdowns=None, offset_baseline=0, offset_id_number=-0.02, offset_delayed_flight=0, offset_marker_minutes=4, plot_title=None):
        self.aircraft_dict = aircraft_dict
        self.initial_flights_dict = flights_dict
        self.rotations_dict = rotations_dict
        self.alt_aircraft_dict = alt_aircraft_dict
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        
        # Initialize uncertain_breakdowns
        self.uncertain_breakdowns = uncertain_breakdowns if uncertain_breakdowns is not None else {}
        
        # Offsets as inputs
        self.offset_baseline = offset_baseline
        self.offset_id_number = offset_id_number
        self.offset_delayed_flight = offset_delayed_flight
        self.offset_marker_minutes = offset_marker_minutes
        self.plot_title = plot_title
        # Define the sorting key function
        def extract_sort_key(aircraft_id):
            letters = ''.join(re.findall(r'[A-Za-z]+', aircraft_id))
            numbers = tuple(int(num) for num in re.findall(r'\d+', aircraft_id))
            return (letters, ) + numbers
        # Sort aircraft IDs using the custom sort key
        sorted_aircraft_ids = sorted(self.aircraft_dict.keys(), key=extract_sort_key)
        self.aircraft_id_to_idx = {aircraft_id: idx + 1 for idx, aircraft_id in enumerate(sorted_aircraft_ids)}
        # Calculate the earliest and latest datetimes
        self.earliest_datetime = min(parse_time_with_day_offset(flight_info['DepTime'], start_datetime) for flight_info in flights_dict.values())
        self.latest_datetime = max(parse_time_with_day_offset(flight_info['ArrTime'], start_datetime) for flight_info in flights_dict.values())
    def plot_state(self, flights_dict, swapped_flights, environment_delayed_flights, cancelled_flights, current_datetime, title_appendix="", debug_print=False, show_plot=True, reward_and_action=None, legend=True):
        if debug_print:
            print(f"Plotting state with following flights: {flights_dict}")
        updated_rotations_dict = self.rotations_dict.copy()
        for swap in swapped_flights:
            flight_id, new_aircraft_id = swap
            updated_rotations_dict[flight_id]['Aircraft'] = new_aircraft_id
        # Define the sorting key function
        def extract_sort_key(aircraft_id):
            letters = ''.join(re.findall(r'[A-Za-z]+', aircraft_id))
            numbers = tuple(int(num) for num in re.findall(r'\d+', aircraft_id))
            return (letters, ) + numbers
        # Collect and sort aircraft IDs using the custom sort key
        all_aircraft_ids = set([rotation_info['Aircraft'] for rotation_info in updated_rotations_dict.values()]).union(set(self.aircraft_dict.keys()))
        aircraft_ids = sorted(all_aircraft_ids, key=extract_sort_key)
        aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(aircraft_ids)}
        fig, ax = plt.subplots(figsize=(6, 2))
        labels = {
            'Scheduled Flight': False,
            'Swapped Flight': False,
            'Delayed Flight': False,
            'Cancelled Flight': False,
            'Certain Breakdown': False,
            'Disruption Start': False,
            'Disruption End': False,
            'Delay of Flight': False,
            'Uncertain Breakdown': False,
            'Zero Probability': False,
            'Current Action Flight': False
        }
        # Hard code earliest and latest times
        earliest_time = self.start_datetime.replace(hour=7, minute=30)
        latest_time = self.start_datetime.replace(hour=20, minute=0)
        for rotation_id, rotation_info in updated_rotations_dict.items():
            flight_id = rotation_id
            aircraft_id = rotation_info['Aircraft']
            
            if flight_id in flights_dict:
                flight_info = flights_dict[flight_id]
                dep_datetime_str = flight_info['DepTime']
                arr_datetime_str = flight_info['ArrTime']
                
                dep_datetime = parse_time_with_day_offset(dep_datetime_str, self.start_datetime)
                arr_datetime = parse_time_with_day_offset(arr_datetime_str, dep_datetime)
                
                # Ensure arrival time is on same day as departure for flights crossing midnight
                if '+1' in dep_datetime_str:
                    # If departure is next day, arrival should be same day
                    arr_datetime = arr_datetime.replace(day=dep_datetime.day)
                
                swapped = any(flight_id == swap[0] for swap in swapped_flights)
                delayed = flight_id in environment_delayed_flights
                cancelled = flight_id in cancelled_flights
                
                # Add check for current action flight
                is_current_action = reward_and_action and flight_id == reward_and_action[1][0]
                
                if cancelled:
                    plot_color = 'red'
                    plot_label = 'Cancelled Flight'
                elif is_current_action:
                    plot_color = 'green'
                    plot_label = 'Current Action Flight'
                elif swapped:
                    plot_color = 'green'
                    plot_label = 'Swapped Flight'
                elif delayed:
                    plot_color = 'orange'
                    plot_label = 'Delayed Flight'
                else:
                    plot_color = 'blue'
                    plot_label = 'Scheduled Flight'
                
                y_offset = aircraft_indices[aircraft_id]
                if delayed:
                    y_offset += self.offset_delayed_flight
                # Ensure arrival time is on same day as departure for flights crossing midnight
                if '+1' in dep_datetime_str:
                    # If departure is next day, arrival should be same day
                    arr_datetime = arr_datetime.replace(day=dep_datetime.day)
                ax.plot([dep_datetime, arr_datetime], [y_offset, y_offset], color=plot_color, label=plot_label if not labels[plot_label] else None)
                
                marker_offset = timedelta(minutes=self.offset_marker_minutes)
                dep_marker = dep_datetime + marker_offset
                arr_marker = arr_datetime - marker_offset
                ax.plot(dep_marker, y_offset, color=plot_color, marker='>', markersize=6, markeredgewidth=0)
                ax.plot(arr_marker, y_offset, color=plot_color, marker='<', markersize=6, markeredgewidth=0)
                if delayed:
                    ax.vlines([dep_datetime, arr_datetime], y_offset - self.offset_delayed_flight, y_offset, color='orange', linestyle='-', linewidth=2)
                labels[plot_label] = True
                
                mid_datetime = dep_datetime + (arr_datetime - dep_datetime) / 2
                ax.text(mid_datetime, y_offset + self.offset_id_number, flight_id, 
                        ha='center', va='bottom', fontsize=8, color='black')
        # Function to compute the data height that corresponds to a given number of pixels
        def get_height_in_data_units(ax, pixels):
            # Transform data coordinates (0, y) to display coordinates (pixels)
            y0_display = ax.transData.transform((0, 0))[1]
            y1_display = ax.transData.transform((0, 1))[1]
            pixels_per_data_unit = abs(y1_display - y0_display)
            data_units_per_pixel = 1 / pixels_per_data_unit
            return data_units_per_pixel * pixels
        # Compute rectangle height in data units corresponding to 50 pixels (reduced from 100)
        rect_height = get_height_in_data_units(ax, 50)
        # Handle alt_aircraft_dict unavailabilities, including uncertain ones with probability < 1.0
        if self.alt_aircraft_dict:
            for aircraft_id, unavailability_info in self.alt_aircraft_dict.items():
                if not isinstance(unavailability_info, list):
                    unavailability_info = [unavailability_info]
                
                for unavail in unavailability_info:
                    start_date = unavail['StartDate']
                    start_time = unavail['StartTime']
                    end_date = unavail['EndDate']
                    end_time = unavail['EndTime']
                    probability = unavail.get('Probability', 1.0)  # Default to 1.0 if Probability is not given
                    
                    # print("In state plotter:")
                    # print(f"    aircraft_id: {aircraft_id}")
                    # print(f"    start_time: {start_time}")
                    # print(f"    end_time: {end_time}")
                    # print(f"    probability: {probability}")
                    # Convert to datetime objects
                    unavail_start = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
                    unavail_end = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
                    y_offset = aircraft_indices[aircraft_id]
                    if np.isnan(probability):
                        probability = 0.0
                    
                    # Set color based on probability
                    if probability == 0.0:
                        rect_color = 'lightgrey'  # Very light grey for zero probability
                        plot_label = 'Zero Probability'
                    elif probability < 1.0:
                        rect_color = 'orange'  # Uncertain breakdown
                        plot_label = 'Uncertain Breakdown'
                    else:
                        rect_color = 'red'  # Certain unavailability
                        plot_label = 'Certain Breakdown'
                    
                    # Plot the unavailability period as a rectangle
                    rect = patches.Rectangle((unavail_start, y_offset - rect_height / 2),
                                            unavail_end - unavail_start,
                                            rect_height,
                                            linewidth=0,
                                            color=rect_color,
                                            alpha=0.3,
                                            zorder=0,
                                            label=plot_label if not labels[plot_label] else None)
                    ax.add_patch(rect)
                    labels[plot_label] = True
                    # Plot the probability below the rectangle
                    x_position = unavail_start + (unavail_end - unavail_start) / 2
                    y_position = y_offset - rect_height / 2 - get_height_in_data_units(ax, -10) + 0.2  # Increased offset from 10 to 20 pixels
                    ax.text(x_position, y_position + 0.1, f"{probability:.2f}", ha='center', va='top', fontsize=8)
        # Add transparent rectangles for each aircraft at start of recovery period
        for aircraft_id in all_aircraft_ids:
            y_offset = aircraft_indices[aircraft_id]
            rect_start = self.start_datetime - timedelta(hours=1)
            rect = patches.Rectangle(
                (rect_start, y_offset - rect_height / 2),
                timedelta(minutes=1),  # Very thin rectangle
                rect_height,
                linewidth=0,
                color='white',  # Color doesn't matter since alpha is 0
                alpha=0,  # Fully transparent
                zorder=0
            )
            ax.add_patch(rect)
        # Set x-axis limits to hard-coded times
        ax.set_xlim(earliest_time, latest_time)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax.axvline(self.start_datetime, color='green', linestyle='--', label='Start Recovery Period')
        # ax.axvline(self.end_datetime, color='green', linestyle='-', label='End Recovery Period')
        ax.axvline(current_datetime, color='black', linestyle='-', label='Current Time')
        ax.axvspan(self.end_datetime, latest_time, color='lightgrey', alpha=0.3)
        ax.axvspan(earliest_time, self.start_datetime, color='lightgrey', alpha=0.3)
        
        # Add extra padding under the last row
        y_min = 0.5  # Start from 0.5 to give padding below the last row
        y_max = len(aircraft_ids) + 0.5  # Add 0.5 for padding above first row
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()
        ytick_labels = [f"{index + 1}: {aircraft_id}" for index, aircraft_id in enumerate(aircraft_ids)]
        plt.yticks(range(1, len(aircraft_ids) + 1), ytick_labels, fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Aircraft')
        if self.plot_title:
            title = f"Aircraft rotations and unavailability {self.plot_title}"  
        else:
            title = f'Aircraft Rotations and Unavailability {title_appendix.title()}'
        if reward_and_action:
            title += f' (Reward: {reward_and_action[0]:.2f}, Action: (flight: {reward_and_action[1][0]}, aircraft: {reward_and_action[1][1]}), Total Reward: {reward_and_action[2]:.2f})'
        # plt.title(title)
        plt.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.tight_layout()
        if legend:
            plt.legend(bbox_to_anchor=(0.5, -0.5), loc='upper center', ncol=3)
        if show_plot:
            plt.show()
        else:
            plt.close()
            return fig


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
    all_aircraft_ids = sorted(list(set([rotation_info['Aircraft'] for rotation_info in rotations_dict.values()] + list(alt_aircraft_dict.keys()))))
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

            # Fix for flights that depart and arrive after midnight
            if dep_datetime.time() > datetime.strptime('00:00', '%H:%M').time() and arr_datetime.time() > datetime.strptime('00:00', '%H:%M').time():
                if arr_datetime.date() > dep_datetime.date():
                    arr_datetime -= timedelta(days=1)

            if arr_datetime < dep_datetime:
                arr_datetime += timedelta(days=1)

            # Standard flight plot
            plot_color = 'blue'
            plot_label = 'Scheduled Flight'

            y_offset = aircraft_indices[aircraft_id]

            # Special handling for first flight (ID 1) and last aircraft
            if flight_id == 1 or aircraft_id == all_aircraft_ids[-1]:
                ax.plot([dep_datetime, arr_datetime], [y_offset, y_offset], color='blue', alpha=0, label=plot_label if not labels['Scheduled Flight'] else "")
            else:
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

    # Ensure all aircraft are included, even those without flights or unavailability
    for aircraft_id in all_aircraft_ids:
        if aircraft_id not in rotations_dict and aircraft_id not in alt_aircraft_dict:
            y_offset = aircraft_indices[aircraft_id]
            # Plot an empty row for the aircraft
            # ax.plot([], [], label=f'{aircraft_id} (No Flights/Unavailability)', color='gray')

    # Plot recovery period
    ax.axvline(start_datetime, color='green', linestyle='--', linewidth=2, label='Start/End Recovery Period')
    ax.axvline(end_datetime, color='green', linestyle='--', linewidth=2)  # Same style, no label

    # Grey out periods outside recovery time
    ax.axvspan(end_datetime, latest_datetime + timedelta(hours=1), color='lightgrey', alpha=0.3)
    ax.axvspan(earliest_datetime - timedelta(hours=1), start_datetime, color='lightgrey', alpha=0.3)

    # Formatting the plot
    ax.set_xlim(earliest_datetime - timedelta(hours=1), latest_datetime + timedelta(hours=1))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Determine the padding for the y-axis
    y_min = 0.2  # Add a bit of space below the first aircraft
    y_max = len(all_aircraft_ids) + 0.2  # Add a bit of space above the last aircraft

    # Set the y-limits with padding
    ax.set_ylim(y_max, y_min)

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
    ax.axvline(start_datetime, color='green', linestyle='--', label='Start/End Recovery Period')
    ax.axvline(end_datetime, color='green', linestyle='--')  # Same style, no label

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





import os
import numpy as np
import matplotlib.pyplot as plt
from src.environment import AircraftDisruptionEnv
import time

def calculate_total_training_timesteps(training_folders_path, n_episodes):
    """
    Calculates the total estimated timesteps required for training based on the scenarios and episodes.
    Times one batch to calculate average time per timestep, scenario, and estimate total training time.

    Args:
        training_folders_path (str): Path to the folder containing training scenarios.
        n_episodes (int): Number of episodes for training.

    Returns:
        int: Estimated total timesteps for the entire training process.
    """
    total_timesteps_per_batch = 0
    scenario_count = 0
    start_time = time.time()  # Start timer for the batch

    # List all the scenario folders in Data/Training
    scenario_folders = [
        os.path.join(training_folders_path, folder)
        for folder in os.listdir(training_folders_path)
        if os.path.isdir(os.path.join(training_folders_path, folder))
    ]

    # print(f"amount of scenario folders:{len(scenario_folders)}")
    # Simulate one batch with a random agent to calculate timesteps per scenario
    for scenario_folder in scenario_folders:
        scenario_count += 1

        # Load the data for the current scenario
        data_dict = load_scenario_data(scenario_folder)
        aircraft_dict = data_dict['aircraft']
        flights_dict = data_dict['flights']
        rotations_dict = data_dict['rotations']
        alt_aircraft_dict = data_dict['alt_aircraft']
        config_dict = data_dict['config']
        env_type = 'myopic'

        # Initialize the environment with the new scenario
        env = AircraftDisruptionEnv(
            aircraft_dict,
            flights_dict,
            rotations_dict,
            alt_aircraft_dict,
            config_dict,
            env_type
        )

        # Reset the environment
        obs, _ = env.reset()
        done = False
        timesteps = 0

        while not done:
            # Random action from valid actions
            action_mask = obs['action_mask']
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            timesteps += 1
            done = terminated or truncated

        # Accumulate timesteps
        total_timesteps_per_batch += timesteps

    # End timer for the batch
    end_time = time.time()
    batch_time = end_time - start_time  # Time taken for the entire batch

    # Calculate averages
    average_time_per_timestep = batch_time / total_timesteps_per_batch
    average_time_per_scenario = batch_time / scenario_count
    total_timesteps_estimate = total_timesteps_per_batch * n_episodes
    estimated_training_time = average_time_per_timestep * total_timesteps_estimate

    # Print timing information
    # print(f"Estimated Total Training Time: {estimated_training_time / 3600:.2f} hours")
    # print(f"Estimated Total Training Time: {estimated_training_time / 60:.2f} minutes")
    # print(f"    Batch Time: {batch_time:.2f} seconds")
    # print(f"    Average Time Per Timestep: {average_time_per_timestep:.6f} seconds")
    # print(f"    Average Time Per Scenario: {average_time_per_scenario:.2f} seconds")
    # print("")
    # print(f"Total Timesteps Estimate: {total_timesteps_estimate}")
    return total_timesteps_estimate



def simulate_and_plot_epsilon_decay(epsilon_start, epsilon_min, epsilon_decay_rate, estimated_total_timesteps, EPSILON_TYPE):
    """
    Simulates a batch of scenarios using a random agent, estimates total timesteps for training,
    generates epsilon decay values, and plots the decay curve.
    """
    
    total_timesteps_estimate = estimated_total_timesteps

    # Generate epsilon values over the estimated total timesteps
    epsilon_values_estimate = []
    epsilon = epsilon_start
    min_epsilon_reached_at = 0

    for t in range(int(total_timesteps_estimate)):
        if EPSILON_TYPE == "exponential":
            epsilon = max(epsilon_min, epsilon * (1 - epsilon_decay_rate))
        elif EPSILON_TYPE == "linear":
            epsilon = max(epsilon_min, epsilon - epsilon_decay_rate)
        epsilon_values_estimate.append(epsilon)

        # Record when epsilon reaches the minimum value
        if epsilon == epsilon_min:
            min_epsilon_reached_at = t
            # Extend the list with EPSILON_MIN for the remaining timesteps
            epsilon_values_estimate.extend([epsilon_min] * (int(total_timesteps_estimate) - t - 1))
            break

    # Calculate the percentage of timesteps where epsilon reaches its minimum value
    percentage_min_epsilon_reached = (min_epsilon_reached_at / total_timesteps_estimate) * 100
    print(f"Epsilon reaches its minimum value at {percentage_min_epsilon_reached:.2f}% of total timesteps.")

    # Plot the estimated epsilon decay curve
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values_estimate)
    plt.xlabel('Timesteps')
    plt.ylabel('Epsilon Value')
    plt.title('Estimated Epsilon Decay over Timesteps')
    plt.show()
