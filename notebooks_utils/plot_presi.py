import sys
sys.path.append("..")
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scripts.utils import *
from scripts.visualizations import *
from src.config import *
from datetime import datetime, timedelta
from src.environment import AircraftDisruptionEnv
from scripts.visualizations import StatePlotter
from src.config import TIMESTEP_HOURS, DEBUG_MODE_PRINT_STATE
import time
import random

env_type = "proactive"

# Function to get next plot number in a folder
def get_next_plot_number(folder_path):
    existing_plots = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    if not existing_plots:
        return 1
    plot_numbers = [int(f.split('.')[0]) for f in existing_plots]
    return max(plot_numbers) + 1

# Function to save plot
def save_plot(folder_path):
    plot_number = get_next_plot_number(folder_path)
    plt.savefig(os.path.join(folder_path, f"{plot_number}.png"), bbox_inches='tight')
    plt.close()

# Function to handle user input for action
def get_user_action(valid_actions, env):
    print("\nAvailable Actions:")
    while True:
        try:
            user_input = input("Enter the flight index from the available flights: ").strip()
            flight_index = int(user_input)

            user_input = input("Enter the aircraft index from the available aircrafts: ").strip()
            aircraft_index = int(user_input)

            action_index = env.map_action_to_index(flight_index, aircraft_index)

            if action_index in valid_actions:
                return action_index
            else:
                print("Invalid action index. Please select from the valid actions.")
                print(f"available actions:")
                for action_index in valid_actions:
                    flight, aircraft = env.map_index_to_action(action_index)
                    print(f"Index {action_index}: Flight {flight}, Aircraft {aircraft}")
        except ValueError:
            print("Invalid input. Please enter a number corresponding to the action index.")

# Run the agent with user input
def run_user_agent(scenario_folder):
    # Create unique plot folder
    plot_id = ''.join([str(random.randint(0,9)) for _ in range(4)])
    plot_folder = f"plt_{plot_id}"
    os.makedirs(plot_folder, exist_ok=True)

    # Set a random seed based on the current second in time
    current_seed = int(time.time() * 1e9) % (2**32 - 1)
    print(current_seed)
    np.random.seed(current_seed)

    # Load the scenario data
    data_dict = load_scenario_data(scenario_folder)

    # Extract necessary data for the environment
    aircraft_dict = data_dict['aircraft']
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_aircraft_dict = data_dict['alt_aircraft']
    config_dict = data_dict['config']

    # Initialize the environment
    env = AircraftDisruptionEnv(
        aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type=env_type
    )

    print(env.current_datetime)

    # Reset the environment
    obs, info = env.reset()
    if DEBUG_MODE_VISUALIZATION:
        print("Observation keys:", obs.keys())

    done = False
    total_reward = 0
    step_num = 0

    # Create StatePlotter object for visualizing the environment state
    state_plotter = StatePlotter(
        aircraft_dict=env.aircraft_dict,
        flights_dict=env.flights_dict,
        rotations_dict=env.rotations_dict,
        alt_aircraft_dict=env.alt_aircraft_dict,
        start_datetime=env.start_datetime,
        end_datetime=env.end_datetime,
        uncertain_breakdowns=env.uncertain_breakdowns,
        last_flight_action=None
    )

    # Print initial state
    print("Initial State:")
    print_state_nicely(env.state)

    # Plot initial scenario with different settings
    print("\nShowing empty scenario without recovery period and current time:")
    fig = state_plotter.plot_state(
        env.flights_dict, env.swapped_flights, env.environment_delayed_flights,
        env.penalized_cancelled_flights, env.current_datetime,
        show_flights=False, show_certain_disruptions=False, show_uncertain_disruptions=False,
        show_recovery_window=False, show_current_time=False
    )
    plt.draw()
    plt.pause(0.1)
    save_plot(plot_folder)

    print("\nShowing empty scenario with recovery window only:")
    fig = state_plotter.plot_state(
        env.flights_dict, env.swapped_flights, env.environment_delayed_flights,
        env.penalized_cancelled_flights, env.current_datetime,
        show_flights=False, show_certain_disruptions=False, show_uncertain_disruptions=False,
        show_recovery_window=True, show_current_time=False
    )
    plt.draw()
    plt.pause(0.1)
    save_plot(plot_folder)

    print("\nShowing empty scenario:")
    fig = state_plotter.plot_state(
        env.flights_dict, env.swapped_flights, env.environment_delayed_flights,
        env.penalized_cancelled_flights, env.current_datetime,
        show_flights=False, show_certain_disruptions=False, show_uncertain_disruptions=False
    )
    plt.draw()
    plt.pause(0.1)
    save_plot(plot_folder)

    print("\nShowing only flights:")
    fig = state_plotter.plot_state(
        env.flights_dict, env.swapped_flights, env.environment_delayed_flights, 
        env.penalized_cancelled_flights, env.current_datetime,
        show_flights=True, show_certain_disruptions=False, show_uncertain_disruptions=False
    )
    plt.draw()
    plt.pause(0.1)
    save_plot(plot_folder)

    print("\nShowing flights and certain disruptions:")
    fig = state_plotter.plot_state(
        env.flights_dict, env.swapped_flights, env.environment_delayed_flights,
        env.penalized_cancelled_flights, env.current_datetime,
        show_flights=True, show_certain_disruptions=True, show_uncertain_disruptions=False
    )
    plt.draw()
    plt.pause(0.1)
    save_plot(plot_folder)

    while not done:
        # Visualize the environment at each step
        print(f"\nStep {step_num}:")

        # Extract necessary information for plotting
        swapped_flights = env.swapped_flights
        environment_delayed_flights = env.environment_delayed_flights
        current_datetime = env.current_datetime 

        updated_flights_dict = env.flights_dict
        updated_rotations_dict = env.rotations_dict
        updated_alt_aircraft_dict = env.alt_aircraft_dict
        cancelled_flights = env.penalized_cancelled_flights

        # Update the StatePlotter's dictionaries
        state_plotter.alt_aircraft_dict = updated_alt_aircraft_dict
        state_plotter.flights_dict = updated_flights_dict
        state_plotter.rotations_dict = updated_rotations_dict

        if step_num == 0:
            last_flight_action = None
        else:
            last_flight_action = env.last_flight_action

        # Plot the state
        fig = state_plotter.plot_state(
            updated_flights_dict, swapped_flights, environment_delayed_flights, cancelled_flights, current_datetime, last_flight_action=last_flight_action
        )
        plt.draw()
        plt.pause(0.1)
        save_plot(plot_folder)

        # Get the action mask
        action_mask = obs['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            print("No valid actions available. Terminating...")
            break

        # Get user input for the action
        action_index = get_user_action(valid_actions, env)

        # Map the action index to the actual action
        action = env.map_index_to_action(action_index)
        print(f"Action chosen: Flight {action[0]}, Aircraft {action[1]}")

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action_index)
        done = terminated or truncated

        env.last_flight_action = action
        print("last_flight_action: ", env.last_flight_action)

        # Accumulate reward
        total_reward += reward

        # Print action and reward
        print(f"Action taken: Flight {action[0]}, Aircraft {action[1]}, Reward: {reward}")

        step_num += 1

    # Final plot after the simulation ends
    fig = state_plotter.plot_state(
        updated_flights_dict, swapped_flights, environment_delayed_flights, cancelled_flights,
        current_datetime + timedelta(hours=TIMESTEP_HOURS)
    )
    plt.draw()
    plt.pause(0.1)
    save_plot(plot_folder)

    if DEBUG_MODE_PRINT_STATE:
        print("Final State:")
        print_state_nicely(env.state, env_type)

    print(f"Total Reward: {total_reward}")

# Set the scenario folder
SCENARIO_FOLDER = "../data/Testing/6ac-700-diverse/mixed_high_Scenario_016"

# Verify folder exists
if not os.path.exists(SCENARIO_FOLDER):
    raise FileNotFoundError(f"Scenario folder not found: {SCENARIO_FOLDER}")

# Run the agent
run_user_agent(SCENARIO_FOLDER)
