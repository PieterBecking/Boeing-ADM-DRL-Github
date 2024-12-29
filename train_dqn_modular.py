import os
import sys
import warnings
import datetime
import math
import numpy as np
import pandas as pd
import platform
import re
import subprocess
import torch as th
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from scripts.utils import *
from scripts.visualizations import *
from src.config import *
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import polyak_update, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from scripts.utils import NumpyEncoder
from scripts.logger import *
import json
import seaborn as sns
sns.set(style="darkgrid")
import time
import subprocess



def run_train_dqn_both_timesteps(
    MAX_TOTAL_TIMESTEPS,
    single_seed,
    brute_force_flag,
    cross_val_flag,
    early_stopping_flag,
    CROSS_VAL_INTERVAL,
    printing_intermediate_results,
    TRAINING_FOLDERS_PATH,
    stripped_scenario_folder,
    save_folder,
    save_results_big_run,
    TESTING_FOLDERS_PATH,
    env_type
):

    print(f"Training on {stripped_scenario_folder} with {env_type} environment")
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Constants and Training Settings
    LEARNING_RATE = 0.0001
    GAMMA = 0.9999
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    TARGET_UPDATE_INTERVAL = 100
    NEURAL_NET_STRUCTURE = dict(net_arch=[256, 256*2, 256])
    LEARNING_STARTS = 10000
    TRAIN_FREQ = 4

    EPSILON_START = 1.0
    EPSILON_MIN = 0.025
    PERCENTAGE_MIN = 85
    EPSILON_TYPE = "exponential"
    if EPSILON_TYPE == "linear":
        EPSILON_MIN = 0

    N_EPISODES = 50 # DOESNT MATTER

    starting_time = time.time()

    # extract number of scenarios in training and testing folders
    num_scenarios_training = len(os.listdir(TRAINING_FOLDERS_PATH))

    # Based on parameters, calculate EPSILON_DECAY_RATE as in original code
    EPSILON_DECAY_RATE = calculate_epsilon_decay_rate(
        MAX_TOTAL_TIMESTEPS, EPSILON_START, EPSILON_MIN, PERCENTAGE_MIN, EPSILON_TYPE
    )
    print("EPSILON DECAY RATE: ", EPSILON_DECAY_RATE)

    # Initialize device
    device = initialize_device()

    # Check device capabilities
    check_device_capabilities()

    # Get device-specific information
    device_info = get_device_info(device)
    print(f"Device info: {device_info}")

    # Verify training folders and gather training data
    training_folders = verify_training_folders(TRAINING_FOLDERS_PATH)

    # Calculate training days and model naming
    num_days_trained_on = calculate_training_days(N_EPISODES, training_folders)
    print(f"Training on {num_days_trained_on} days of data "
          f"({N_EPISODES} episodes of {len(training_folders)} scenarios)")

    formatted_days = format_days(num_days_trained_on)
    MODEL_SAVE_PATH = f'../trained_models/dqn/'

    # Create results directory
    results_dir = create_results_directory(append_to_name='dqn')
    print(f"Results directory created at: {results_dir}")

    from scripts.logger import create_new_id, get_config_variables
    import src.config as config

    all_logs = {}
    def train_dqn_agent(env_type, seed):
        log_data = {}  # Main dictionary to store all logs

        config_variables = get_config_variables(config)

        # Generate unique ID for training
        training_id = create_new_id("training").split("_")[1]
        runtime_start_in_seconds = time.time()

        # Construct model_path with required directory structure
        model_save_dir = f"{save_folder}/{stripped_scenario_folder}"
        os.makedirs(model_save_dir, exist_ok=True)

        model_path = f"{model_save_dir}/{env_type}_{single_seed}.zip"

        print(f"Models will be saved to: {model_path}")

        training_metadata = {
            "myopic_or_proactive": env_type,
            "model_type": "dqn",
            "training_id": training_id,
            "MODEL_SAVE_PATH": model_path,
            "N_EPISODES": N_EPISODES,
            "num_scenarios_training": num_scenarios_training,
            "results_dir": results_dir,
            "CROSS_VAL_FLAG": cross_val_flag,
            "CROSS_VAL_INTERVAL": CROSS_VAL_INTERVAL,
            **config_variables,
            "LEARNING_RATE": LEARNING_RATE,
            "GAMMA": GAMMA,
            "BUFFER_SIZE": BUFFER_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "TARGET_UPDATE_INTERVAL": TARGET_UPDATE_INTERVAL,
            "EPSILON_START": EPSILON_START,
            "EPSILON_MIN": EPSILON_MIN,
            "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE,
            "LEARNING_STARTS": LEARNING_STARTS,
            "TRAIN_FREQ": TRAIN_FREQ,
            "NEURAL_NET_STRUCTURE": NEURAL_NET_STRUCTURE,
            "device_info": str(get_device_info(device)),
            "TRAINING_FOLDERS_PATH": TRAINING_FOLDERS_PATH,
            "TESTING_FOLDERS_PATH": TESTING_FOLDERS_PATH,
            "runtime_start": datetime.utcnow().isoformat() + "Z",
            "runtime_start_in_seconds": runtime_start_in_seconds,
        }

        log_data['metadata'] = {}
        log_data['episodes'] = {}
        log_data['cross_validation'] = {}

        best_reward_avg = float('-inf')
        # Initialize variables
        rewards = {}
        good_rewards = {}
        test_rewards = []
        epsilon_values = []
        total_timesteps = 0  # Added to track total timesteps
        consecutive_drops = 0  # Track consecutive performance drops
        best_test_reward = float('-inf')  # Track best test performance

        def cross_validate_on_test_data(model, current_episode, log_data):
            cross_val_data = {
                "episode": current_episode,
                "scenarios": [],
                "avg_test_reward": 0,
            }

            test_scenario_folders = [
                os.path.join(TESTING_FOLDERS_PATH, folder)
                for folder in os.listdir(TESTING_FOLDERS_PATH)
                if os.path.isdir(os.path.join(TESTING_FOLDERS_PATH, folder))
            ]
            total_test_reward = 0
            for test_scenario_folder in test_scenario_folders:
                scenario_data = {
                    "scenario_folder": test_scenario_folder,
                    "total_reward": 0,
                }
                # Load data
                data_dict = load_scenario_data(test_scenario_folder)
                aircraft_dict = data_dict['aircraft']
                flights_dict = data_dict['flights']
                rotations_dict = data_dict['rotations']
                alt_aircraft_dict = data_dict['alt_aircraft']
                config_dict = data_dict['config']

                from src.environment import AircraftDisruptionEnv
                env = AircraftDisruptionEnv(
                    aircraft_dict,
                    flights_dict,
                    rotations_dict,
                    alt_aircraft_dict,
                    config_dict,
                    env_type=env_type
                )
                model.set_env(env)  # Update the model's environment with the new instance

                obs, _ = env.reset()

                done_flag = False
                total_reward_local = 0
                timesteps_local = 0

                while not done_flag:
                    # Get the action mask from the environment
                    action_mask = obs['action_mask']

                    # Convert observation to float32
                    obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}

                    # Preprocess observation and get Q-values
                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

                    # Apply the action mask (set invalid actions to -np.inf)
                    masked_q_values = q_values.copy()
                    masked_q_values[action_mask == 0] = -np.inf

                    # Select the action with the highest masked Q-value
                    action = np.argmax(masked_q_values)

                    # Take the selected action in the environment
                    result = env.step(action)

                    obs_next, reward, terminated, truncated, info = result

                    done_flag = terminated or truncated
                    total_reward_local += reward
                    obs = obs_next

                    timesteps_local += 1
                    if done_flag:
                        break

                total_test_reward += total_reward_local
                scenario_data["total_reward"] = total_reward_local
                cross_val_data["scenarios"].append(scenario_data)

            avg_test_reward = total_test_reward / len(test_scenario_folders)
            cross_val_data["avg_test_reward"] = avg_test_reward
            test_rewards.append(avg_test_reward)
            print(f"cross-val done at episode {current_episode}")

            log_data['cross_validation'] = {}

            return avg_test_reward

        scenario_folders = [
            os.path.join(TRAINING_FOLDERS_PATH, folder)
            for folder in os.listdir(TRAINING_FOLDERS_PATH)
            if os.path.isdir(os.path.join(TRAINING_FOLDERS_PATH, folder))
        ]

        epsilon = EPSILON_START
        total_timesteps = 0

        # Initialize the DQN
        dummy_scenario_folder = scenario_folders[0]
        data_dict = load_scenario_data(dummy_scenario_folder)
        aircraft_dict = data_dict['aircraft']
        flights_dict = data_dict['flights']
        rotations_dict = data_dict['rotations']
        alt_aircraft_dict = data_dict['alt_aircraft']
        config_dict = data_dict['config']

        from src.environment import AircraftDisruptionEnv
        env = AircraftDisruptionEnv(
            aircraft_dict,
            flights_dict,
            rotations_dict,
            alt_aircraft_dict,
            config_dict,
            env_type=env_type
        )

        model = DQN(
            policy='MultiInputPolicy',
            env=env,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            verbose=0,
            policy_kwargs=NEURAL_NET_STRUCTURE,
            device=device
        )

        logger = configure()
        model._logger = logger

        episode = 0
        while total_timesteps < MAX_TOTAL_TIMESTEPS:
            rewards[episode] = {}
            episode_data = {
                "episode_number": episode + 1,
                "epsilon_start": epsilon,
                "scenarios": {},
            }

            for scenario_folder in scenario_folders:
                scenario_data = {
                    "scenario_folder": scenario_folder,
                    "total_reward": 0,
                }
                rewards[episode][scenario_folder] = {}
                best_reward_local = float('-inf')
                data_dict = load_scenario_data(scenario_folder)
                aircraft_dict = data_dict['aircraft']
                flights_dict = data_dict['flights']
                rotations_dict = data_dict['rotations']
                alt_aircraft_dict = data_dict['alt_aircraft']
                config_dict = data_dict['config']

                env = AircraftDisruptionEnv(
                    aircraft_dict,
                    flights_dict,
                    rotations_dict,
                    alt_aircraft_dict,
                    config_dict,
                    env_type=env_type
                )
                model.set_env(env)

                obs, _ = env.reset()
                done_flag = False
                total_reward_local = 0
                timesteps_local = 0

                while not done_flag:
                    num_cancelled_flights_before_step = len(env.cancelled_flights)
                    num_delayed_flights_before_step = len(env.environment_delayed_flights)
                    num_penalized_delays_before_step = len(env.penalized_delays)
                    num_penalized_cancelled_before_step = len(env.penalized_cancelled_flights)

                    model.exploration_rate = epsilon

                    action_mask = obs['action_mask']
                    obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}

                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

                    masked_q_values = q_values.copy()
                    masked_q_values[action_mask == 0] = -np.inf

                    current_seed = int(time.time() * 1e9) % (2**32 - 1)
                    np.random.seed(current_seed)

                    action_reason = "None"
                    if env_type == "drl-greedy":
                        if np.random.rand() < epsilon or brute_force_flag:
                            # During exploration (50% greedy, 50% random)
                            if np.random.rand() < 0.5:
                                # Use the greedy optimizer to select action
                                from src.environment import AircraftDisruptionOptimizer
                                # Create a copy of the current environment state for the optimizer
                                optimizer = AircraftDisruptionOptimizer(
                                    aircraft_dict=env.aircraft_dict,
                                    flights_dict=env.flights_dict,
                                    rotations_dict=env.rotations_dict,
                                    alt_aircraft_dict=env.alt_aircraft_dict,
                                    config_dict=env.config_dict
                                )
                                # Set the optimizer's state to match current environment
                                optimizer.current_datetime = env.current_datetime
                                optimizer.state = env.state.copy()
                                optimizer.unavailabilities_dict = env.unavailabilities_dict.copy()
                                optimizer.cancelled_flights = env.cancelled_flights.copy()
                                optimizer.environment_delayed_flights = env.environment_delayed_flights.copy()
                                optimizer.penalized_delays = env.penalized_delays.copy()
                                optimizer.penalized_cancelled_flights = env.penalized_cancelled_flights.copy()
                                optimizer.initial_conflict_combinations = env.initial_conflict_combinations
                                optimizer.eligible_flights_for_resolved_bonus = env.eligible_flights_for_resolved_bonus
                                optimizer.eligible_flights_for_not_being_cancelled_when_disruption_happens = env.eligible_flights_for_not_being_cancelled_when_disruption_happens
                                optimizer.scenario_wide_initial_disrupted_flights_list = env.scenario_wide_initial_disrupted_flights_list
                                optimizer.scenario_wide_actual_disrupted_flights = env.scenario_wide_actual_disrupted_flights
                                optimizer.something_happened = False
                                optimizer.tail_swap_happened = False
                                optimizer.scenario_wide_reward_components = env.scenario_wide_reward_components.copy()
                                optimizer.scenario_wide_delay_minutes = env.scenario_wide_delay_minutes
                                optimizer.scenario_wide_cancelled_flights = env.scenario_wide_cancelled_flights
                                optimizer.scenario_wide_steps = env.scenario_wide_steps
                                optimizer.scenario_wide_resolved_conflicts = env.scenario_wide_resolved_conflicts
                                optimizer.scenario_wide_solution_slack = env.scenario_wide_solution_slack
                                optimizer.scenario_wide_tail_swaps = env.scenario_wide_tail_swaps
                                optimizer.info_after_step = {}

                                
                                # Get the best action from the optimizer and convert to numpy array
                                action = optimizer.select_best_action()
                                action = np.array(action).reshape(1, -1)
                                action_reason = "greedy-optimizer"
                            else:
                                # Random exploration
                                valid_actions = np.where(action_mask == 1)[0]
                                action = np.random.choice(valid_actions)
                                action = np.array(action).reshape(1, -1)
                                action_reason = "exploration"
                        else:
                            # Exploitation: always use Q-values
                            action = np.argmax(masked_q_values)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploitation"
                    else:
                        # For other environment types, use standard DQN logic
                        if np.random.rand() < epsilon or brute_force_flag:
                            valid_actions = np.where(action_mask == 1)[0]
                            action = np.random.choice(valid_actions)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploration"
                        else:
                            action = np.argmax(masked_q_values)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploitation"

                    result = env.step(action.item())  # Convert back to scalar for the environment
                    obs_next, reward, terminated, truncated, info = result

                    rewards[episode][scenario_folder][timesteps_local] = reward

                    done_flag = terminated or truncated

                    model.replay_buffer.add(
                        obs=obs,
                        next_obs=obs_next,
                        action=action,  # Now action is already in the correct format
                        reward=reward,
                        done=done_flag,
                        infos=[info]
                    )

                    obs = obs_next

                    epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE))
                    epsilon_values.append((episode + 1, epsilon))

                    timesteps_local += 1
                    total_timesteps += 1

                    if total_timesteps > model.learning_starts and total_timesteps % TRAIN_FREQ == 0:
                        model.train(gradient_steps=1, batch_size=BATCH_SIZE)

                    if total_timesteps % model.target_update_interval == 0:
                        polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)
                        polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)

                    num_cancelled_flights_after_step = len(env.cancelled_flights)
                    num_delayed_flights_after_step = len(env.environment_delayed_flights)
                    num_penalized_delays_after_step = len(env.penalized_delays)
                    num_penalized_cancelled_after_step = len(env.penalized_cancelled_flights)

                    impact_of_action = {
                        "num_cancelled_flights": num_cancelled_flights_after_step - num_cancelled_flights_before_step,
                        "num_delayed_flights": num_delayed_flights_after_step - num_delayed_flights_before_step,
                        "num_penalized_delays": num_penalized_delays_after_step - num_penalized_delays_before_step,
                        "num_penalized_cancelled": num_penalized_cancelled_after_step - num_penalized_cancelled_before_step,
                    }

                    if done_flag:
                        break

                total_reward_local = sum(rewards[episode][scenario_folder].values())
                rewards[episode][scenario_folder]["total"] = total_reward_local

                scenario_data["total_reward"] = total_reward_local
                episode_data["scenarios"][scenario_folder] = scenario_data

            # Perform cross-validation if enabled
            if cross_val_flag:
                if (episode + 1) % CROSS_VAL_INTERVAL == 0:
                    current_test_reward = cross_validate_on_test_data(model, episode + 1, log_data)
                    if not hasattr(train_dqn_agent, 'best_test_reward'):
                        train_dqn_agent.best_test_reward = current_test_reward
                    best_test_reward_local = train_dqn_agent.best_test_reward

                    # Early stopping logic
                    if current_test_reward < best_test_reward_local:
                        consecutive_drops += 1
                        print(f"Performance drop {consecutive_drops}/5 (current: {current_test_reward:.2f}, best: {best_test_reward_local:.2f})")
                        if consecutive_drops >= 500:
                            print(f"Early stopping triggered at episode {episode + 1} due to 5 consecutive drops in test performance")
                            break
                    else:
                        consecutive_drops = 0
                        train_dqn_agent.best_test_reward = current_test_reward
                        best_test_reward_local = current_test_reward

            # Calculate the average reward for this batch of episodes
            avg_reward_for_this_batch = 0
            for i in range(len(scenario_folders)):
                avg_reward_for_this_batch += rewards[episode][scenario_folders[i]]["total"]
            avg_reward_for_this_batch /= len(scenario_folders)

            rewards[episode]["avg_reward"] = avg_reward_for_this_batch
            rewards[episode]["total_timesteps"] = total_timesteps

            current_time = time.time()
            elapsed_time = current_time - starting_time
            percentage_complete = (total_timesteps / MAX_TOTAL_TIMESTEPS) * 100

            estimated_time_remaining = (elapsed_time / percentage_complete) * (100 - percentage_complete)
            hours = int(estimated_time_remaining // 3600)
            minutes = int((estimated_time_remaining % 3600) // 60)

            # Calculate time per 1000 timesteps
            if episode > 0:
                time_this_episode = current_time - previous_episode_time
                timesteps_this_episode = total_timesteps - previous_timesteps
                time_per_10000 = (time_this_episode / timesteps_this_episode) * 10000
            else:
                time_per_10000 = 0
            
            rewards[episode]["timestamp"] = current_time
            time_remaining_str = f"{hours}h{minutes}m" if hours > 0 else f"{minutes}m"
            print(f"({total_timesteps:.0f}/{MAX_TOTAL_TIMESTEPS:.0f} - {percentage_complete:.0f}% - {time_remaining_str} remaining, {time_per_10000:.0f}s/10k steps) {env_type:<10} - episode {episode + 1} - epsilon {epsilon:.2f} - reward this episode: {avg_reward_for_this_batch:.2f}")

            previous_episode_time = current_time
            previous_timesteps = total_timesteps

            episode_data["avg_reward"] = avg_reward_for_this_batch
            log_data['episodes'] = {}
            episode += 1

        model.save(model_path)
        runtime_end_in_seconds = time.time()
        runtime_in_seconds = runtime_end_in_seconds - runtime_start_in_seconds
        actual_total_timesteps = total_timesteps

        # Return collected data
        return rewards, test_rewards, total_timesteps, epsilon_values, good_rewards, {}, model_path

    # Run training for the specified environment type only
    rewards, test_rewards, total_timesteps, epsilon_values, good_rewards, action_sequences, model_path = train_dqn_agent(env_type, single_seed)

    # Extract only the necessary data
    episode_rewards = [rewards[e]["avg_reward"] for e in sorted(rewards.keys()) if "avg_reward" in rewards[e]]
    episode_steps = [rewards[e]["total_timesteps"] for e in sorted(rewards.keys()) if "total_timesteps" in rewards[e]]

    # Save results for this environment type
    os.makedirs(f"{save_results_big_run}/numpy", exist_ok=True)
    np.save(f'{save_results_big_run}/numpy/{env_type}_runs_seed_{single_seed}.npy', np.array(episode_rewards))
    np.save(f'{save_results_big_run}/numpy/{env_type}_steps_runs_seed_{single_seed}.npy', np.array(episode_steps))
    if test_rewards:  # Only save if we have test rewards
        np.save(f'{save_results_big_run}/numpy/test_rewards_{env_type}_seed_{single_seed}.npy', test_rewards)

    # Return empty arrays for the other environment types
    empty_array = np.array([])
    if env_type == 'myopic':
        return episode_rewards, empty_array, empty_array, empty_array, test_rewards, [], [], []
    elif env_type == 'proactive':
        return empty_array, episode_rewards, empty_array, empty_array, [], test_rewards, [], []
    elif env_type == 'reactive':
        return empty_array, empty_array, episode_rewards, empty_array, [], [], test_rewards, []
    else:  # drl-greedy
        return empty_array, empty_array, empty_array, episode_rewards, [], [], [], test_rewards
