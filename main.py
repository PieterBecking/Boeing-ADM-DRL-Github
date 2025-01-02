import argparse
import os
import time
import subprocess
import src.config as config
import pandas as pd
from train_dqn_modular import run_train_dqn_both_timesteps
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_config_variables(config_module):
    config_vars = {
        key: value for key, value in vars(config_module).items()
        if not key.startswith("__") and not callable(value)
    }
    return config_vars

def run_for_single_folder(training_folder, MAX_TOTAL_TIMESTEPS, single_seed, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH, env_type):
    """
    Runs training for a single scenario folder and a single seed.
    Saves the results for each environment type immediately after training.
    """
    stripped_scenario_folder = training_folder.strip("/").split("/")[-1]  # handle trailing slash
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Create the numpy directory if it doesn't exist
    os.makedirs(f"{save_results_big_run}/numpy", exist_ok=True)

    # Run training for the given seed - results will be saved during training for each env_type
    rewards_myopic, rewards_proactive, rewards_reactive, rewards_drl_greedy, \
    test_rewards_myopic, test_rewards_proactive, test_rewards_reactive, test_rewards_drl_greedy = run_train_dqn_both_timesteps(
        MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
        single_seed=single_seed,
        brute_force_flag=brute_force_flag,
        cross_val_flag=cross_val_flag,
        early_stopping_flag=early_stopping_flag,
        CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
        printing_intermediate_results=printing_intermediate_results,
        TRAINING_FOLDERS_PATH=training_folder,
        stripped_scenario_folder=stripped_scenario_folder,
        save_folder=save_folder,
        save_results_big_run=save_results_big_run,
        TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH,
        env_type=env_type
    )

    # Note: The rewards are already in the correct format (lists) and saved to disk
    # No need to process them further here since they're saved during training
    # Just return them for any subsequent processing if needed
    return

def aggregate_results_and_plot(SEEDS, MAX_TOTAL_TIMESTEPS, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH):
    """
    After all seeds have finished, this function will:
    - Identify all scenario folders
    - Load the per-seed arrays
    - Combine them (mean, std) just like originally
    - Produce a single plot per scenario with mean and std shaded area
    """

    # Add debug logging at the start
    print(f"\nStarting aggregation in folder: {save_folder}")
    print(f"Looking for data from seeds: {SEEDS}")

    # Define environment types and their properties
    env_types = [
        {'name': 'myopic', 'label': 'DQN Proactive-N', 'color': 'blue'},
        {'name': 'proactive', 'label': 'DQN Proactive-U', 'color': 'orange'},
        {'name': 'reactive', 'label': 'DQN Reactive', 'color': 'green'},
        {'name': 'drl-greedy', 'label': 'DQN Greedy-Guided', 'color': 'red'}
    ]

    # Identify scenario folders that contain numpy data
    scenario_folders = []
    for d in os.listdir(save_folder):
        folder_path = os.path.join(save_folder, d)
        numpy_path = os.path.join(folder_path, "numpy")
        
        print(f"\nChecking folder: {folder_path}")
        print(f"Numpy path exists: {os.path.exists(numpy_path)}")
        
        if os.path.isdir(folder_path) and os.path.exists(numpy_path):
            # Check which env types have data
            available_env_types = []
            for env_type in env_types:
                expected_file = os.path.join(numpy_path, f"{env_type['name']}_runs_seed_{SEEDS[0]}.npy")
                print(f"Looking for: {expected_file}")
                if os.path.exists(expected_file):
                    print(f"✓ Found {env_type['name']} data")
                    available_env_types.append(env_type)
                else:
                    print(f"✗ Missing {env_type['name']} data")
            
            # If we have at least one environment type with data, include this folder
            if available_env_types:
                scenario_folders.append({
                    'path': folder_path,
                    'available_env_types': available_env_types
                })
                print(f"✓ Added {folder_path} to scenario folders with {len(available_env_types)} environment types")

    if not scenario_folders:
        print("\nERROR: No scenario folders with any data found. Skipping aggregation.")
        return

    def smooth(data, window=10):
        if window > 1 and len(data) >= window:
            return np.convolve(data, np.ones(window)/window, mode='valid')
        return data

    for scenario in scenario_folders:
        scenario_path = scenario['path']
        available_env_types = scenario['available_env_types']
        stripped_scenario_folder = os.path.basename(scenario_path)
        numpy_path = os.path.join(scenario_path, "numpy")
        print(f"\nProcessing scenario: {stripped_scenario_folder}")

        # Initialize data structures only for available environment types
        all_runs = {env_type['name']: [] for env_type in available_env_types}
        all_steps_runs = {env_type['name']: [] for env_type in available_env_types}
        all_test_rewards = {env_type['name']: [] for env_type in available_env_types}

        # Load data for each seed and available environment type
        for seed in SEEDS:
            for env_type in available_env_types:
                try:
                    runs = np.load(os.path.join(numpy_path, f"{env_type['name']}_runs_seed_{seed}.npy"), allow_pickle=True)
                    steps = np.load(os.path.join(numpy_path, f"{env_type['name']}_steps_runs_seed_{seed}.npy"), allow_pickle=True)
                    
                    all_runs[env_type['name']].append(runs)
                    all_steps_runs[env_type['name']].append(steps)

                    if cross_val_flag:
                        test_rewards = np.load(os.path.join(numpy_path, f"test_rewards_{env_type['name']}_seed_{seed}.npy"), allow_pickle=True)
                        all_test_rewards[env_type['name']].append(test_rewards)

                except FileNotFoundError as e:
                    print(f"Warning: Files missing for seed {seed} and env_type {env_type['name']}: {e}")
                    continue

        # Process data only for available environment types
        processed_data = {}
        for env_type in available_env_types:
            name = env_type['name']
            if not all_runs[name]:
                continue

            # Find minimum length and truncate arrays
            min_length = min(len(run) for run in all_runs[name] if len(run) > 0)
            runs_array = np.array([run[:min_length] for run in all_runs[name]])
            steps_array = np.array([steps[:min_length] for steps in all_steps_runs[name]])

            # Calculate statistics
            mean = runs_array.mean(axis=0)
            std = runs_array.std(axis=0)
            steps_mean = steps_array.mean(axis=0).astype(int)

            # Apply smoothing
            smooth_window = 1
            mean_sm = smooth(mean, smooth_window)
            std_sm = smooth(std, smooth_window)
            steps_sm = steps_mean[:len(mean_sm)]

            processed_data[name] = {
                'mean_sm': mean_sm,
                'std_sm': std_sm,
                'steps_sm': steps_sm
            }

            # Save combined arrays
            np.save(f'{numpy_path}/all_{name}_runs.npy', runs_array)
            np.save(f'{numpy_path}/all_{name}_steps_runs.npy', steps_array)

            if cross_val_flag and all_test_rewards[name]:
                np.save(f'{numpy_path}/all_test_rewards_{name}.npy', np.array(all_test_rewards[name]))

        # Create plots directory
        plots_dir = f"{scenario_path}/plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Plot training curves only for available environment types
        plt.figure(figsize=(12,6))
        for env_type in available_env_types:
            name = env_type['name']
            if name in processed_data:
                data = processed_data[name]
                plt.plot(data['steps_sm'], data['mean_sm'], label=env_type['label'], color=env_type['color'])
                plt.fill_between(data['steps_sm'],
                               data['mean_sm'] - data['std_sm'],
                               data['mean_sm'] + data['std_sm'],
                               alpha=0.2, color=env_type['color'])

        plt.xlabel("Environment Steps (Frames)")
        plt.ylabel("Episode Reward")
        plt.title(f"Episode Rewards over {len(SEEDS)} Seeds ({stripped_scenario_folder})" if len(SEEDS) > 1 else f"Episode Rewards ({stripped_scenario_folder})")
        plt.legend(frameon=False)
        plt.grid(True)

        plot_file = os.path.join(plots_dir, f"averaged_rewards_over_steps_{stripped_scenario_folder}.png")
        plt.savefig(plot_file)
        print(f"Combined plot saved for scenario {stripped_scenario_folder} at {plot_file}")

        # Plot cross validation results if enabled
        if cross_val_flag:
            plt.figure(figsize=(12, 6))
            cv_steps = np.arange(len(next(iter(processed_data.values()))['mean_sm'])) * CROSS_VAL_INTERVAL

            # Plot training curves
            for env_type in available_env_types:
                name = env_type['name']
                if name in processed_data:
                    data = processed_data[name]
                    plt.plot(data['steps_sm'], data['mean_sm'], 
                            label=f"Train {env_type['label']}", color=env_type['color'])
                    plt.fill_between(data['steps_sm'],
                                   data['mean_sm'] - data['std_sm'],
                                   data['mean_sm'] + data['std_sm'],
                                   alpha=0.2, color=env_type['color'])

                    # Plot cross validation points
                    if all_test_rewards[name]:
                        test_data = np.array(all_test_rewards[name])
                        test_mean = np.mean(test_data, axis=0)
                        test_std = np.std(test_data, axis=0)
                        plt.plot(cv_steps, test_mean, 'o-', 
                                label=f"CV {env_type['label']}", 
                                color=env_type['color'], alpha=0.5, markersize=4)
                        plt.fill_between(cv_steps,
                                       test_mean - test_std,
                                       test_mean + test_std,
                                       alpha=0.1, color=env_type['color'])

            plt.xlabel("Environment Steps")
            plt.ylabel("Episode Reward")
            plt.title(f"Training and Cross Validation Rewards over {len(SEEDS)} Seeds ({stripped_scenario_folder})" if len(SEEDS) > 1 else f"Training and Cross Validation Rewards ({stripped_scenario_folder})")
            plt.legend(frameon=False)
            plt.grid(True)

            combined_plot_file = os.path.join(plots_dir, f"averaged_rewards_over_steps_{stripped_scenario_folder}_combined.png")
            plt.savefig(combined_plot_file)
            print(f"Combined training and cross validation plot saved for scenario {stripped_scenario_folder} at {combined_plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_folder", type=str, help="Path to a single training folder")
    parser.add_argument("--seed", type=int, help="Specific seed to run")
    parser.add_argument("--env_type", type=str, choices=['myopic', 'proactive', 'reactive', 'drl-greedy'], help="Environment type to run")
    args = parser.parse_args()

    # Common configuration
    MAX_TOTAL_TIMESTEPS = 2e6
    SEEDS = [232323, 242424]
    brute_force_flag = False
    cross_val_flag = False
    early_stopping_flag = False
    CROSS_VAL_INTERVAL = 1
    printing_intermediate_results = False
    save_folder = "3-aaa-130-supertje-diverse-with-old-config"
    TESTING_FOLDERS_PATH = "data/Testing/6ac-100-superdiverse/"

    # Define environment types
    env_types = ['myopic', 'proactive', 'reactive']

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_folders_temp = [
        # "data/Training/6ac-100-stochastic-low/",
        # "data/Training/6ac-100-stochastic-medium/",
        # "data/Training/6ac-100-stochastic-high/",
        # "data/Training/6ac-700-diverse/",
        "data/RESULTS/6ac-130-supertje-diverse/"
    ]

    if args.seed is None and args.training_folder is None:
        # Controller mode: Spawn multiple subprocesses, one per combination of seed and env_type
        config_values = get_config_variables(config)
        config_df = pd.DataFrame([config_values])
        config_df['MAX_TOTAL_TIMESTEPS'] = MAX_TOTAL_TIMESTEPS
        config_df['SEEDS'] = str([str(seed) for seed in SEEDS])
        config_df['brute_force_flag'] = brute_force_flag
        config_df['cross_val_flag'] = cross_val_flag
        config_df['early_stopping_flag'] = early_stopping_flag
        config_df['CROSS_VAL_INTERVAL'] = CROSS_VAL_INTERVAL
        config_df['printing_intermediate_results'] = printing_intermediate_results
        config_df.to_csv(f"{save_folder}/config.csv", index=False)

        start_time = time.time()

        processes = []
        for seed in SEEDS:
            for env_type in env_types:
                cmd = [
                    "python", "main.py",
                    "--seed", str(seed),
                    "--env_type", env_type,
                    "--training_folder", all_folders_temp[0]  # Add the training folder
                ]
                p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                processes.append(p)

        # Wait for all subprocesses to finish
        for p in processes:
            p.wait()

        # After all combinations are done, aggregate and plot the combined results
        aggregate_results_and_plot(
            SEEDS=SEEDS,
            MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
            brute_force_flag=brute_force_flag,
            cross_val_flag=cross_val_flag,
            early_stopping_flag=early_stopping_flag,
            CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
            printing_intermediate_results=printing_intermediate_results,
            save_folder=save_folder,
            TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH
        )

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"\nTotal runtime: {total_runtime:.2f} seconds")

    elif args.seed is not None and args.training_folder:
        # Worker mode: run a single scenario folder for a single seed and env_type
        run_for_single_folder(
            training_folder=args.training_folder,
            MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
            single_seed=args.seed,
            brute_force_flag=brute_force_flag,
            cross_val_flag=cross_val_flag,
            early_stopping_flag=early_stopping_flag,
            CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
            printing_intermediate_results=printing_intermediate_results,
            save_folder=save_folder,
            TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH,
            env_type=args.env_type
        )

    elif args.seed is not None and args.training_folder is None:
        # Worker mode: run all scenario folders for this single seed and env_type
        for folder in all_folders_temp:
            run_for_single_folder(
                training_folder=folder,
                MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
                single_seed=args.seed,
                brute_force_flag=brute_force_flag,
                cross_val_flag=cross_val_flag,
                early_stopping_flag=early_stopping_flag,
                CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
                printing_intermediate_results=printing_intermediate_results,
                save_folder=save_folder,
                TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH,
                env_type=args.env_type
            )
    else:
        pass
