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

def run_for_single_folder(training_folder, MAX_TOTAL_TIMESTEPS, single_seed, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH):
    """
    Runs training for a single scenario folder and a single seed.
    Saves the per-seed arrays but does NOT produce the final combined plot.
    The final combined plot is produced later by an aggregation step after all seeds finish.
    """

    stripped_scenario_folder = training_folder.strip("/").split("/")[-1]  # handle trailing slash
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Run training for the given seed
    rewards_myopic, rewards_proactive, rewards_reactive, \
    test_rewards_myopic, test_rewards_proactive, test_rewards_reactive = run_train_dqn_both_timesteps(
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
        TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH
    )

    myopic_episode_rewards = [
        rewards_myopic[e]["avg_reward"] for e in sorted(rewards_myopic.keys())
        if "avg_reward" in rewards_myopic[e]
    ]
    myopic_episode_steps = [
        rewards_myopic[e]["total_timesteps"] for e in sorted(rewards_myopic.keys())
        if "total_timesteps" in rewards_myopic[e]
    ]

    proactive_episode_rewards = [
        rewards_proactive[e]["avg_reward"] for e in sorted(rewards_proactive.keys())
        if "avg_reward" in rewards_proactive[e]
    ]
    proactive_episode_steps = [
        rewards_proactive[e]["total_timesteps"] for e in sorted(rewards_proactive.keys())
        if "total_timesteps" in rewards_proactive[e]
    ]

    reactive_episode_rewards = [
        rewards_reactive[e]["avg_reward"] for e in sorted(rewards_reactive.keys())
        if "avg_reward" in rewards_reactive[e]
    ]
    reactive_episode_steps = [
        rewards_reactive[e]["total_timesteps"] for e in sorted(rewards_reactive.keys())
        if "total_timesteps" in rewards_reactive[e]
    ]

    # Save results for this seed
    os.makedirs(f"{save_results_big_run}/numpy", exist_ok=True)

    np.save(f'{save_results_big_run}/numpy/myopic_runs_seed_{single_seed}.npy', np.array(myopic_episode_rewards))
    np.save(f'{save_results_big_run}/numpy/proactive_runs_seed_{single_seed}.npy', np.array(proactive_episode_rewards))
    np.save(f'{save_results_big_run}/numpy/myopic_steps_runs_seed_{single_seed}.npy', np.array(myopic_episode_steps))
    np.save(f'{save_results_big_run}/numpy/proactive_steps_runs_seed_{single_seed}.npy', np.array(proactive_episode_steps))
    np.save(f'{save_results_big_run}/numpy/reactive_runs_seed_{single_seed}.npy', np.array(reactive_episode_rewards))
    np.save(f'{save_results_big_run}/numpy/reactive_steps_runs_seed_{single_seed}.npy', np.array(reactive_episode_steps))
    np.save(f'{save_results_big_run}/numpy/test_rewards_myopic_seed_{single_seed}.npy', test_rewards_myopic)
    np.save(f'{save_results_big_run}/numpy/test_rewards_proactive_seed_{single_seed}.npy', test_rewards_proactive)
    np.save(f'{save_results_big_run}/numpy/test_rewards_reactive_seed_{single_seed}.npy', test_rewards_reactive)

def aggregate_results_and_plot(SEEDS, MAX_TOTAL_TIMESTEPS, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH):
    """
    After all seeds have finished, this function will:
    - Identify all scenario folders
    - Load the per-seed arrays
    - Combine them (mean, std) just like originally
    - Produce a single plot per scenario with mean and std shaded area
    """

    # Identify scenario folders that were created
    # We assume all scenario results are in save_folder and contain a numpy folder
    scenario_folders = [os.path.join(save_folder, d) for d in os.listdir(save_folder) 
                        if os.path.isdir(os.path.join(save_folder, d)) and os.path.exists(os.path.join(save_folder, d, "numpy"))]

    def smooth(data, window=10):
        if window > 1 and len(data) >= window:
            return np.convolve(data, np.ones(window)/window, mode='valid')
        return data

    for scenario_path in scenario_folders:
        stripped_scenario_folder = os.path.basename(scenario_path)
        numpy_path = os.path.join(scenario_path, "numpy")

        all_myopic_runs = []
        all_proactive_runs = []
        all_reactive_runs = []
        all_myopic_steps_runs = []
        all_proactive_steps_runs = []
        all_reactive_steps_runs = []
        all_test_rewards_myopic = []
        all_test_rewards_proactive = []
        all_test_rewards_reactive = []

        for seed in SEEDS:
            myopic_runs_seed = np.load(os.path.join(numpy_path, f"myopic_runs_seed_{seed}.npy"), allow_pickle=True)
            proactive_runs_seed = np.load(os.path.join(numpy_path, f"proactive_runs_seed_{seed}.npy"), allow_pickle=True)
            reactive_runs_seed = np.load(os.path.join(numpy_path, f"reactive_runs_seed_{seed}.npy"), allow_pickle=True)

            myopic_steps_seed = np.load(os.path.join(numpy_path, f"myopic_steps_runs_seed_{seed}.npy"), allow_pickle=True)
            proactive_steps_seed = np.load(os.path.join(numpy_path, f"proactive_steps_runs_seed_{seed}.npy"), allow_pickle=True)
            reactive_steps_seed = np.load(os.path.join(numpy_path, f"reactive_steps_runs_seed_{seed}.npy"), allow_pickle=True)

            test_rewards_myopic_seed = np.load(os.path.join(numpy_path, f"test_rewards_myopic_seed_{seed}.npy"), allow_pickle=True)
            test_rewards_proactive_seed = np.load(os.path.join(numpy_path, f"test_rewards_proactive_seed_{seed}.npy"), allow_pickle=True)
            test_rewards_reactive_seed = np.load(os.path.join(numpy_path, f"test_rewards_reactive_seed_{seed}.npy"), allow_pickle=True)

            all_myopic_runs.append(myopic_runs_seed)
            all_proactive_runs.append(proactive_runs_seed)
            all_reactive_runs.append(reactive_runs_seed)
            all_myopic_steps_runs.append(myopic_steps_seed)
            all_proactive_steps_runs.append(proactive_steps_seed)
            all_reactive_steps_runs.append(reactive_steps_seed)

            all_test_rewards_myopic.append(test_rewards_myopic_seed)
            all_test_rewards_proactive.append(test_rewards_proactive_seed)
            all_test_rewards_reactive.append(test_rewards_reactive_seed)

        # Ensure arrays are consistent length
        min_length_myopic = min(len(run) for run in all_myopic_runs if len(run) > 0) if all_myopic_runs and all(len(r) > 0 for r in all_myopic_runs) else 0
        min_length_proactive = min(len(run) for run in all_proactive_runs if len(run) > 0) if all_proactive_runs and all(len(r) > 0 for r in all_proactive_runs) else 0
        min_length_reactive = min(len(run) for run in all_reactive_runs if len(run) > 0) if all_reactive_runs and all(len(r) > 0 for r in all_reactive_runs) else 0

        all_myopic_runs = [run[:min_length_myopic] for run in all_myopic_runs]
        all_proactive_runs = [run[:min_length_proactive] for run in all_proactive_runs]
        all_reactive_runs = [run[:min_length_reactive] for run in all_reactive_runs]

        all_myopic_steps_runs = [steps[:min_length_myopic] for steps in all_myopic_steps_runs]
        all_proactive_steps_runs = [steps[:min_length_proactive] for steps in all_proactive_steps_runs]
        all_reactive_steps_runs = [steps[:min_length_reactive] for steps in all_reactive_steps_runs]

        all_myopic_runs = np.array(all_myopic_runs)
        all_proactive_runs = np.array(all_proactive_runs)
        all_myopic_steps_runs = np.array(all_myopic_steps_runs)
        all_proactive_steps_runs = np.array(all_proactive_steps_runs)
        all_reactive_runs = np.array(all_reactive_runs)
        all_reactive_steps_runs = np.array(all_reactive_steps_runs)

        # Save combined arrays
        np.save(f'{numpy_path}/all_myopic_runs.npy', all_myopic_runs)
        np.save(f'{numpy_path}/all_proactive_runs.npy', all_proactive_runs) 
        np.save(f'{numpy_path}/all_reactive_runs.npy', all_reactive_runs)
        np.save(f'{numpy_path}/all_myopic_steps_runs.npy', all_myopic_steps_runs)
        np.save(f'{numpy_path}/all_proactive_steps_runs.npy', all_proactive_steps_runs)
        np.save(f'{numpy_path}/all_reactive_steps_runs.npy', all_reactive_steps_runs)
        if cross_val_flag:
            np.save(f'{numpy_path}/all_test_rewards_myopic.npy', all_test_rewards_myopic)
            np.save(f'{numpy_path}/all_test_rewards_proactive.npy', all_test_rewards_proactive)
            np.save(f'{numpy_path}/all_test_rewards_reactive.npy', all_test_rewards_reactive)

        myopic_mean = all_myopic_runs.mean(axis=0) if all_myopic_runs.size > 0 else []
        myopic_std = all_myopic_runs.std(axis=0) if all_myopic_runs.size > 0 else []
        proactive_mean = all_proactive_runs.mean(axis=0) if all_proactive_runs.size > 0 else []
        proactive_std = all_proactive_runs.std(axis=0) if all_proactive_runs.size > 0 else []
        reactive_mean = all_reactive_runs.mean(axis=0) if all_reactive_runs.size > 0 else []
        reactive_std = all_reactive_runs.std(axis=0) if all_reactive_runs.size > 0 else []

        myopic_steps_mean = all_myopic_steps_runs.mean(axis=0).astype(int) if all_myopic_steps_runs.size > 0 else []
        proactive_steps_mean = all_proactive_steps_runs.mean(axis=0).astype(int) if all_proactive_steps_runs.size > 0 else []
        reactive_steps_mean = all_reactive_steps_runs.mean(axis=0).astype(int) if all_reactive_steps_runs.size > 0 else []

        # Smoothing not really needed if window=1, but kept for parity
        smooth_window = 1
        def apply_smooth(mean_arr, std_arr, steps_arr):
            mean_sm = smooth(mean_arr, smooth_window)
            std_sm = smooth(std_arr, smooth_window)
            steps_sm = steps_arr[:len(mean_sm)]
            return mean_sm, std_sm, steps_sm

        if len(proactive_mean) > 0:
            proactive_mean_sm, proactive_std_sm, proactive_steps_sm = apply_smooth(proactive_mean, proactive_std, proactive_steps_mean)
        else:
            proactive_mean_sm, proactive_std_sm, proactive_steps_sm = [], [], []

        if len(myopic_mean) > 0:
            myopic_mean_sm, myopic_std_sm, myopic_steps_sm = apply_smooth(myopic_mean, myopic_std, myopic_steps_mean)
        else:
            myopic_mean_sm, myopic_std_sm, myopic_steps_sm = [], [], []

        if len(reactive_mean) > 0:
            reactive_mean_sm, reactive_std_sm, reactive_steps_sm = apply_smooth(reactive_mean, reactive_std, reactive_steps_mean)
        else:
            reactive_mean_sm, reactive_std_sm, reactive_steps_sm = [], [], []
        # Create one figure that shows both training and cross-validation curves on the same axis
        plt.figure(figsize=(12, 6))

        # Plot training curves
        if len(proactive_mean_sm) > 0:
            plt.plot(proactive_steps_sm, proactive_mean_sm, label="Train DQN Proactive-U", color='orange')
            plt.fill_between(proactive_steps_sm, 
                             proactive_mean_sm - proactive_std_sm, 
                             proactive_mean_sm + proactive_std_sm, 
                             alpha=0.2, color='orange')

        if len(myopic_mean_sm) > 0:
            plt.plot(myopic_steps_sm, myopic_mean_sm, label="Train DQN Proactive-N", color='blue')
            plt.fill_between(myopic_steps_sm, 
                             myopic_mean_sm - myopic_std_sm, 
                             myopic_mean_sm + myopic_std_sm, 
                             alpha=0.2, color='blue')

        if len(reactive_mean_sm) > 0:
            plt.plot(reactive_steps_sm, reactive_mean_sm, label="Train DQN Reactive", color='green')
            plt.fill_between(reactive_steps_sm, 
                             reactive_mean_sm - reactive_std_sm, 
                             reactive_mean_sm + reactive_std_sm, 
                             alpha=0.2, color='green')

        # If we have cross-validation data, plot them at their actual training steps
        if cross_val_flag:
            all_test_rewards_myopic = np.array(all_test_rewards_myopic)
            all_test_rewards_proactive = np.array(all_test_rewards_proactive)
            all_test_rewards_reactive = np.array(all_test_rewards_reactive)

            if all_test_rewards_myopic.size > 0:
                myopic_test_mean = all_test_rewards_myopic.mean(axis=0)
                myopic_test_std = all_test_rewards_myopic.std(axis=0)
            else:
                myopic_test_mean, myopic_test_std = [], []

            if all_test_rewards_proactive.size > 0:
                proactive_test_mean = all_test_rewards_proactive.mean(axis=0)
                proactive_test_std = all_test_rewards_proactive.std(axis=0)
            else:
                proactive_test_mean, proactive_test_std = [], []

            if all_test_rewards_reactive.size > 0:
                reactive_test_mean = all_test_rewards_reactive.mean(axis=0)
                reactive_test_std = all_test_rewards_reactive.std(axis=0)
            else:
                reactive_test_mean, reactive_test_std = [], []

            # Determine min length to align all CV arrays
            lengths = []
            if len(myopic_test_mean) > 0:
                lengths.append(len(myopic_test_mean))
            if len(proactive_test_mean) > 0:
                lengths.append(len(proactive_test_mean))
            if len(reactive_test_mean) > 0:
                lengths.append(len(reactive_test_mean))

            if lengths:
                min_len = min(lengths)
                myopic_test_mean = myopic_test_mean[:min_len]
                myopic_test_std = myopic_test_std[:min_len]
                proactive_test_mean = proactive_test_mean[:min_len]
                proactive_test_std = proactive_test_std[:min_len]
                reactive_test_mean = reactive_test_mean[:min_len]
                reactive_test_std = reactive_test_std[:min_len]

                # Calculate the steps at which cross validation was performed
                cv_steps = np.arange(min_len) * CROSS_VAL_INTERVAL

                # Plot cross validation curves on the same axis
                if len(proactive_test_mean) > 0:
                    plt.plot(cv_steps, proactive_test_mean, label="CV DQN Proactive-U",
                             linestyle='--', marker='.', color='orange', alpha=0.5)
                    plt.fill_between(cv_steps,
                                     proactive_test_mean - proactive_test_std,
                                     proactive_test_mean + proactive_test_std,
                                     alpha=0.2, color='orange')

                if len(myopic_test_mean) > 0:
                    plt.plot(cv_steps, myopic_test_mean, label="CV DQN Proactive-N",
                             linestyle='--', marker='.', color='blue', alpha=0.5)
                    plt.fill_between(cv_steps,
                                     myopic_test_mean - myopic_test_std,
                                     myopic_test_mean + myopic_test_std,
                                     alpha=0.2, color='blue')

                if len(reactive_test_mean) > 0:
                    plt.plot(cv_steps, reactive_test_mean, label="CV DQN Reactive",
                             linestyle='--', marker='.', color='green', alpha=0.5)
                    plt.fill_between(cv_steps,
                                     reactive_test_mean - reactive_test_std,
                                     reactive_test_mean + reactive_test_std,
                                     alpha=0.2, color='green')

        plt.xlabel("Environment Steps (Frames)")
        plt.ylabel("Episode Reward")
        if len(SEEDS) > 1:
            plt.title(f"Training and Cross Validation Rewards over {len(SEEDS)} Seeds ({stripped_scenario_folder})")
        else:
            plt.title(f"Training and Cross Validation Rewards ({stripped_scenario_folder})")
        plt.legend(frameon=False)
        plt.grid(True)

        plots_dir = f"{scenario_path}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        combined_plot_file = os.path.join(plots_dir, f"averaged_rewards_over_steps_{stripped_scenario_folder}_combined.png")
        plt.savefig(combined_plot_file)
        print(f"Combined training and cross validation plot saved for scenario {stripped_scenario_folder} at {combined_plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_folder", type=str, help="Path to a single training folder")
    parser.add_argument("--seed", type=int, help="Specific seed to run")
    args = parser.parse_args()

    # Common configuration
    MAX_TOTAL_TIMESTEPS = 25000
    SEEDS = [25] 
    brute_force_flag = False
    cross_val_flag = True
    early_stopping_flag = False
    CROSS_VAL_INTERVAL = 2
    printing_intermediate_results = False
    save_folder = "03-novel-run"
    TESTING_FOLDERS_PATH = "data/Testing/6ac-700-diverse/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_folders_temp = [
        # "data/Training/6ac-100-stochastic-low/",
        # "data/Training/6ac-100-stochastic-medium/",
        # "data/Training/6ac-100-stochastic-high/",
        "data/Training/6ac-700-diverse/",
    ]

    if args.seed is None and args.training_folder is None:
        # Controller mode: Spawn multiple subprocesses, one per seed
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
            cmd = [
                "python", "main.py",
                "--seed", str(seed)
            ]
            p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            processes.append(p)

        # Wait for all subprocesses to finish
        for p in processes:
            p.wait()

        # After all seeds are done, aggregate and plot the combined results
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
        # Worker mode: run a single scenario folder for a single seed
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
            TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH
        )

    elif args.seed is not None and args.training_folder is None:
        # Worker mode: run all scenario folders for this single seed
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
                TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH
            )
    else:
        pass
