import argparse
import os
import time
import subprocess
import src.config as config
import pandas as pd
from train_dqn_modular import run_train_dqn_both_timesteps

def get_config_variables(config_module):
    config_vars = {
        key: value for key, value in vars(config_module).items()
        if not key.startswith("__") and not callable(value)
    }
    return config_vars

all_folders = [
    "data/Training/6ac-100-stochastic-low/",
    "data/Training/6ac-100-stochastic-medium/",
    "data/Training/6ac-100-stochastic-high/",
    "data/Training/6ac-100-mixed-low/",
    "data/Training/6ac-100-mixed-medium/",
    "data/Training/6ac-100-mixed-high/",
]

def run_for_single_folder(training_folder, MAX_TOTAL_TIMESTEPS, SEEDS, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH):

    stripped_scenario_folder = training_folder.strip("/").split("/")[-1]  # handle trailing slash
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    run_train_dqn_both_timesteps(
        MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
        SEEDS=SEEDS,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This argument is optional. If given, we run for a single folder.
    parser.add_argument("--training_folder", type=str, help="Path to a single training folder")
    args = parser.parse_args()

    # Common configuration
    MAX_TOTAL_TIMESTEPS = 1000
    SEEDS = [42]
    brute_force_flag = False
    cross_val_flag = False
    early_stopping_flag = False
    CROSS_VAL_INTERVAL = 10000
    printing_intermediate_results = True
    save_folder = "4-abc-big-run"
    TESTING_FOLDERS_PATH = "none"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if args.training_folder:
        # Worker mode: Run the training for a single specified folder
        run_for_single_folder(
            training_folder=args.training_folder,
            MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
            SEEDS=SEEDS,
            brute_force_flag=brute_force_flag,
            cross_val_flag=cross_val_flag,
            early_stopping_flag=early_stopping_flag,
            CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
            printing_intermediate_results=printing_intermediate_results,
            save_folder=save_folder,
            TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH
        )
    else:
        # Controller mode: Spawn multiple subprocesses
        all_folders_temp = [
            # "data/Training/6ac-100-stochastic-low/",
            "data/Training/6ac-100-stochastic-medium/",
            # "data/Training/6ac-100-stochastic-high/",
        ]

        # Save config only once
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
        for folder in all_folders_temp:
            cmd = [
                "python", "main.py",
                "--training_folder", folder
            ]
            # Start subprocess in root directory
            p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            processes.append(p)

        # Wait for all subprocesses to finish
        for p in processes:
            p.wait()

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"\nTotal runtime: {total_runtime:.2f} seconds")
