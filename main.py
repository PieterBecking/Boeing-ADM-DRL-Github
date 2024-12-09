import numpy as np
import random
import time
import src.config as config
import pandas as pd
import os
from train_dqn_modular import run_train_dqn_both_timesteps

def get_config_variables(config_module):
    config_vars = {
        key: value for key, value in vars(config_module).items()
        if not key.startswith("__") and not callable(value)  # Exclude magic methods and functions
    }
    return config_vars

all_folders = [
    "../data/Training/6ac-100-deterministic-na/",
    "../data/Training/6ac-100-mixed-low/",
    "../data/Training/6ac-100-mixed-medium/",
    "../data/Training/6ac-100-mixed-high/",
    "../data/Training/6ac-100-stochastic-low/",
    "../data/Training/6ac-100-stochastic-medium/",
    "../data/Training/6ac-100-stochastic-high/",
]

all_folders_temp = [
    "../data/Training/6ac-100-mixed-low/",
    "../data/Training/6ac-100-mixed-medium/",
    "../data/Training/6ac-100-mixed-high/",
]


MAX_TOTAL_TIMESTEPS = 1000
SEEDS = [42, 43, 44]
brute_force_flag = False
cross_val_flag = False
early_stopping_flag = False
CROSS_VAL_INTERVAL = 10000
printing_intermediate_results = True

save_folder = "abc-big-run"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

config_values = get_config_variables(config)
print(config_values)
config_df = pd.DataFrame([config_values])
config_df['MAX_TOTAL_TIMESTEPS'] = MAX_TOTAL_TIMESTEPS
config_df['SEEDS'] = str([str(seed) for seed in SEEDS])
config_df['brute_force_flag'] = brute_force_flag
config_df['cross_val_flag'] = cross_val_flag
config_df['early_stopping_flag'] = early_stopping_flag
config_df['CROSS_VAL_INTERVAL'] = CROSS_VAL_INTERVAL
config_df['printing_intermediate_results'] = printing_intermediate_results
config_df.to_csv(f"{save_folder}/config.csv", index=False)


step = 0
for training_folder in all_folders_temp:
    step += 1
    print(f"Step {step}")
    print("\n\n\n\n\n")
    print(f"**** TRAINING_FOLDERS_PATH: {training_folder} ****")
    print("\n\n\n--- DQN ---\n\n")
    
    TRAINING_FOLDERS_PATH = training_folder
    stripped_scenario_folder = TRAINING_FOLDERS_PATH.split("/")[-2]
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Now we directly call the function instead of using %run
    run_train_dqn_both_timesteps(
        MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
        SEEDS=SEEDS,
        brute_force_flag=brute_force_flag,
        cross_val_flag=cross_val_flag,
        early_stopping_flag=early_stopping_flag,
        CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
        printing_intermediate_results=printing_intermediate_results,
        TRAINING_FOLDERS_PATH=TRAINING_FOLDERS_PATH,
        stripped_scenario_folder=stripped_scenario_folder,
        save_folder=save_folder,
        save_results_big_run=save_results_big_run
    )

    print("\n\n--- PPO ---\n\n")
    # If needed, call the PPO training similarly (after you refactor it as well)
    # %run train_ppo_both.ipynb
