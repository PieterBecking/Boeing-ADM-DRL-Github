import sys
import pandas as pd
import os
import torch as th
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from scripts.utils import *
from scripts.visualizations import *
from src.config import *
from datetime import datetime, timedelta

# Setting all debugging flags to False
DEBUG_MODE = False 
DEBUG_MODE_TRAINING = False 
DEBUG_MODE_REWARD = False  

# Training Settings
LEARNING_RATE = 0.0003
GAMMA = 0.99
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
CLIP_RANGE = 0.2
MAX_TIMESTEPS = 500

N_EPISODES = 25000
NEURAL_NET_STRUCTURE = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

TRAINING_FOLDERS_PATH = '../data/Training-1-day/'

device = th.device('mps' if th.backends.mps.is_available() else 'cpu')

# Verify folders exists
if not os.path.exists(TRAINING_FOLDERS_PATH):
    raise FileNotFoundError(f'Training folder not found at {TRAINING_FOLDERS_PATH}')

# Print all folders in the training folder
training_folders = [folder for folder in os.listdir(TRAINING_FOLDERS_PATH) if os.path.isdir(os.path.join(TRAINING_FOLDERS_PATH, folder))]

num_days_trained_on = N_EPISODES * len(training_folders)
print(f'Training on {num_days_trained_on} days of data ({N_EPISODES} episodes of {len(training_folders)} scenarios)')

model_name = 'ppo_' + str(num_days_trained_on) + "d_" + str(len(training_folders)) + "u"
print('Model name:', model_name)
model_version = get_model_version(model_name)
MODEL_SAVE_PATH = '../trained_models/' + model_name + '-' + model_version + '.zip'

print('Model will be saved to:', MODEL_SAVE_PATH)

from src.environment import AircraftDisruptionEnv

def train_ppo_agent():
    scenario_folders = [os.path.join(TRAINING_FOLDERS_PATH, folder) for folder in os.listdir(TRAINING_FOLDERS_PATH) if os.path.isdir(os.path.join(TRAINING_FOLDERS_PATH, folder))]
    
    total_timesteps = 0
    rewards = []
    action_sequences = {folder: {"best_actions": [], "best_reward": float('-inf'),
                                 "worst_actions": [], "worst_reward": float('inf')} 
                        for folder in scenario_folders}

    # Initialize the PPO model
    env = create_env(scenario_folders[0])
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        verbose=1,
        policy_kwargs=NEURAL_NET_STRUCTURE,
        device='cpu'
    )

    for episode in range(N_EPISODES):
        for scenario_folder in scenario_folders:
            if DEBUG_MODE_TRAINING:
                print(f"Training on scenario {scenario_folder}")
            
            env = create_env(scenario_folder)
            model.set_env(env)

            obs = env.reset()
            done = False
            total_reward = 0
            timesteps = 0
            action_sequence = []

            while not done and timesteps < MAX_TIMESTEPS:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, _ = env.step(action)

                action_sequence.append(action)
                total_reward += reward
                timesteps += 1
                total_timesteps += 1

            rewards.append((episode, scenario_folder, total_reward))

            if total_reward < action_sequences[scenario_folder]["worst_reward"]:
                action_sequences[scenario_folder]["worst_actions"] = action_sequence
                action_sequences[scenario_folder]["worst_reward"] = total_reward

            if total_reward > action_sequences[scenario_folder]["best_reward"]:
                action_sequences[scenario_folder]["best_actions"] = action_sequence
                action_sequences[scenario_folder]["best_reward"] = total_reward

            print(f"Episode {episode}, Scenario {scenario_folder}, Total Reward: {total_reward}")

        # Train the model after each full iteration through all scenarios
        model.learn(total_timesteps=N_STEPS * len(scenario_folders))

    # Save the model after training
    model.save(MODEL_SAVE_PATH)

    return rewards, action_sequences

def create_env(scenario_folder):
    data_dict = load_scenario_data(scenario_folder)
    aircraft_dict = data_dict['aircraft']
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_aircraft_dict = data_dict['alt_aircraft']
    config_dict = data_dict['config']

    env = AircraftDisruptionEnv(aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict)
    return DummyVecEnv([lambda: env])

# Run the training process
rewards, action_sequences = train_ppo_agent()

for scenario, data in action_sequences.items():
    print(f"Scenario: {scenario}, Worst Reward: {data['worst_reward']}, Best Reward: {data['best_reward']}")
    print(f"Worst Action Sequence: {data['worst_actions']}")
    print(f"Best Action Sequence: {data['best_actions']}")
    
    save_best_and_worst_to_csv(scenario, MODEL_SAVE_PATH, data['worst_actions'], data['best_actions'], data['worst_reward'], data['best_reward'])

# Plotting functions (same as in the original script)
# ... [Include the plotting functions here]

# Run the plots
# ... [Include the plot calls here]
