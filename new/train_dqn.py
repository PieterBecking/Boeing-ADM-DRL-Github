# train_module.py

import os
import numpy as np
import json
import time
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environment import AircraftDisruptionEnv
from scripts.utils import load_scenario_data, calculate_epsilon_decay_rate, initialize_device
from scripts.utils import verify_training_folders

def train_dqn(env_type, training_parameters, TRAINING_FOLDERS_PATH):
    """
    Train a DQN agent on the given training folders.

    Parameters:
    - env_type: str, "myopic" or "proactive"
    - training_parameters: dict containing necessary hyperparameters
    - TRAINING_FOLDERS_PATH: str, path to training scenario folders

    Returns:
    - results: dict with keys:
        "episode_rewards": list of average rewards per episode
        "step_rewards": dict of {episode: [rewards_per_step_in_that_episode]}
    """

    # Extract training parameters
    MAX_TOTAL_TIMESTEPS = training_parameters.get('MAX_TOTAL_TIMESTEPS', 100000)
    SEEDS = training_parameters.get('SEEDS', [0])
    LEARNING_RATE = training_parameters.get('LEARNING_RATE', 0.0001)
    GAMMA = training_parameters.get('GAMMA', 0.99)
    BUFFER_SIZE = training_parameters.get('BUFFER_SIZE', 100000)
    BATCH_SIZE = training_parameters.get('BATCH_SIZE', 128)
    TARGET_UPDATE_INTERVAL = training_parameters.get('TARGET_UPDATE_INTERVAL', 1000)
    NEURAL_NET_STRUCTURE = training_parameters.get('NEURAL_NET_STRUCTURE', dict(net_arch=[256, 256]))
    LEARNING_STARTS = training_parameters.get('LEARNING_STARTS', 10000)
    TRAIN_FREQ = training_parameters.get('TRAIN_FREQ', 4)
    EPSILON_START = training_parameters.get('EPSILON_START', 1.0)
    EPSILON_MIN = training_parameters.get('EPSILON_MIN', 0.025)
    PERCENTAGE_MIN = training_parameters.get('PERCENTAGE_MIN', 95)
    EPSILON_TYPE = training_parameters.get('EPSILON_TYPE', 'exponential')
    N_EPISODES = training_parameters.get('N_EPISODES', 50)
    brute_force_flag = training_parameters.get('brute_force_flag', False)

    # Calculate epsilon decay
    EPSILON_DECAY_RATE = calculate_epsilon_decay_rate(
        MAX_TOTAL_TIMESTEPS, EPSILON_START, EPSILON_MIN, PERCENTAGE_MIN, EPSILON_TYPE
    )

    # Initialize device
    device = initialize_device()

    # Verify training folders
    training_folders = verify_training_folders(TRAINING_FOLDERS_PATH)
    scenario_folders = [
        os.path.join(TRAINING_FOLDERS_PATH, folder)
        for folder in os.listdir(TRAINING_FOLDERS_PATH)
        if os.path.isdir(os.path.join(TRAINING_FOLDERS_PATH, folder))
    ]

    # For reproducibility
    seed = SEEDS[0] if SEEDS else 0
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    th.manual_seed(seed)

    # Load a dummy scenario to initialize the environment
    dummy_scenario_folder = scenario_folders[0]
    data_dict = load_scenario_data(dummy_scenario_folder)
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

    # Initialize DQN
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

    epsilon = EPSILON_START
    total_timesteps = 0

    # Data structures to store rewards
    step_rewards = {}
    episode_rewards = []

    # Training loop
    # Note: Either loop for N_EPISODES or until MAX_TOTAL_TIMESTEPS is reached.
    # The original code used both conditions. We will mimic that behavior:
    # run until total_timesteps < MAX_TOTAL_TIMESTEPS
    episode = 0
    while total_timesteps < MAX_TOTAL_TIMESTEPS and episode < N_EPISODES:

        # A dictionary to store step rewards for this episode
        step_rewards[episode] = []
        episode_total_reward = 0

        # Cycle through all scenario folders
        # (If you wish to do per-episode with random scenario each time, adjust accordingly)
        for scenario_folder in scenario_folders:
            # Load the data for this scenario
            data_dict = load_scenario_data(scenario_folder)
            aircraft_dict = data_dict['aircraft']
            flights_dict = data_dict['flights']
            rotations_dict = data_dict['rotations']
            alt_aircraft_dict = data_dict['alt_aircraft']
            config_dict = data_dict['config']

            # Reinitialize the environment with the new scenario
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

            while not done_flag:
                model.exploration_rate = epsilon
                action_mask = obs['action_mask']

                # Convert observation to float32
                obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}

                # Preprocess and get Q-values
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

                # Mask invalid actions
                masked_q_values = q_values.copy()
                masked_q_values[action_mask == 0] = -np.inf

                # Action selection
                if np.random.rand() < epsilon or brute_force_flag:
                    # Exploration
                    valid_actions = np.where(action_mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    # Exploitation
                    action = np.argmax(masked_q_values)

                # Step in the environment
                obs_next, reward, terminated, truncated, info = env.step(action)

                # Save step reward
                step_rewards[episode].append(reward)
                episode_total_reward += reward

                done_flag = terminated or truncated

                # Add to replay buffer
                model.replay_buffer.add(
                    obs=obs,
                    next_obs=obs_next,
                    action=action,
                    reward=reward,
                    done=done_flag,
                    infos=[info]
                )

                # Update observation
                obs = obs_next

                # Update epsilon
                epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE))
                total_timesteps += 1

                # Train periodically
                if total_timesteps > model.learning_starts and total_timesteps % TRAIN_FREQ == 0:
                    model.train(gradient_steps=1, batch_size=BATCH_SIZE)

                # Update target network
                if total_timesteps % model.target_update_interval == 0:
                    polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)
                    polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)

                if done_flag:
                    break

            # If we've reached max timesteps, break out of scenario loop
            if total_timesteps >= MAX_TOTAL_TIMESTEPS:
                break

        episode_rewards.append(episode_total_reward / len(scenario_folders))
        episode += 1

    # Return results
    results = {
        "episode_rewards": episode_rewards,
        "step_rewards": step_rewards
    }

    # If needed, you can save the results to JSON here or return them directly
    # Convert results to serializable format before saving
    # results_serializable = convert_to_serializable(results)
    # with open("rewards_log.json", "w") as f:
    #     json.dump(results_serializable, f, indent=4)

    return results
