# scripts/logger.py

import json
import os
from datetime import datetime
from scripts.utils import NumpyEncoder
import numpy as np
from filelock import FileLock

def create_new_id(id_type):
    file_path = "ids.json"  
    lock_path = file_path + ".lock"


    with FileLock(lock_path):
        # If file doesn't exist, create it with an empty list
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump([], f)

        # Attempt to read existing data
        with open(file_path, "r") as f:
            try:
                ids = json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted or empty, re-init with empty list
                ids = []

        new_id = f"{id_type}_{len(ids) + 1}"

        # Append the new ID and write back to file
        ids.append(new_id)
        with open(file_path, "w") as f:
            json.dump(ids, f, indent=2)

    print(f"New ID created: {new_id}")

    return new_id

def log_scenario_folder(logs_id, scenario_folder_path, inputs, outputs):
    """
    Logs the scenario creation details into a JSON file.

    Args:
        logs_id (str): Unique ID for the logging session.
        scenario_folder_path (str): Path to the data folder of the scenario.
        inputs (dict): Hyperparameters and inputs used for scenario creation.
        outputs (dict): Generated data and statistics for each scenario.
    """
    log_data = {
        "scenario_folder_id": logs_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_folder": scenario_folder_path,
        "inputs": inputs,
        "outputs": outputs
    }

    log_file_path = os.path.join("..\logs", "scenarios", f"scenario_folder_{logs_id}.json")

    # with open(log_file_path, 'w') as log_file:
    #     json.dump(log_data, log_file, indent=4)

    print(f"Scenario not logged to {log_file_path}")

def mark_log_as_finished(logs_id, additional_info=None):
    """
    Marks the logging session as finished in ids.json.

    Args:
        logs_id (str): Unique ID for the logging session.
    """
    ids_file_path = os.path.join("../logs", "ids.json")
    
    # Load existing IDs
    with open(ids_file_path, 'r') as f:
        ids = json.load(f)
    
    # Update the finished flag
    if logs_id in ids:
        ids[logs_id]["finished"] = True
        if additional_info is not None:
            ids[logs_id]["additional_info"] = additional_info
    else:
        print(f"Warning: logs_id {logs_id} not found in ids.json")
    
    # Save back to ids.json
    with open(ids_file_path, 'w') as f:
        json.dump(ids, f, indent=4)
    
    print(f"Marked logs_id {logs_id} as finished in {ids_file_path}")


def log_training_metadata(logs_id, env_type, training_metadata):
    """
    Logs the metadata for the training session.

    Args:
        logs_id (str): Unique ID for the logging session.
        env_type (str): "myopic" or "proactive"
        training_metadata (dict): Hyperparameters, environment configuration, and other setup details.
    """
    log_file_path = os.path.join("../logs", "training", f"training_{logs_id}.json")

    # Check if the file exists and load existing data if it does
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_data = json.load(log_file)
    else:
        log_data = {}

    # Append new data to the existing data
    if env_type not in log_data:
        log_data[env_type] = {
            "training_id": logs_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "metadata": training_metadata,
            "episodes": {}
        }
    else:
        # If the env_type already exists, update the metadata and created_at
        log_data[env_type]["metadata"] = training_metadata
        log_data[env_type]["created_at"] = datetime.utcnow().isoformat() + "Z"
    # Convert log_data to a serializable format before saving
    serializable_log_data = convert_to_serializable(log_data)
    # Save the updated data back to the file
    with open(log_file_path, 'w') as log_file:
        json.dump(serializable_log_data, log_file, indent=4, cls=NumpyEncoder)   

    print(f"Training metadata logged to {log_file_path}")

def log_training_episode(logs_id, env_type, episode_number, episode_data):
    """
    Logs data for a single episode during training.

    Args:
        logs_id (str): Unique ID for the logging session.
        env_type (str): "myopic" or "proactive"
        episode_number (int): The episode number being logged.
        episode_data (dict): Detailed data about the episode (rewards, actions, scenarios, etc.).
    """
    log_file_path = os.path.join("../logs", "training", f"training_{logs_id}.json")

    # Load existing log file
    with open(log_file_path, 'r') as log_file:
        log_data = json.load(log_file)

    # Add episode data
    log_data[env_type]["episodes"][f"{episode_number}"] = episode_data

    # Save back to log file
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4, cls=NumpyEncoder)

    print(f"Episode {episode_number} logged to {log_file_path}")


def finalize_training_log(logs_id, summary_data, model_save_path):
    """
    Marks the training log as finished and adds summary data.

    Args:
        logs_id (str): Unique ID for the logging session.
        summary_data (dict): Final statistics and summary of the training session.
    """
    log_file_path = os.path.join("../logs", "training", f"training_{logs_id}.json")

    # Load existing log file
    with open(log_file_path, 'r') as log_file:
        log_data = json.load(log_file)

    # Update with summary and mark as finished
    log_data["completed_at"] = datetime.utcnow().isoformat() + "Z"
    log_data["summary"] = summary_data
    log_data["model_save_path"] = model_save_path
    # Save back to log file
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

    # Update the ids.json to mark as finished
    mark_log_as_finished(logs_id, model_save_path)

    print(f"Training log finalized and summary added to {log_file_path}")


def convert_to_serializable(obj):
    """
    Converts various non-JSON-serializable objects into JSON-serializable formats.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):  # Handle datetime objects
        return obj.isoformat()
    elif isinstance(obj, str):
        return obj
    else:
        # Fallback for unknown types
        return str(obj)




def get_config_variables(config_module):
    """
    Extracts all non-special, non-private variables from the config module
    and ensures JSON-serializability.
    """
    config_vars = {
        key: value for key, value in vars(config_module).items()
        if not key.startswith("__") and not callable(value)  # Exclude magic methods and functions
    }
    return convert_to_serializable(config_vars)

def log_inference_metadata(inference_id, metadata):
    """
    Logs metadata from an inference run to the specified log file.
    
    Args:
        inference_id (str): Unique ID for the inference session
        metadata (dict): Dictionary containing metadata about the inference run
    """
    # Load existing log data if file exists
    log_file_path = os.path.join("../logs", "inference", f"inference_{inference_id}.json")
    if os.path.exists(log_file_path):   
        with open(log_file_path, 'r') as log_file:
            log_data = json.load(log_file)
    else:
        log_data = {}

    # Convert metadata to serializable format
    serializable_metadata = convert_to_serializable(metadata)
    
    # Add timestamp
    serializable_metadata['logged_at'] = datetime.utcnow().isoformat() + "Z"
    
    # Update log data with inference metadata
    log_data['inference_metadata'] = serializable_metadata

    # Save back to log file
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

    print(f"Inference metadata logged to {log_file_path}")


def find_corresponding_training_id(model_path, env_type):
    """
    Find the training ID corresponding to a model path and environment type by searching through ids.json.
    
    Args:
        model_path (str): Path to the model
        env_type (str): Type of environment ('myopic' or 'proactive')
        
    Returns:
        str: Training ID if found, None otherwise
    """
    with open('../logs/ids.json', 'r') as f:
        ids = json.load(f)
        
    # Search through all entries
    for id_num, details in ids.items():
        if details.get("additional_info") == model_path:
            # Check if training file exists
            training_file = f'../logs/training/training_{id_num}.json'
            if os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    training_data = json.load(f)
                    # Check if matches the environment type in metadata
                    if training_data.get('metadata', {}).get('myopic_or_proactive') == env_type:
                        return id_num
                        
    return None




def log_inference_scenario_data(inference_id, scenario_data):
    """
    Logs scenario-level data for inference.
    """
    log_file_path = os.path.join("../logs", "inference", f"inference_{inference_id}.json")

    # Load existing log file or initialize a new log
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_data = json.load(log_file)
    else:
        log_data = {"inference_id": inference_id, "scenarios": {}}  # Initialize with "scenarios" key

    # Ensure "scenarios" key exists
    if "scenarios" not in log_data:
        log_data["scenarios"] = {}

    # Add scenario data
    scenario_folder = scenario_data["scenario_folder"]
    log_data["scenarios"][scenario_folder] = scenario_data

    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4, cls=NumpyEncoder)
    print(f"Scenario data logged for {scenario_folder} to {log_file_path}")

def update_id_status(inference_id, status):
    """
    Updates the status of an inference ID in ids.json.
    """
    with open('../logs/ids.json', 'r') as f:
        ids = json.load(f)
    ids[inference_id]["finished"] = status
    with open('../logs/ids.json', 'w') as f:
        json.dump(ids, f, indent=4)


from stable_baselines3.common.callbacks import BaseCallback

class DetailedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = {
            'delay_penalty': [],
            'conflict_penalty': [],
            'cancellation_penalty': [],
            'ahead_of_time_penalty': []
        }
        self.episode_metrics = {
            'delay_minutes': [],
            'ahead_of_time_minutes': [],
            'num_conflicts': [],
            'num_cancellations': []
        }
        
    def _on_step(self) -> bool:
        # Get info from last step
        info = self.locals['infos'][0]  # Assuming single environment
        
        # Log rewards
        for reward_type, value in info['rewards'].items():
            self.logger.record(f"rewards/{reward_type}", value)
            self.episode_rewards[reward_type].append(value)
            
        # Log metrics
        for metric, value in info['metrics'].items():
            self.logger.record(f"metrics/{metric}", value)
            self.episode_metrics[metric].append(value)
            
        return True
    
    def _on_rollout_end(self) -> None:
        # Log episode averages
        for reward_type, values in self.episode_rewards.items():
            if values:
                avg_value = sum(values) / len(values)
                self.logger.record(f"episode_rewards/{reward_type}_avg", avg_value)
        
        for metric, values in self.episode_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                self.logger.record(f"episode_metrics/{metric}_avg", avg_value)
                
        # Clear episode data
        for reward_type in self.episode_rewards:
            self.episode_rewards[reward_type] = []
        for metric in self.episode_metrics:
            self.episode_metrics[metric] = []