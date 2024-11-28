# scripts/logger.py

import json
import os
from datetime import datetime


def create_new_id(logs_type):
    # Load the ids from the json file
    with open('../logs/ids.json', 'r') as f:
        ids = json.load(f)

    # Get the latest id
    try:
        latest_id = max(ids.keys())
    except ValueError:
        latest_id = "0000"

    # Increment the latest id by 1
    new_id = str(int(latest_id) + 1).zfill(4)

    # Update the ids in the json file with finished set to false
    ids[new_id] = {"type": logs_type, "finished": False}
    with open('../logs/ids.json', 'w') as f:
        json.dump(ids, f)

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

    log_file_path = os.path.join("../logs", "scenarios", f"scenario_folder_{logs_id}.json")

    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

    print(f"Scenario logged to {log_file_path}")

def mark_log_as_finished(logs_id):
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
    else:
        print(f"Warning: logs_id {logs_id} not found in ids.json")
    
    # Save back to ids.json
    with open(ids_file_path, 'w') as f:
        json.dump(ids, f, indent=4)
    
    print(f"Marked logs_id {logs_id} as finished in {ids_file_path}")
