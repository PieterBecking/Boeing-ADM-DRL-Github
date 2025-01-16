import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import preprocess_obs
from typing import Dict, List, Tuple
from src.environment import AircraftDisruptionEnv
from scripts.utils import load_scenario_data

MODEL_PATH = "trained_models/dqn/6ac-700-diverse/1025/proactive-94.zip"
SCENARIO_FOLDER = "data/Testing/6ac-700-diverse/deterministic_na_Scenario_005"
ENV_TYPE = "proactive"
SAVE_FOLDER = "005"

class QValueWrapper(torch.nn.Module):
    def __init__(self, q_network):
        super().__init__()
        self.q_network = q_network
        
    def forward(self, x):
        # Convert flat tensor back to dictionary format
        batch_size = x.shape[0]
        state_dim = 448
        action_mask_dim = 147
        
        # Split the input tensor into state and action mask
        states = x[:, :state_dim]
        action_masks = x[:, state_dim:state_dim + action_mask_dim]
        
        # Create the observation dictionary
        obs_dict = {
            "state": states,
            "action_mask": action_masks
        }
        
        # Get Q-values
        q_values = self.q_network(obs_dict)
        return q_values

def prepare_environment(scenario_folder: str, env_type: str) -> Tuple[AircraftDisruptionEnv, int, int]:
    """Prepare the environment and get observation dimensions."""
    # Load scenario data
    data_dict = load_scenario_data(scenario_folder)
    
    # Create environment
    env = AircraftDisruptionEnv(
        data_dict['aircraft'],
        data_dict['flights'],
        data_dict['rotations'],
        data_dict['alt_aircraft'],
        data_dict['config'],
        env_type=env_type
    )
    
    # Get dimensions
    state_dim = env.observation_space['state'].shape[0]
    action_mask_dim = env.observation_space['action_mask'].shape[0]
    
    return env, state_dim, action_mask_dim

def collect_states(env: AircraftDisruptionEnv, num_samples: int = 80) -> torch.Tensor:
    """Collect representative states from the environment."""
    states = []
    action_masks = []
    
    for _ in range(num_samples):
        obs, _ = env.reset()
        states.append(obs['state'])
        action_masks.append(obs['action_mask'])
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(np.array(states))
    action_masks_tensor = torch.FloatTensor(np.array(action_masks))
    
    # Combine into flat tensor
    flat_states = torch.cat([states_tensor, action_masks_tensor], dim=1)
    
    return flat_states

def compute_feature_importance(model, states, feature_names, state_dim, save_folder):
    """Compute feature importance using gradients."""
    model.train()  # Enable gradients
    
    # Convert states to require gradients
    states = states.clone().detach().requires_grad_(True)
    
    # Get Q-values
    q_values = model(states)
    
    # Initialize importance scores
    importance_scores = torch.zeros((states.shape[1], q_values.shape[1]))
    
    # Compute gradients for each action
    for action in range(q_values.shape[1]):
        # Zero gradients
        if states.grad is not None:
            states.grad.zero_()
        
        # Compute gradients
        q_values[:, action].sum().backward(retain_graph=True)
        
        # Get absolute gradients and take mean across batch
        importance_scores[:, action] = states.grad.abs().mean(dim=0)
    
    # Convert to numpy
    importance_scores = importance_scores.detach().numpy()
    
    # Create output directory 
    os.makedirs(f"shap_plots/{save_folder}", exist_ok=True)
    
    # Only consider state features (exclude action mask features)
    state_importance = importance_scores[:state_dim]
    
    # Create descriptive feature names based on environment structure
    ROWS_STATE_SPACE = 7  # From environment constants
    COLUMNS_STATE_SPACE = 64  # From environment constants
    
    state_feature_names = []
    
    # Fill up to state_dim with names
    current_idx = 0
    for row in range(ROWS_STATE_SPACE):
        if current_idx >= state_dim:
            break
            
        if row == 0:
            state_feature_names.extend([
                "Current Time",
                "Time Until End"
            ])
            current_idx += 2
        else:
            aircraft_idx = row - 1
            # Aircraft info
            state_feature_names.extend([
                f"Aircraft_{aircraft_idx}_ID",
                f"Aircraft_{aircraft_idx}_Breakdown_Prob",
                f"Aircraft_{aircraft_idx}_Unavail_Start",
                f"Aircraft_{aircraft_idx}_Unavail_End"
            ])
            current_idx += 4
            
            # Flight info (in groups of 3)
            remaining_cols = (state_dim - current_idx) // row  # Distribute remaining columns
            num_flights = min((COLUMNS_STATE_SPACE - 4) // 3, remaining_cols // 3)
            
            for flight_idx in range(num_flights):
                if current_idx + 3 > state_dim:
                    break
                state_feature_names.extend([
                    f"Aircraft_{aircraft_idx}_Flight_{flight_idx}_ID",
                    f"Aircraft_{aircraft_idx}_Flight_{flight_idx}_DepTime",
                    f"Aircraft_{aircraft_idx}_Flight_{flight_idx}_ArrTime"
                ])
                current_idx += 3
    
    # If we still haven't reached state_dim, pad with generic names
    while len(state_feature_names) < state_dim:
        state_feature_names.append(f"State_{len(state_feature_names)}")
    
    print(f"Number of state features: {len(state_feature_names)}")
    print(f"State dimension: {state_dim}")
    
    # Create overall state feature importance plot
    mean_state_importance = state_importance.mean(axis=1)
    top_indices = np.argsort(mean_state_importance)[-20:][::-1]
    
    print(f"Top indices max: {max(top_indices)}")
    
    # Ensure we don't exceed the number of features
    top_indices = [i for i in top_indices if i < len(state_feature_names)]
    n_features = min(20, len(top_indices))
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), mean_state_importance[top_indices[:n_features]])
    plt.yticks(range(n_features), [state_feature_names[i] for i in top_indices[:n_features]])
    plt.xlabel("Mean Gradient Magnitude")
    plt.title("Top Most Important State Features (Averaged Across All Actions)")
    plt.tight_layout()
    plt.savefig(f"shap_plots/{save_folder}/overall_state_importance.png")
    plt.close()
    
    # Create heatmap with descriptive names
    plt.figure(figsize=(20, 10))
    n_top_features = min(30, len(top_indices))
    top_features = np.argsort(mean_state_importance)[-n_top_features:][::-1]
    plt.imshow(state_importance[top_features].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Gradient Magnitude')
    plt.yticks(range(q_values.shape[1]), [f"Action {i}" for i in range(q_values.shape[1])])
    plt.xticks(range(n_top_features), [state_feature_names[i] for i in top_features], rotation=45, ha='right')
    plt.title("State Feature Importance Heatmap")
    plt.tight_layout()
    plt.savefig(f"shap_plots/{save_folder}/state_importance_heatmap.png")
    plt.close()
    
    return importance_scores, mean_state_importance

def run_shap_analysis(model_path: str, scenario_folder: str, env_type: str, save_folder: str):
    """Run SHAP analysis on the trained DQN model."""
    print(f"\nRunning SHAP analysis for model: {model_path}")
    print(f"Environment type: {env_type}\n")
    
    # Load the model
    model = DQN.load(model_path)
    
    # Create Q-value wrapper
    q_wrapper = QValueWrapper(model.policy.q_net)
    
    print("Collecting representative states...")
    # Prepare environment and get dimensions
    env, state_dim, action_mask_dim = prepare_environment(scenario_folder, env_type)
    print(f"\nState dimension: {state_dim}")
    print(f"Action mask dimension: {action_mask_dim}\n")
    
    # Collect states
    flat_states = collect_states(env)
    print(f"Flat states shape: {flat_states.shape}")
    print(f"Flat states min/max: {flat_states.min():.4f}/{flat_states.max():.4f}\n")
    
    # Create feature names
    feature_names = []
    for i in range(state_dim):
        feature_names.append(f"State_{i}")
    for i in range(action_mask_dim):
        feature_names.append(f"Mask_{i}")
    
    # Compute feature importance
    print("Computing feature importance...")
    importance_scores, mean_state_importance = compute_feature_importance(q_wrapper, flat_states, feature_names, state_dim, save_folder)
    
    # Save results
    np.save(f"shap_plots/{save_folder}/importance_scores.npy", importance_scores)
    np.save(f"shap_plots/{save_folder}/mean_state_importance.npy", mean_state_importance)
    np.save(f"shap_plots/{save_folder}/state_feature_names.npy", np.array(feature_names[:state_dim]))

if __name__ == "__main__":
    run_shap_analysis(MODEL_PATH, SCENARIO_FOLDER, ENV_TYPE, SAVE_FOLDER)  