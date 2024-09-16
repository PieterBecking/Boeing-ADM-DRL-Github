import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp

class MaskedMlpExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that applies a mask to the observation
    and removes the dummy values (NaNs) before passing it to the rest of the network.
    """
    def __init__(self, observation_space, features_dim=256):
        print("now doing init of custom extractor")
        super(MaskedMlpExtractor, self).__init__(observation_space, features_dim)
        
        # Define the MLP structure after masking
        self.mlp = nn.Sequential(
            *create_mlp(observation_space.shape[0], features_dim, [128, 128])  # Example hidden layers
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        print("Original Observations in Forward Pass:", observations)
        
        # Create a mask where valid values are 1 and NaN values are 0
        mask = (~th.isnan(observations)).float()  # Mask valid values (1 if valid, 0 if NaN)
        
        # Replace NaN with 0 (neutral value)
        masked_obs = th.nan_to_num(observations, nan=0.0) * mask

        # Debug: Check for any remaining NaNs after masking (should be none)
        if th.isnan(masked_obs).any():
            print("NaN values detected after masking!")
        
        print("Masked Observations in Forward Pass:", masked_obs)
        
        # Forward pass through the network after masking
        return self.mlp(masked_obs)


class CustomMaskedDQNPolicy(DQNPolicy):
    """
    Custom DQN Policy that uses the MaskedMlpExtractor to ignore NaN values.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        print("now doing init of custom policy")

        # Initialize the custom feature extractor
        features_extractor_class = MaskedMlpExtractor
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {"features_dim": 256})

        # Initialize the DQNPolicy with the custom extractor
        super(CustomMaskedDQNPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs
        )
        
        # Print statement to confirm the correct feature extractor is used
        print(f"Using custom feature extractor: {self.features_extractor}")
