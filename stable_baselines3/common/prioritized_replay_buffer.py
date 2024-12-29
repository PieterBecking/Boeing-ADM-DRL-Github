import numpy as np
import torch as th
from typing import Any, Dict, List, Optional, Union, Tuple
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

class PrioritizedReplayBuffer(DictReplayBuffer):
    """
    Prioritized Replay Buffer that emphasizes positive reward outliers.
    Extends DictReplayBuffer to handle dictionary observation spaces.
    
    This implementation uses a simple priority scheme based on reward values,
    giving higher sampling probability to transitions with higher rewards.
    
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space (must be a Dict space)
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable memory efficient variant
    :param handle_timeout_termination: Handle timeout termination separately
    :param alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
    :param beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
    :param epsilon: Small constant to add to priorities to ensure non-zero probabilities
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        
        self.priorities = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)  # Initialize with ones instead of zeros
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.min_priority = self.epsilon
        
    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add a new transition to the buffer with maximum priority."""
        super().add(obs, next_obs, action, reward, done, infos)
        
        pos = self.pos - 1 if self.pos > 0 else self.buffer_size - 1
        
        # New transitions get max priority with some noise to ensure exploration
        self.priorities[pos] = np.clip(self.max_priority + np.random.normal(0, 0.01, self.n_envs), 
                                     self.min_priority, 
                                     self.max_priority)
        
        # Update priorities based on reward values
        # We want to prioritize positive rewards and especially positive outliers
        if np.any(reward > 0):  # Handle vectorized environments
            # Calculate z-score of the reward if we have enough samples
            if self.full:
                reward_mean = np.mean(self.rewards[:self.pos])
                reward_std = np.std(self.rewards[:self.pos]) + 1e-6  # avoid division by zero
                z_scores = (reward - reward_mean) / reward_std
                
                # Clip z-scores to prevent overflow
                z_scores = np.clip(z_scores, -10, 10)
                
                # Increase priority for positive outliers (z_score > 1)
                outlier_mask = z_scores > 1
                if np.any(outlier_mask):
                    # Calculate new priorities with numerical stability
                    new_priorities = np.clip(
                        self.max_priority * (1 + z_scores[outlier_mask] * 0.1),  # Reduce impact of z-scores
                        self.min_priority,
                        100.0  # Hard cap on maximum priority
                    )
                    self.priorities[pos, outlier_mask] = new_priorities
                    self.max_priority = min(100.0, float(np.max(new_priorities)))

    def _get_probabilities(self, batch_inds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get sampling probabilities and importance sampling weights."""
        # Get priorities for the requested indices
        priorities = np.clip(self.priorities[batch_inds], self.min_priority, self.max_priority)
        
        # Convert priorities to probabilities with numerical stability
        probs = priorities ** self.alpha
        prob_sum = np.sum(probs) + 1e-8  # Add small constant to prevent division by zero
        probs = probs / prob_sum
        
        # Importance sampling weights with numerical stability
        N = self.size()
        weights = (N * probs + 1e-8) ** (-self.beta)
        weights = weights / (np.max(weights) + 1e-8)  # Normalize weights
        
        return probs, weights
        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """Sample elements from the replay buffer with priorities."""
        upper_bound = self.buffer_size if self.full else self.pos
        
        # Sample indices based on priorities with numerical stability
        priorities = np.clip(self.priorities[:upper_bound], self.min_priority, self.max_priority) ** self.alpha
        probs = priorities / (np.sum(priorities) + 1e-8)
        
        # Flatten and normalize probabilities
        probs = probs.flatten()
        probs = probs / (np.sum(probs) + 1e-8)
        
        # Sample indices
        try:
            batch_inds = np.random.choice(upper_bound, size=batch_size, p=probs)
        except ValueError:
            # Fallback to uniform sampling if there's an issue with probabilities
            batch_inds = np.random.choice(upper_bound, size=batch_size)
        
        # Get importance sampling weights with numerical stability
        weights = (upper_bound * probs[batch_inds] + 1e-8) ** (-self.beta)
        weights = weights / (np.max(weights) + 1e-8)
        
        # Get samples
        samples = self._get_samples(batch_inds, env=env)
        
        # Convert weights to tensor and add to samples
        weights = th.FloatTensor(weights.reshape(-1, 1)).to(self.device)
        
        # Return samples with importance sampling weights
        return DictReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards * weights  # Apply importance sampling weights to rewards
        )

    def update_priorities(self, batch_inds: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for the given batch indices."""
        priorities = np.clip(priorities + self.epsilon, self.min_priority, 100.0)  # Hard cap on maximum priority
        self.priorities[batch_inds] = priorities
        self.max_priority = min(100.0, float(np.max(priorities))) 