"""
PyTorch Dataset for loading robosuite demonstrations.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple


class RobomimicDataset(Dataset):
    """
    PyTorch Dataset for loading robomimic demonstration data from HDF5 files.
    
    Compatible with robomimic's low_dim format which stores observations
    in a different structure than our custom format.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        pred_horizon: int = 16,
        obs_horizon: int = 4,
        action_horizon: int = 8,
        normalize: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            hdf5_path: Path to robomimic HDF5 file with demonstrations
            pred_horizon: Number of future actions to predict
            obs_horizon: Number of past observations to stack
            action_horizon: Number of actions to execute (subset of pred_horizon)
            normalize: Whether to normalize observations and actions
        """
        self.hdf5_path = Path(hdf5_path)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.normalize = normalize
        
        # Load data
        self._load_data()
        
        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_stats()
        
        # Create indices for sampling
        self._create_indices()
        
        print(f"Loaded {len(self.demos)} demonstrations")
        print(f"Total samples: {len(self.indices)}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
    
    def _load_data(self):
        """Load demonstration data from HDF5 file.
        
        Supports two formats:
        1. Robomimic converted format (recommended): uses 'obs' group with semantic observations
        2. Raw format (legacy): uses 'states' with raw MuJoCo state
        
        For robomimic format, we concatenate these observation keys:
        - robot0_eef_pos (3) - end effector position
        - robot0_eef_quat (4) - end effector orientation
        - robot0_gripper_qpos (2) - gripper position
        - object (14) - object state
        Total: 23 dims (or varies based on available keys)
        """
        self.demos = []
        
        # Define which obs keys to use (robomimic low_dim format)
        self.obs_keys = [
            'robot0_eef_pos',      # 3 - end effector position
            'robot0_eef_quat',     # 4 - end effector orientation
            'robot0_gripper_qpos', # 2 - gripper state
            'object',              # 14 - object state
        ]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get sorted list of demo keys
            demo_keys = sorted(
                [k for k in f['data'].keys() if k.startswith('demo_')],
                key=lambda x: int(x.split('_')[1])
            )
            
            # Check if this is robomimic format (has 'obs' group)
            first_demo = f[f'data/{demo_keys[0]}']
            self.use_robomimic_obs = 'obs' in first_demo
            
            if self.use_robomimic_obs:
                # Filter to only available obs keys
                available_keys = list(first_demo['obs'].keys())
                self.obs_keys = [k for k in self.obs_keys if k in available_keys]
                print(f"Using robomimic obs format with keys: {self.obs_keys}")
            else:
                print("Using raw states format (legacy)")
            
            for demo_key in demo_keys:
                demo_grp = f[f'data/{demo_key}']
                
                if self.use_robomimic_obs:
                    # Concatenate selected observation keys
                    obs_list = [demo_grp['obs'][k][:] for k in self.obs_keys]
                    observations = np.concatenate(obs_list, axis=1)
                else:
                    # Legacy: use raw states
                    observations = demo_grp['states'][:]
                
                # Load actions
                actions = demo_grp['actions'][:]
                
                demo = {
                    'observations': observations,
                    'actions': actions,
                    'episode_length': len(observations),
                }
                self.demos.append(demo)
        
        # Set dimensions from first demo
        self.obs_dim = self.demos[0]['observations'].shape[1]
        self.action_dim = self.demos[0]['actions'].shape[1]
    
    def _compute_stats(self):
        """Compute normalization statistics for observations and actions.
        
        Uses min-max normalization to [-1, 1] range for actions, which is crucial
        for diffusion models since the noise schedule assumes bounded targets.
        Observations use z-score normalization with minimum std of 0.1 to prevent
        extreme normalized values for near-constant dimensions.
        """
        all_obs = np.concatenate([d['observations'] for d in self.demos], axis=0)
        all_actions = np.concatenate([d['actions'] for d in self.demos], axis=0)
        
        # Observations: z-score normalization (mean/std)
        # Use minimum std of 0.1 to prevent extreme normalized values
        # for dimensions that are nearly constant in training data
        self.obs_mean = all_obs.mean(axis=0)
        self.obs_std = np.maximum(all_obs.std(axis=0), 0.1)
        
        # Actions: min-max normalization to [-1, 1]
        # This is critical for diffusion models!
        self.action_min = all_actions.min(axis=0)
        self.action_max = all_actions.max(axis=0)
        
        # Log dimensions with small variance (for debugging)
        small_std_dims = np.where(all_obs.std(axis=0) < 0.1)[0]
        if len(small_std_dims) > 0:
            print(f"Note: {len(small_std_dims)} obs dims had std < 0.1, clamped to 0.1: {small_std_dims[:10]}...")
    
    def _create_indices(self):
        """
        Create indices for sampling.
        
        Each index is (demo_idx, start_timestep) where start_timestep
        is the beginning of a valid chunk.
        """
        self.indices = []
        
        for demo_idx, demo in enumerate(self.demos):
            T = demo['episode_length']
            
            # We need at least obs_horizon steps before and pred_horizon steps after
            for t in range(self.obs_horizon - 1, T - self.pred_horizon + 1):
                self.indices.append((demo_idx, t))
    
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation with clamping to [-5, 5]."""
        if self.normalize:
            normalized = (obs - self.obs_mean) / self.obs_std
            return np.clip(normalized, -5.0, 5.0)
        return obs
    
    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize observation."""
        if self.normalize:
            return obs * self.obs_std + self.obs_mean
        return obs
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] range using min-max normalization.
        
        Formula: normalized = 2 * (action - min) / (max - min) - 1
        """
        if self.normalize:
            # Add small epsilon to avoid division by zero for constant dimensions
            range_val = self.action_max - self.action_min + 1e-6
            return 2.0 * (action - self.action_min) / range_val - 1.0
        return action
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action from [-1, 1] range back to original scale.
        
        Formula: action = (normalized + 1) / 2 * (max - min) + min
        """
        if self.normalize:
            range_val = self.action_max - self.action_min + 1e-6
            return (action + 1.0) / 2.0 * range_val + self.action_min
        return action
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            obs: Observation tensor of shape (obs_horizon, obs_dim) - preserves temporal structure
            actions: Action tensor of shape (pred_horizon, action_dim)
        """
        demo_idx, t = self.indices[idx]
        demo = self.demos[demo_idx]
        
        # Get observation(s) - always return (obs_horizon, obs_dim) shape
        if self.obs_horizon == 1:
            obs = demo['observations'][t:t+1]  # (1, obs_dim)
        else:
            # Stack past observations, preserving temporal structure
            start = t - self.obs_horizon + 1
            obs = demo['observations'][start:t+1]  # (obs_horizon, obs_dim)
        
        # Get future actions
        actions = demo['actions'][t:t + self.pred_horizon]
        
        # Pad if necessary
        if len(actions) < self.pred_horizon:
            pad_length = self.pred_horizon - len(actions)
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (pad_length, 1))
            ], axis=0)
        
        # Normalize
        obs = self.normalize_obs(obs)
        actions = self.normalize_action(actions)
        
        return torch.FloatTensor(obs), torch.FloatTensor(actions)
    
    def get_normalizer(self) -> dict:
        """Get normalization parameters for use during inference."""
        return {
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'action_min': self.action_min,
            'action_max': self.action_max,
        }
    
    def save_normalizer(self, path: str):
        """Save normalization parameters to file."""
        np.savez(
            path,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            action_min=self.action_min,
            action_max=self.action_max,
        )
        print(f"Saved normalizer to {path}")


def load_normalizer(path: str) -> dict:
    """Load normalization parameters from file."""
    data = np.load(path)
    return {
        'obs_mean': data['obs_mean'],
        'obs_std': data['obs_std'],
        'action_mean': data['action_mean'],
        'action_std': data['action_std'],
        'action_min': data['action_min'],
        'action_max': data['action_max'],
    }


