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
        self.hdf5_path = Path(hdf5_path)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.normalize = normalize
        
        self._load_data()
        
        if self.normalize:
            self._compute_stats()
        
        self._create_indices()
        
        print(f"Loaded {len(self.demos)} demonstrations")
        print(f"Total samples: {len(self.indices)}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
    
    def _load_data(self):
        """Load demonstration data from HDF5 file.
        
        For robomimic format, we concatenate these observation keys:
        - robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_qpos (2), object (14)
        Total: 23 dims (or varies based on available keys)
        """
        self.demos = []
        
        self.obs_keys = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'object',
        ]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_keys = sorted(
                [k for k in f['data'].keys() if k.startswith('demo_')],
                key=lambda x: int(x.split('_')[1])
            )
            
            first_demo = f[f'data/{demo_keys[0]}']
            self.use_robomimic_obs = 'obs' in first_demo
            
            if self.use_robomimic_obs:
                available_keys = list(first_demo['obs'].keys())
                self.obs_keys = [k for k in self.obs_keys if k in available_keys]
                print(f"Using robomimic obs format with keys: {self.obs_keys}")
            else:
                print("Using raw states format (legacy)")
            
            for demo_key in demo_keys:
                demo_grp = f[f'data/{demo_key}']
                
                if self.use_robomimic_obs:
                    obs_list = [demo_grp['obs'][k][:] for k in self.obs_keys]
                    observations = np.concatenate(obs_list, axis=1)
                else:
                    observations = demo_grp['states'][:]
                
                actions = demo_grp['actions'][:]
                
                demo = {
                    'observations': observations,
                    'actions': actions,
                    'episode_length': len(observations),
                }
                self.demos.append(demo)
        
        self.obs_dim = self.demos[0]['observations'].shape[1]
        self.action_dim = self.demos[0]['actions'].shape[1]
    
    def _compute_stats(self):
        """Compute normalization statistics.
        
        Uses min-max normalization to [-1, 1] for actions (critical for diffusion models).
        Uses z-score normalization for observations with minimum std of 0.01.
        """
        all_obs = np.concatenate([d['observations'] for d in self.demos], axis=0)
        all_actions = np.concatenate([d['actions'] for d in self.demos], axis=0)
        
        self.obs_mean = all_obs.mean(axis=0)
        self.obs_std = np.maximum(all_obs.std(axis=0), 0.01)
        
        self.action_min = all_actions.min(axis=0)
        self.action_max = all_actions.max(axis=0)
        
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0) + 1e-6
        
        small_std_dims = np.where(all_obs.std(axis=0) < 0.01)[0]
        if len(small_std_dims) > 0:
            print(f"Note: {len(small_std_dims)} obs dims had std < 0.01, clamped to 0.01: {small_std_dims[:10]}...")
    
    def _create_indices(self):
        """Create indices for sampling valid chunks."""
        self.indices = []
        
        for demo_idx, demo in enumerate(self.demos):
            T = demo['episode_length']
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
        """Normalize action to [-1, 1] range using min-max normalization."""
        if self.normalize:
            range_val = self.action_max - self.action_min + 1e-6
            return 2.0 * (action - self.action_min) / range_val - 1.0
        return action
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action from [-1, 1] range back to original scale."""
        if self.normalize:
            range_val = self.action_max - self.action_min + 1e-6
            return (action + 1.0) / 2.0 * range_val + self.action_min
        return action
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training sample."""
        demo_idx, t = self.indices[idx]
        demo = self.demos[demo_idx]
        
        if self.obs_horizon == 1:
            obs = demo['observations'][t:t+1]
        else:
            start = t - self.obs_horizon + 1
            obs = demo['observations'][start:t+1]
        
        actions = demo['actions'][t:t + self.pred_horizon]
        
        if len(actions) < self.pred_horizon:
            pad_length = self.pred_horizon - len(actions)
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (pad_length, 1))
            ], axis=0)
        
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


