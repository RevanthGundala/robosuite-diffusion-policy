"""
Robosuite environment wrapper with gym-style interface.
"""

import logging
import numpy as np

# Suppress robosuite's excessive INFO logging
logging.getLogger('robosuite').setLevel(logging.WARNING)

import robosuite as suite
from robosuite.wrappers import GymWrapper



class RobosuiteWrapper:
    """
    Wrapper for robosuite environments with standardized interface.
    
    Supports both state-based and image-based observations.
    """
    
    def __init__(
        self,
        env_name: str = "Lift",
        robot: str = "Panda",
        use_camera_obs: bool = False,
        camera_names: list = None,
        camera_height: int = 84,
        camera_width: int = 84,
        horizon: int = 500,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        reward_shaping: bool = True,
    ):
        """
        Initialize robosuite environment.
        
        Args:
            env_name: Name of the environment (e.g., "Lift", "Stack", "PickPlace")
            robot: Robot type (e.g., "Panda", "Sawyer", "IIWA")
            use_camera_obs: Whether to include camera observations
            camera_names: List of camera names to use
            camera_height: Height of camera images
            camera_width: Width of camera images
            horizon: Maximum episode length
            has_renderer: Whether to use on-screen rendering
            has_offscreen_renderer: Whether to use off-screen rendering (required for camera obs)
            reward_shaping: Whether to use dense reward shaping
        """
        self.env_name = env_name
        self.use_camera_obs = use_camera_obs
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        
        print(f"Creating {env_name} environment (this may take a moment)...")
        
        # Suppress robosuite controller warnings about missing robot parts
        import warnings
        warnings.filterwarnings('ignore', message='.*controller.*robot does not have this component.*')
        
        # Controller config matching robomimic data collection (composite format for robosuite 1.5+)
        # This is critical - the actions are in OSC_POSE space with specific gains
        controller_config = {
            "type": "BASIC",
            "body_parts": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    "damping": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [0, 300],
                    "damping_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2,
                    "gripper": {
                        "type": "GRIP"
                    }
                }
            }
        }
        
        # Create environment with object observations enabled
        # This is needed to get the 'object' observation key
        self.env = suite.make(
            env_name,
            robots=robot,
            has_renderer=has_renderer,
            has_offscreen_renderer=use_camera_obs,  # Only if using cameras
            use_camera_obs=use_camera_obs,
            use_object_obs=True,  # Enable object observations for 'object' key
            camera_names=self.camera_names if use_camera_obs else None,
            camera_heights=camera_height,
            camera_widths=camera_width,
            horizon=horizon,
            reward_shaping=reward_shaping,
            control_freq=20,  # 20 Hz control - must match training data
            controller_configs=controller_config,
        )
        
        print("Environment created successfully!")
        
        self.horizon = horizon
        self._step_count = 0
        
        # Get action dimensions
        self.action_dim = self.env.action_dim
        self.action_low, self.action_high = self.env.action_spec
        
        # Determine observation dimensions
        self._obs_keys = self._get_obs_keys()
        self.obs_dim = self._compute_obs_dim()
        
        print(f"Environment: {env_name}")
        print(f"Action dim: {self.action_dim}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Observation keys: {self._obs_keys}")
        
    def _get_obs_keys(self) -> list:
        """Get observation keys matching robomimic low_dim format.
        
        Note: Robomimic HDF5 uses 'object' but live robosuite env uses 'object-state'.
        We map between them in _extract_obs().
        """
        # These are the keys we want in the output observation (matching HDF5 format)
        return [
            'robot0_eef_pos',      # 3 - end effector position
            'robot0_eef_quat',     # 4 - end effector orientation
            'robot0_gripper_qpos', # 2 - gripper state
            'object-state',        # 14 - object state (called 'object' in HDF5, 'object-state' in live env)
        ]
    
    def _compute_obs_dim(self) -> int:
        """Compute total observation dimension matching robomimic format."""
        # Get a sample observation to determine dimensions
        obs_dict = self.env.reset()
        total_dim = 0
        for key in self._obs_keys:
            if key in obs_dict:
                total_dim += obs_dict[key].shape[0]
            else:
                print(f"Warning: key '{key}' not found in env obs. Available: {list(obs_dict.keys())}")
        return total_dim
    
    def _extract_obs(self, obs_dict: dict) -> dict:
        """
        Extract observations matching robomimic low_dim format.
        
        Concatenates: robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object
        This matches the format used in training data from converted HDF5 files.
        
        Returns:
            dict with:
                - 'state': observation vector matching training format
                - 'images': dict of camera images (if use_camera_obs)
                - 'gripper_pos': end-effector position
        """
        result = {}
        
        # Concatenate observation keys to match training data format
        obs_list = []
        for key in self._obs_keys:
            if key in obs_dict:
                obs_list.append(obs_dict[key])
        
        state = np.concatenate(obs_list)
        result["state"] = state
        
        # Extract end-effector position for reference
        result["gripper_pos"] = obs_dict.get("robot0_eef_pos", np.zeros(3))
        
        # Extract images if using camera observations
        if self.use_camera_obs:
            result["images"] = {}
            for cam_name in self.camera_names:
                img_key = f"{cam_name}_image"
                if img_key in obs_dict:
                    result["images"][cam_name] = obs_dict[img_key]
        
        return result
    
    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self._step_count = 0
        obs_dict = self.env.reset()
        return self._extract_obs(obs_dict)
    
    def step(self, action: np.ndarray) -> tuple:
        """
        Take a step in the environment.
        
        Args:
            action: Action array of shape (action_dim,)
            
        Returns:
            obs: Observation dict
            reward: Scalar reward
            done: Whether episode is done
            info: Additional info dict
        """
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        obs_dict, reward, done, info = self.env.step(action)
        self._step_count += 1
        
        # Check for success (task-specific)
        info["success"] = self._check_success(obs_dict)
        info["step"] = self._step_count
        
        obs = self._extract_obs(obs_dict)
        
        return obs, reward, done, info
    
    def _check_success(self, obs_dict: dict) -> bool:
        """Check if task is successfully completed."""
        return self.env._check_success()
    
    def render(self):
        """Render the environment."""
        self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_object_pos(self, obs: dict) -> np.ndarray:
        """Get object position from observation."""
        return obs["object"][:3]
    
    def get_gripper_pos(self, obs: dict) -> np.ndarray:
        """Get gripper (end-effector) position from observation."""
        return obs.get("gripper_pos")
    
    @property
    def action_space_low(self) -> np.ndarray:
        return self.action_low
    
    @property
    def action_space_high(self) -> np.ndarray:
        return self.action_high


def create_lift_env(
    use_images: bool = False,
    render: bool = False,
) -> RobosuiteWrapper:
    """
    Convenience function to create a Lift environment.
    
    Args:
        use_images: Whether to use image observations
        render: Whether to enable on-screen rendering
        
    Returns:
        RobosuiteWrapper instance
    """
    return RobosuiteWrapper(
        env_name="Lift",
        robot="Panda",
        use_camera_obs=use_images,
        has_renderer=render,
        has_offscreen_renderer=use_images,  # Only if using cameras
        horizon=500,
    )


def create_can_env(
    use_images: bool = False,
    render: bool = False,
) -> RobosuiteWrapper:
    """
    Convenience function to create a PickPlaceCan environment.
    
    Args:
        use_images: Whether to use image observations
        render: Whether to enable on-screen rendering
        
    Returns:
        RobosuiteWrapper instance
    """
    return RobosuiteWrapper(
        env_name="PickPlaceCan",
        robot="Panda",
        use_camera_obs=use_images,
        has_renderer=render,
        has_offscreen_renderer=use_images,
        horizon=500,
    )


def create_square_env(
    use_images: bool = False,
    render: bool = False,
) -> RobosuiteWrapper:
    """
    Convenience function to create a NutAssemblySquare environment.
    
    Args:
        use_images: Whether to use image observations
        render: Whether to enable on-screen rendering
        
    Returns:
        RobosuiteWrapper instance
    """
    return RobosuiteWrapper(
        env_name="NutAssemblySquare",
        robot="Panda",
        use_camera_obs=use_images,
        has_renderer=render,
        has_offscreen_renderer=use_images,
        horizon=500,
    )


def create_env(
    task: str = "lift",
    use_images: bool = False,
    render: bool = False,
) -> RobosuiteWrapper:
    """
    Factory function to create environment by task name.
    
    Args:
        task: Task name ('lift', 'can', 'square')
        use_images: Whether to use image observations
        render: Whether to enable on-screen rendering
        
    Returns:
        RobosuiteWrapper instance
    """
    env_creators = {
        "lift": create_lift_env,
        "can": create_can_env,
        "square": create_square_env,
    }
    
    if task not in env_creators:
        raise ValueError(f"Unknown task: {task}. Available: {list(env_creators.keys())}")
    
    return env_creators[task](use_images=use_images, render=render)
