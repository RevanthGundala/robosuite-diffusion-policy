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
    """Wrapper for robosuite environments with standardized interface."""
    
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
        self.env_name = env_name
        self.use_camera_obs = use_camera_obs
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        
        print(f"Creating {env_name} environment (this may take a moment)...")
        
        import warnings
        warnings.filterwarnings('ignore', message='.*controller.*robot does not have this component.*')
        
        # Controller config matching robomimic data collection (OSC_POSE with specific gains)
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
        
        self.env = suite.make(
            env_name,
            robots=robot,
            has_renderer=has_renderer,
            has_offscreen_renderer=use_camera_obs,
            use_camera_obs=use_camera_obs,
            use_object_obs=True,
            camera_names=self.camera_names if use_camera_obs else None,
            camera_heights=camera_height,
            camera_widths=camera_width,
            horizon=horizon,
            reward_shaping=reward_shaping,
            control_freq=20,
            controller_configs=controller_config,
        )
        
        print("Environment created successfully!")
        
        self.horizon = horizon
        self._step_count = 0
        
        self.action_dim = self.env.action_dim
        self.action_low, self.action_high = self.env.action_spec
        
        self._obs_keys = self._get_obs_keys()
        self.obs_dim = self._compute_obs_dim()
        
        print(f"Environment: {env_name}")
        print(f"Action dim: {self.action_dim}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Observation keys: {self._obs_keys}")
        
    def _get_obs_keys(self) -> list:
        """Get observation keys matching robomimic low_dim format.
        
        Note: Robomimic HDF5 uses 'object' but live robosuite env uses 'object-state'.
        """
        return [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'object-state',
        ]
    
    def _compute_obs_dim(self) -> int:
        """Compute total observation dimension."""
        obs_dict = self.env.reset()
        total_dim = 0
        for key in self._obs_keys:
            if key in obs_dict:
                total_dim += obs_dict[key].shape[0]
            else:
                print(f"Warning: key '{key}' not found in env obs. Available: {list(obs_dict.keys())}")
        return total_dim
    
    def _extract_obs(self, obs_dict: dict) -> dict:
        """Extract observations matching robomimic low_dim format."""
        result = {}
        
        obs_list = []
        for key in self._obs_keys:
            if key in obs_dict:
                obs_list.append(obs_dict[key])
        
        state = np.concatenate(obs_list)
        result["state"] = state
        result["gripper_pos"] = obs_dict.get("robot0_eef_pos", np.zeros(3))
        
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
        """Take a step in the environment."""
        action = np.clip(action, self.action_low, self.action_high)
        
        obs_dict, reward, done, info = self.env.step(action)
        self._step_count += 1
        
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


def create_lift_env(use_images: bool = False, render: bool = False) -> RobosuiteWrapper:
    """Create a Lift environment."""
    return RobosuiteWrapper(
        env_name="Lift",
        robot="Panda",
        use_camera_obs=use_images,
        has_renderer=render,
        has_offscreen_renderer=use_images,
        horizon=500,
    )


def create_can_env(use_images: bool = False, render: bool = False) -> RobosuiteWrapper:
    """Create a PickPlaceCan environment."""
    return RobosuiteWrapper(
        env_name="PickPlaceCan",
        robot="Panda",
        use_camera_obs=use_images,
        has_renderer=render,
        has_offscreen_renderer=use_images,
        horizon=500,
    )


def create_square_env(use_images: bool = False, render: bool = False) -> RobosuiteWrapper:
    """Create a NutAssemblySquare environment."""
    return RobosuiteWrapper(
        env_name="NutAssemblySquare",
        robot="Panda",
        use_camera_obs=use_images,
        has_renderer=render,
        has_offscreen_renderer=use_images,
        horizon=500,
    )


def create_env(task: str = "lift", use_images: bool = False, render: bool = False) -> RobosuiteWrapper:
    """Factory function to create environment by task name."""
    env_creators = {
        "lift": create_lift_env,
        "can": create_can_env,
        "square": create_square_env,
    }
    
    if task not in env_creators:
        raise ValueError(f"Unknown task: {task}. Available: {list(env_creators.keys())}")
    
    return env_creators[task](use_images=use_images, render=render)
