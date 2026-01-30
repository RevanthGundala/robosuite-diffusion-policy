"""
Test to compare training data observations vs simulator observations.
Run with: modal run debug_obs_mismatch.py
"""

import modal
from pathlib import Path

app = modal.App("debug-obs-mismatch")
volume = modal.Volume.from_name("diffusion-policy-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libosmesa6")
    .pip_install(
        "numpy>=1.24.0",
        "h5py>=3.9.0",
        "mujoco>=3.0.0",
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={"/data": volume},
)
def compare_obs_and_actions():
    """Compare training data format vs simulator format using robomimic obs."""
    import os
    os.environ["MUJOCO_GL"] = "osmesa"
    
    import numpy as np
    import h5py
    import robosuite as suite
    
    print("=" * 70)
    print("COMPARING TRAINING DATA vs SIMULATOR (Robomimic Format)")
    print("=" * 70)
    
    # Observation keys to use for training data
    train_obs_keys = [
        'robot0_eef_pos',      # 3
        'robot0_eef_quat',     # 4
        'robot0_gripper_qpos', # 2
        'object',              # 14
    ]
    
    # Observation keys for the live environment (different naming!)
    sim_obs_keys = [
        'robot0_eef_pos',      # 3
        'robot0_eef_quat',     # 4
        'robot0_gripper_qpos', # 2
        'object-state',        # 14 (different name in live env!)
    ]
    
    # Load training data from CONVERTED file
    with h5py.File("/data/can_converted.hdf5", "r") as f:
        demo_key = "demo_0"
        
        # Concatenate obs keys
        obs_list = [f[f"data/{demo_key}/obs/{k}"][:] for k in train_obs_keys]
        train_obs = np.concatenate(obs_list, axis=1)
        train_actions = f[f"data/{demo_key}/actions"][:]
        
        print(f"\n--- TRAINING DATA (from can_converted.hdf5) ---")
        print(f"Obs shape: {train_obs.shape}")
        print(f"Actions shape: {train_actions.shape}")
        print(f"\nFirst observation (step 0):")
        print(f"  Length: {len(train_obs[0])}")
        print(f"  robot0_eef_pos (0:3): {train_obs[0, 0:3]}")
        print(f"  robot0_eef_quat (3:7): {train_obs[0, 3:7]}")
        print(f"  robot0_gripper_qpos (7:9): {train_obs[0, 7:9]}")
        print(f"  object (9:23): {train_obs[0, 9:23]}")
        print(f"  Min: {train_obs[0].min():.4f}, Max: {train_obs[0].max():.4f}")
        
        print(f"\nFirst action:")
        print(f"  {train_actions[0]}")
    
    # Create simulator with EXACT settings from HDF5
    print(f"\n--- SIMULATOR (with matching controller config) ---")
    
    # Controller config from HDF5 env_args (composite format for robosuite 1.5+)
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
    
    env = suite.make(
        "PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,  # Important! Need this for 'object' key
        horizon=500,
        reward_shaping=True,
        control_freq=20,
        controller_configs=controller_config,
    )
    
    obs_dict = env.reset()
    
    print(f"\nAvailable obs keys: {list(obs_dict.keys())}")
    
    # Extract observations - use sim_obs_keys (with 'object-state')
    sim_obs_list = [obs_dict[k] for k in sim_obs_keys]
    sim_obs = np.concatenate(sim_obs_list)
    
    print(f"\nAvailable obs keys: {list(obs_dict.keys())}")
    print(f"\nSimulator observation (step 0):")
    print(f"  Length: {len(sim_obs)}")
    print(f"  robot0_eef_pos (0:3): {sim_obs[0:3]}")
    print(f"  robot0_eef_quat (3:7): {sim_obs[3:7]}")
    print(f"  robot0_gripper_qpos (7:9): {sim_obs[7:9]}")
    print(f"  object (9:23): {sim_obs[9:23]}")
    print(f"  Min: {sim_obs.min():.4f}, Max: {sim_obs.max():.4f}")
    
    # Compare dimensions
    print(f"\n--- COMPARISON ---")
    print(f"Training obs dim: {train_obs.shape[1]}")
    print(f"Simulator obs dim: {len(sim_obs)}")
    
    if train_obs.shape[1] != len(sim_obs):
        print(f"⚠️  DIMENSION MISMATCH!")
    else:
        print(f"✓ Dimensions match!")
    
    # Check action space
    print(f"\n--- ACTION SPACE ---")
    print(f"Training action dim: {train_actions.shape[1]}")
    print(f"Env action_dim: {env.action_dim}")
    
    # Replay full demonstration
    print(f"\n--- REPLAYING FULL DEMONSTRATION ---")
    env.reset()
    
    total_reward = 0
    success = False
    for i in range(len(train_actions)):
        action = train_actions[i]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if env._check_success():
            success = True
            print(f"  ✓ SUCCESS at step {i}!")
            break
        if done:
            print(f"  Episode ended at step {i}")
            break
    
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Success: {success}")
    
    # Replay full demonstration WITH state restoration
    print(f"\n--- REPLAYING WITH STATE RESTORATION ---")
    with h5py.File("/data/can_converted.hdf5", "r") as f:
        # Get the initial state from the demo
        demo_key = "demo_0"
        demo_grp = f[f"data/{demo_key}"]
        
        # Check for state data
        print(f"Demo keys: {list(demo_grp.keys())}")
        if 'states' in demo_grp:
            init_state = demo_grp['states'][0]
            print(f"Found initial state, shape: {init_state.shape}")
        else:
            print("No 'states' key found in demo")
        
        # Check env_args for environment info
        if 'env_args' in f['data'].attrs:
            import json
            env_args = json.loads(f['data'].attrs['env_args'])
            print(f"\nEnvironment args:")
            for k, v in env_args.items():
                print(f"  {k}: {v}")
        else:
            print("No env_args found")
    
    env.close()
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


@app.local_entrypoint()
def main():
    compare_obs_and_actions.remote()
