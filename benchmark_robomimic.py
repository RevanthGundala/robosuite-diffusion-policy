"""
Run robomimic's Diffusion Policy benchmark on Modal (Linux).
This bypasses the macOS egl-probe build issue.

Usage:
    uv run modal run benchmark_robomimic.py --epochs 100
"""

import modal
from pathlib import Path

app = modal.App("robomimic-benchmark")

volume = modal.Volume.from_name("diffusion-policy-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")  # robomimic works better with 3.10
    .apt_install("git", "libgl1-mesa-glx", "libosmesa6-dev", "libglfw3")
    .pip_install(
        "robomimic",
        "robosuite",
        "torch>=2.0.0",
        "numpy",
        "h5py",
        "tqdm",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
)
def train_robomimic_diffusion_policy(epochs: int = 100):
    """Train robomimic's diffusion policy on Lift task."""
    import robomimic
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.config import config_factory
    from robomimic.scripts.train import train
    import torch
    
    print("=" * 60)
    print("ROBOMIMIC DIFFUSION POLICY BENCHMARK")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # First, convert dataset to robomimic format if needed
    import subprocess
    
    # Check if converted dataset exists
    converted_path = "/data/lift_low_dim.hdf5"
    raw_path = "/data/lift.hdf5"
    
    import os
    if not os.path.exists(converted_path):
        print("Converting dataset to robomimic format...")
        subprocess.run([
            "python", "-m", "robomimic.scripts.dataset_states_to_obs",
            "--dataset", raw_path,
            "--output_name", converted_path,
            "--done_mode", "2",
        ], check=True)
    
    # Create diffusion policy config
    config = config_factory(algo_name="diffusion_policy")
    
    # Set config
    config.experiment.name = "diffusion_policy_lift_benchmark"
    config.train.data = [{"path": converted_path}]
    config.train.output_dir = "/data/robomimic_output"
    config.train.num_epochs = epochs
    config.train.batch_size = 64
    
    # Diffusion policy specific settings
    config.algo.horizon.observation_horizon = 2
    config.algo.horizon.prediction_horizon = 16
    config.algo.horizon.action_horizon = 8
    config.algo.ddpm.enabled = True
    config.algo.ddim.enabled = False
    
    # Get device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    train(config, device=device)
    
    volume.commit()
    print("\nTraining complete!")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
)
def eval_robomimic_diffusion_policy(checkpoint_path: str, n_episodes: int = 50):
    """Evaluate robomimic's trained diffusion policy."""
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import numpy as np
    import torch
    
    print("=" * 60)
    print("ROBOMIMIC DIFFUSION POLICY EVALUATION")
    print("=" * 60)
    
    # Load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path)
    policy.start_episode()
    
    # Create environment
    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(env_meta, render=False)
    
    successes = []
    rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        success = False
        
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if info.get("success", False):
                success = True
        
        successes.append(success)
        rewards.append(total_reward)
        status = "✓" if success else "✗"
        print(f"  Ep {ep+1}: {status} | Reward: {total_reward:.2f} | Success Rate: {np.mean(successes)*100:.1f}%")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("=" * 60)
    
    return {
        "method": "robomimic Diffusion Policy",
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "n_episodes": n_episodes,
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    eval_only: bool = False,
    n_episodes: int = 50,
):
    if not eval_only:
        print("Training robomimic Diffusion Policy on Modal...")
        train_robomimic_diffusion_policy.remote(epochs=epochs)
    
    # Find latest checkpoint
    # Note: You'd need to specify the actual checkpoint path after training
    # checkpoint_path = "/data/robomimic_output/models/model_epoch_100.pth"
    # result = eval_robomimic_diffusion_policy.remote(checkpoint_path, n_episodes)
    # print(f"Result: {result}")
