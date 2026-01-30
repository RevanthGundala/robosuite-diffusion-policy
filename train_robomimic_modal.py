"""
Train robomimic's Diffusion Policy on Modal (GPU).

Usage:
    uv run modal run train_robomimic_modal.py --epochs 100
"""

import modal
from pathlib import Path

app = modal.App("robomimic-diffusion-policy")

volume = modal.Volume.from_name("diffusion-policy-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", 
        "libgl1-mesa-glx", 
        "libosmesa6-dev", 
        "libglfw3", 
        "patchelf",
        "libglib2.0-0",  # Required for cv2/opencv
        "libsm6",
        "libxrender1",
        "libxext6",
    )
    .pip_install(
        "torch>=2.0.0",
        "numpy<2.0",  # robomimic needs numpy < 2
        "h5py",
        "tqdm",
        "termcolor",
        "tensorboard",
        "tensorboardX",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "robosuite>=1.5.0",
        "mujoco>=3.0.0",
        "huggingface_hub==0.23.4",
        "transformers==4.41.2",
        "diffusers==0.11.1",
        "psutil",
    )
    .run_commands(
        # Clone robomimic
        "git clone https://github.com/ARISE-Initiative/robomimic.git /opt/robomimic",
        # Remove egl_probe from setup.py
        "sed -i 's/\"egl_probe>=1.0.1\",/# egl_probe removed/' /opt/robomimic/setup.py",
        # Create and run a Python script to patch env_robosuite.py properly
        "echo 'import re' > /tmp/patch.py",
        "echo 'with open(\"/opt/robomimic/robomimic/envs/env_robosuite.py\", \"r\") as f: content = f.read()' >> /tmp/patch.py",
        "echo 'content = content.replace(\"import egl_probe\", \"egl_probe = None  # patched out\")' >> /tmp/patch.py",
        "echo 'content = content.replace(\"egl_probe.get_available_devices()\", \"list(range(8))\")' >> /tmp/patch.py",
        "echo 'with open(\"/opt/robomimic/robomimic/envs/env_robosuite.py\", \"w\") as f: f.write(content)' >> /tmp/patch.py",
        "python /tmp/patch.py",
        "pip install -e /opt/robomimic",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={"/data": volume},
)
def train_robomimic_diffusion_policy(task: str = "lift", epochs: int = 100):
    """Train robomimic's diffusion policy on specified task."""
    import subprocess
    import sys
    import os
    import json
    import torch
    import shutil
    from datetime import datetime
    
    print("=" * 60)
    print(f"ROBOMIMIC DIFFUSION POLICY TRAINING ON MODAL - Task: {task}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    robomimic_path = "/opt/robomimic"
    
    # Check if converted dataset exists, if not convert it
    converted_path = f"/data/{task}_converted.hdf5"
    raw_path = f"/data/{task}.hdf5"
    
    if not os.path.exists(converted_path):
        print(f"\nConverting {task} dataset to robomimic format...")
        convert_script = f"{robomimic_path}/robomimic/scripts/dataset_states_to_obs.py"
        subprocess.run([
            sys.executable, convert_script,
            "--dataset", raw_path,
            "--output_name", converted_path,
            "--done_mode", "2",
        ], check=True)
        volume.commit()
    
    # Create custom config with our settings
    config_template = f"{robomimic_path}/robomimic/exps/templates/diffusion_policy.json"
    with open(config_template, 'r') as f:
        config = json.load(f)
    
    # Use unique experiment name with task to avoid overwrite prompt
    exp_name = f"diffusion_policy_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"/data/robomimic_trained_models_{task}"
    
    # Clean up old runs to avoid prompt
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed old output directory: {output_dir}")
    
    # Update config
    config["experiment"]["name"] = exp_name
    config["train"]["num_epochs"] = epochs
    config["train"]["data"] = [{"path": converted_path}]
    config["train"]["output_dir"] = output_dir
    # Disable rollouts during training - they require rendering which fails on Modal
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["rollout"]["rate"] = 999999  # Set to very high number to effectively disable
    config["experiment"]["rollout"]["n"] = 0
    config["experiment"]["save"]["every_n_epochs"] = 50
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    
    # Save custom config
    custom_config_path = "/data/diffusion_policy_config.json"
    with open(custom_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nExperiment name: {exp_name}")
    print(f"Config saved to {custom_config_path}")
    print(f"Training for {epochs} epochs...")
    
    # Run training
    train_script = f"{robomimic_path}/robomimic/scripts/train.py"
    subprocess.run([
        sys.executable, train_script,
        "--config", custom_config_path,
    ], check=True)
    
    volume.commit()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}/{exp_name}/")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
)
def evaluate_robomimic(task: str = "lift", checkpoint_path: str = None, n_episodes: int = 50):
    """Evaluate the trained robomimic model."""
    import os
    import glob
    import numpy as np
    
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.torch_utils as TorchUtils
    
    # Find checkpoint if not specified
    if checkpoint_path is None:
        model_dir = f"/data/robomimic_trained_models_{task}"
        # Find latest run
        runs = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])
        if runs:
            latest_run = os.path.join(model_dir, runs[-1])
            
            # robomimic saves checkpoints in models/ subdirectory
            models_dir = os.path.join(latest_run, "models")
            if os.path.exists(models_dir):
                # Find best or latest checkpoint
                checkpoint_path = os.path.join(models_dir, "model_best_training.pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(models_dir, "model_epoch_100.pth")
                if not os.path.exists(checkpoint_path):
                    # Find any checkpoint
                    ckpts = glob.glob(os.path.join(models_dir, "model_epoch_*.pth"))
                    if ckpts:
                        checkpoint_path = sorted(ckpts)[-1]  # Get latest epoch
            else:
                # Fallback to old path format
                checkpoint_path = os.path.join(latest_run, "model_best.pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(latest_run, "last.pth")
        
        # Debug: List what files exist
        print(f"Model directory: {model_dir}")
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                print(f"  Found: {os.path.join(root, f)}")
    
    print("=" * 60)
    print("ROBOMIMIC DIFFUSION POLICY EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    # Load policy
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device)
    
    # Create environment
    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
    )
    
    successes = []
    
    for ep in range(n_episodes):
        policy.start_episode()
        obs = env.reset()
        done = False
        step = 0
        success = False
        
        while not done and step < 400:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            step += 1
            if "success" in info and info["success"]:
                success = True
                break
        
        successes.append(success)
        status = "✓" if success else "✗"
        running_rate = np.mean(successes) * 100
        print(f"  Ep {ep+1:3d}: {status} | Steps: {step:3d} | Success Rate: {running_rate:.1f}%")
    
    env.close()
    
    final_rate = np.mean(successes) * 100
    print("\n" + "=" * 60)
    print(f"FINAL SUCCESS RATE: {final_rate:.1f}%")
    print("=" * 60)
    
    return {"success_rate": final_rate, "n_episodes": n_episodes}


@app.local_entrypoint()
def main(
    task: str = "lift",
    epochs: int = 100,
    eval_only: bool = False,
    n_episodes: int = 50,
):
    if not eval_only:
        print(f"Training robomimic Diffusion Policy on {task} for {epochs} epochs on Modal...")
        train_robomimic_diffusion_policy.remote(task=task, epochs=epochs)
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    result = evaluate_robomimic.remote(task=task, n_episodes=n_episodes)
    print(f"\nResult: {result}")
