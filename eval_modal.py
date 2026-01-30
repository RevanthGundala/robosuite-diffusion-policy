"""
Modal serverless GPU evaluation for Diffusion Policy.
Run with: modal run eval_modal.py --task can --n-episodes 10 --debug
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("diffusion-policy-eval")

# Use the same volume as training
volume = modal.Volume.from_name("diffusion-policy-vol", create_if_missing=True)

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libosmesa6")  # git for robosuite, others for MuJoCo headless
    .pip_install(
        "torch>=2.0.0",
        "diffusers>=0.36.0",
        "numpy>=1.24.0",
        "h5py>=3.9.0",
        "tqdm>=4.65.0",
        "mujoco>=3.0.0",
        "gymnasium>=0.29.0",
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.8",
    )
    .add_local_file(
        Path(__file__).parent / "data" / "dataset.py",
        remote_path="/root/diffusion-policy/data/dataset.py",
    )
    .add_local_file(
        Path(__file__).parent / "policy.py",
        remote_path="/root/diffusion-policy/policy.py",
    )
    .add_local_file(
        Path(__file__).parent / "envs" / "robosuite_wrapper.py",
        remote_path="/root/diffusion-policy/envs/robosuite_wrapper.py",
    )
    .add_local_file(
        Path(__file__).parent / "eval" / "evaluate.py",
        remote_path="/root/diffusion-policy/eval/evaluate.py",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hour timeout for 50 episodes
    volumes={"/data": volume},
)
def evaluate_on_modal(
    task: str = "can",
    checkpoint_name: str = None,
    n_episodes: int = 10,
    debug: bool = False,
    save_videos: bool = False,
    n_videos: int = 3,
):
    """Evaluate diffusion policy on Modal GPU."""
    import sys
    import os
    
    os.environ["MUJOCO_GL"] = "osmesa"
    
    sys.path.insert(0, "/root/diffusion-policy")
    
    import torch
    import json
    import numpy as np
    from policy import DiffusionPolicy
    from data.dataset import load_normalizer
    from envs.robosuite_wrapper import create_env
    from eval.evaluate import PolicyEvaluator, visualize_results
    
    print("=" * 60)
    print(f"EVALUATION ON MODAL GPU - Task: {task}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    checkpoint_dir = Path("/data/checkpoints")
    if checkpoint_name:
        checkpoint_path = checkpoint_dir / checkpoint_name
    else:
        checkpoint_path = checkpoint_dir / f"policy_{task}_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / f"policy_{task}.pt"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"Available checkpoints: {list(checkpoint_dir.glob('*.pt'))}")
        return {"error": "checkpoint not found"}
    
    print(f"Checkpoint: {checkpoint_path}")
    
    policy = DiffusionPolicy.load(str(checkpoint_path), device="cuda" if torch.cuda.is_available() else "cpu")
    
    normalizer_path = checkpoint_dir / f"normalizer_{task}.npz"
    if not normalizer_path.exists():
        normalizer_path = checkpoint_dir / "normalizer.npz"
    
    if normalizer_path.exists():
        normalizer = load_normalizer(str(normalizer_path))
        print(f"Loaded normalizer from {normalizer_path}")
    else:
        print("WARNING: No normalizer found!")
        normalizer = None
    
    print(f"\nCreating {task} environment...")
    env = create_env(task=task, use_images=save_videos, render=False)
    
    evaluator = PolicyEvaluator(
        env=env,
        policy=policy,
        normalizer=normalizer,
        action_horizon=8,
        pred_horizon=16,
        obs_horizon=4,
        debug=debug,
    )
    
    print(f"\nRunning {n_episodes} episodes...")
    metrics, results = evaluator.evaluate_n_episodes(
        n_episodes=n_episodes,
        render=False,
        save_videos=save_videos,
        video_dir="/data/eval_videos",
        n_videos=n_videos,
    )
    
    evaluator.print_metrics(metrics)
    
    output_dir = Path("/data/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in metrics.items()}
    with open(output_dir / f"metrics_{task}.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    volume.commit()
    
    env.close()
    print("\nEvaluation complete!")
    
    return metrics


@app.local_entrypoint()
def main(
    task: str = "can",
    checkpoint: str = None,
    n_episodes: int = 10,
    debug: bool = False,
    save_videos: bool = False,
    n_videos: int = 3,
    download_results: bool = True,
):
    """Local entrypoint for Modal evaluation."""
    print(f"Starting evaluation on Modal for task: {task}...")
    
    metrics = evaluate_on_modal.remote(
        task=task,
        checkpoint_name=checkpoint,
        n_episodes=n_episodes,
        debug=debug,
        save_videos=save_videos,
        n_videos=n_videos,
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
    else:
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print(f"Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"Avg Steps: {metrics['avg_steps']:.1f}")
    
    if download_results:
        print("\nDownloading results from Modal volume...")
        import subprocess
        subprocess.run([
            "modal", "volume", "get", "diffusion-policy-vol", 
            "eval_results", "eval_results_modal/"
        ], check=False)
        print("Results saved to eval_results_modal/")
