"""
Modal serverless GPU training for Diffusion Policy.
Run with: uv run diffusion-policy train --modal
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("diffusion-policy")

# Create a volume for checkpoints and data
volume = modal.Volume.from_name("diffusion-policy-vol", create_if_missing=True)

# Define the image with all dependencies and copy local code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "diffusers>=0.36.0",
        "numpy>=1.24.0",
        "h5py>=3.9.0",
        "tqdm>=4.65.0",
    )
    .add_local_file(
        Path(__file__).parent / "data" / "dataset.py",
        remote_path="/root/diffusion-policy/data/dataset.py",
    )
    .add_local_file(
        Path(__file__).parent / "policy.py",
        remote_path="/root/diffusion-policy/policy.py",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
)
def train_on_modal(
    task: str = "lift",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    hidden_dim: int = 512,
    pred_horizon: int = 16,
    obs_horizon: int = 4,
    n_layers: int = 16,
    n_diffusion_steps: int = 100,
    interleave_cross_attn: bool = True,
):
    """Train diffusion policy on Modal GPU using existing code."""
    import sys
    sys.path.insert(0, "/root/diffusion-policy")
    
    import torch
    from torch.utils.data import DataLoader, random_split
    # Import directly to avoid __init__.py which imports collect.py
    from data.dataset import RobomimicDataset
    from policy import DiffusionPolicy
    
    print("=" * 50)
    print(f"TRAINING ON MODAL GPU - Task: {task}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    
    # Load dataset based on task
    # Use converted HDF5 files which have proper robomimic obs format
    data_path = f"/data/{task}_converted.hdf5"
    dataset = RobomimicDataset(
        data_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        normalize=True,
    )
    
    # Split into train/val (90/10)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create policy
    policy = DiffusionPolicy(
        hidden_dim=hidden_dim,
        action_dim=dataset.action_dim,
        obs_dim=dataset.obs_dim,
        action_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        n_layers=n_layers,
        n_diffusion_steps=n_diffusion_steps,
        interleave_cross_attn=interleave_cross_attn,
        device="cuda",
    )
    
    print(f"\nModel config:")
    print(f"  Observation dim: {dataset.obs_dim}")
    print(f"  Observation horizon: {obs_horizon}")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Transformer layers: {n_layers}")
    print(f"  Prediction horizon: {pred_horizon}")
    print(f"  Diffusion steps: {n_diffusion_steps}")
    print(f"  Interleave cross-attn: {interleave_cross_attn}")
    print(f"  Train samples: {train_size}, Val samples: {val_size}")
    print()
    
    # Checkpoint path for best model
    checkpoint_dir = Path("/data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"policy_{task}_best.pt"
    normalizer_path = checkpoint_dir / f"normalizer_{task}.npz"
    
    # Save normalizer BEFORE training so it's available even if training interrupted
    dataset.save_normalizer(str(normalizer_path))
    volume.commit()
    print(f"Saved normalizer (will be used for evaluation)")
    
    # Train with validation and best checkpoint saving
    policy.train(
        train_dataloader, 
        epochs=epochs, 
        lr=lr,
        val_dataloader=val_dataloader,
        checkpoint_path=str(checkpoint_path),
    )
    
    # Also save the final model
    policy.save(checkpoint_dir / f"policy_{task}.pt")
    # Resave normalizer (in case anything changed)
    dataset.save_normalizer(str(normalizer_path))
    
    volume.commit()
    print(f"\nTraining complete! Checkpoint saved to Modal volume as policy_{task}.pt")


@app.local_entrypoint()
def main(
    task: str = "lift",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    hidden_dim: int = 512,
    obs_horizon: int = 4,
    n_layers: int = 16,
    upload_data: bool = False,
    data_path: str = None,
    interleave_cross_attn: bool = True,
):
    """Local entrypoint for Modal training."""
    # Default data path based on task - use converted format
    if data_path is None:
        data_path = f"data/{task}_converted.hdf5"
    
    if upload_data:
        print(f"Uploading {data_path} to Modal...")
        with volume.batch_upload() as batch:
            # Store as task_converted.hdf5 to match the remote path
            batch.put_file(data_path, f"{task}_converted.hdf5")
        print("Data uploaded!")
    
    print(f"Starting training on Modal for task: {task}...")
    train_on_modal.remote(
        task=task,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        obs_horizon=obs_horizon,
        n_layers=n_layers,
        interleave_cross_attn=interleave_cross_attn,
    )
    print("Training complete!")
