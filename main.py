"""
Diffusion Policy for Robotic Manipulation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union

import torch
import tyro


@dataclass
class TrainConfig:
    """Train diffusion policy on collected demonstrations."""
    task: Annotated[str, tyro.conf.arg(help="Task name (lift, can, square)")] = "lift"
    data_path: Annotated[str, tyro.conf.arg(help="Path to demonstration data")] = "data/lift.hdf5"
    epochs: Annotated[int, tyro.conf.arg(help="Number of training epochs")] = 100
    batch_size: Annotated[int, tyro.conf.arg(help="Batch size")] = 32
    lr: Annotated[float, tyro.conf.arg(help="Learning rate")] = 1e-4
    hidden_dim: Annotated[int, tyro.conf.arg(help="Hidden dimension")] = 256
    pred_horizon: Annotated[int, tyro.conf.arg(help="Prediction horizon")] = 16
    obs_horizon: Annotated[int, tyro.conf.arg(help="Observation horizon (frames of history)")] = 4
    n_layers: Annotated[int, tyro.conf.arg(help="Number of transformer layers")] = 16
    n_diffusion_steps: Annotated[int, tyro.conf.arg(help="Number of diffusion steps")] = 100
    num_workers: Annotated[int, tyro.conf.arg(help="Number of data loading workers")] = 4
    checkpoint_dir: Annotated[str, tyro.conf.arg(help="Directory to save checkpoints")] = "checkpoints"
    modal: Annotated[bool, tyro.conf.arg(help="Run training on Modal serverless GPUs")] = False
    interleave_cross_attn: Annotated[bool, tyro.conf.arg(help="Interleave cross-attention layers")] = True


@dataclass
class EvaluateConfig:
    """Evaluate trained policy."""
    checkpoint: Annotated[str, tyro.conf.arg(help="Path to model checkpoint")]
    task: Annotated[str, tyro.conf.arg(help="Task name (lift, can, square)")] = "lift"
    n_episodes: Annotated[int, tyro.conf.arg(help="Number of evaluation episodes")] = 50
    render: Annotated[bool, tyro.conf.arg(help="Render during evaluation")] = False
    save_videos: Annotated[bool, tyro.conf.arg(help="Save evaluation videos")] = False
    video_dir: Annotated[str, tyro.conf.arg(help="Directory to save videos")] = "eval_videos"
    n_videos: Annotated[int, tyro.conf.arg(help="Number of videos to save")] = 5
    output_dir: Annotated[str, tyro.conf.arg(help="Directory to save evaluation results")] = "eval_results"
    debug: Annotated[bool, tyro.conf.arg(help="Enable debug logging for action tracing")] = False
    modal: Annotated[bool, tyro.conf.arg(help="Run evaluation on Modal serverless GPUs")] = False


def train_policy(config: TrainConfig):
    """Train diffusion policy on collected demonstrations."""
    
    if config.modal:
        import subprocess
        print("=" * 50)
        print("TRAINING ON MODAL (Serverless GPU)")
        print("=" * 50)
        
        cmd = [
            "modal", "run", "train_modal.py",
            "--epochs", str(config.epochs),
            "--batch-size", str(config.batch_size),
            "--lr", str(config.lr),
            "--upload-data",
            "--data-path", config.data_path,
        ]
        subprocess.run(cmd, check=True)
        return
    
    from data.dataset import RobomimicDataset
    from torch.utils.data import DataLoader
    from policy import DiffusionPolicy
    
    print("=" * 50)
    print("TRAINING")
    print("=" * 50)
    print(f"Data path: {config.data_path}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print("=" * 50)
    
    print("Using RobomimicDataset loader...")
    from torch.utils.data import random_split
    
    full_dataset = RobomimicDataset(
        config.data_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        normalize=True,
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    dataset = full_dataset
    
    policy = DiffusionPolicy(
        hidden_dim=config.hidden_dim,
        action_dim=dataset.action_dim,
        obs_dim=dataset.obs_dim,
        action_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        n_layers=config.n_layers,
        n_diffusion_steps=config.n_diffusion_steps,
        interleave_cross_attn=config.interleave_cross_attn,
    )
    
    print(f"\nModel config:")
    print(f"  Observation dim: {dataset.obs_dim}")
    print(f"  Observation horizon: {config.obs_horizon}")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Transformer layers: {config.n_layers}")
    print(f"  Prediction horizon: {config.pred_horizon}")
    print(f"  Diffusion steps: {config.n_diffusion_steps}")
    print(f"  Interleave cross-attn: {config.interleave_cross_attn}")
    print(f"  Total samples: {len(dataset)}")
    print()
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"policy_{config.task}_best.pt"
    
    history = policy.train(
        dataloader, 
        epochs=config.epochs, 
        lr=config.lr,
        val_dataloader=val_dataloader,
        checkpoint_path=best_checkpoint_path
    )
    
    policy.save(checkpoint_dir / f"policy_{config.task}.pt")
    
    if history['val_losses']:
        print(f"\nBest validation loss: {history['best_val_loss']:.6f}")
        print(f"Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"Final val loss: {history['val_losses'][-1]:.6f}")
    
    dataset.save_normalizer(str(checkpoint_dir / f"normalizer_{config.task}.npz"))
    
    print("\nTraining complete!")


def evaluate_policy(config: EvaluateConfig):
    """Evaluate trained policy."""
    
    if config.modal:
        import subprocess
        print("=" * 50)
        print("EVALUATION ON MODAL (Serverless GPU)")
        print("=" * 50)
        
        cmd = [
            "modal", "run", "eval_modal.py",
            "--task", config.task,
            "--n-episodes", str(config.n_episodes),
        ]
        if config.debug:
            cmd.append("--debug")
        if config.save_videos:
            cmd.extend(["--save-videos", "--n-videos", str(config.n_videos)])
        
        subprocess.run(cmd, check=True)
        return
    
    from policy import DiffusionPolicy
    from data.dataset import load_normalizer
    from envs.robosuite_wrapper import create_env
    from eval.evaluate import PolicyEvaluator, visualize_results
    
    print("=" * 50)
    print("EVALUATION")
    print("=" * 50)
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Task: {config.task}")
    print(f"Episodes: {config.n_episodes}")
    print("=" * 50)
    
    policy = DiffusionPolicy.load(config.checkpoint)
    
    normalizer_path = Path(config.checkpoint).parent / f"normalizer_{config.task}.npz"
    if not normalizer_path.exists():
        normalizer_path = Path(config.checkpoint).parent / "normalizer.npz"
    
    if normalizer_path.exists():
        normalizer = load_normalizer(str(normalizer_path))
        print(f"Loaded normalizer from {normalizer_path}")
    else:
        print("Warning: No normalizer found, using unnormalized observations")
        normalizer = None
    
    env = create_env(task=config.task, use_images=config.save_videos, render=config.render)
    
    evaluator = PolicyEvaluator(
        env=env,
        policy=policy,
        normalizer=normalizer,
        action_horizon=8,
        pred_horizon=16,
        obs_horizon=4,
        debug=config.debug,
    )
    
    metrics, results = evaluator.evaluate_n_episodes(
        n_episodes=config.n_episodes,
        render=config.render,
        save_videos=config.save_videos,
        video_dir=config.video_dir,
        n_videos=config.n_videos,
    )
    
    evaluator.print_metrics(metrics)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_metrics(metrics, str(output_dir / "metrics.json"))
    visualize_results(results, save_path=str(output_dir / "evaluation_results.png"))
    
    env.close()
    print("\nEvaluation complete!")

Commands = Union[
    Annotated[TrainConfig, tyro.conf.subcommand("train")],
    Annotated[EvaluateConfig, tyro.conf.subcommand("evaluate")],
]

def main():
    config = tyro.cli(Commands)
    
    if isinstance(config, TrainConfig):
        train_policy(config)
    elif isinstance(config, EvaluateConfig):
        evaluate_policy(config)

if __name__ == "__main__":
    main()
