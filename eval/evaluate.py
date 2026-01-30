"""
Policy evaluation utilities.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm


class PolicyEvaluator:
    """Evaluate a trained policy in the environment."""
    
    def __init__(
        self,
        env,
        policy,
        normalizer: Optional[dict] = None,
        action_horizon: int = 8,
        pred_horizon: int = 16,
        obs_horizon: int = 4,
        debug: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            env: Environment wrapper
            policy: Trained DiffusionPolicy
            normalizer: Dict with obs_mean, obs_std, action_mean, action_std
            action_horizon: Number of actions to execute from each prediction
            pred_horizon: Full prediction horizon
            obs_horizon: Number of observation frames for context
            debug: Enable debug logging for action tracing
        """
        self.env = env
        self.policy = policy
        self.normalizer = normalizer
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.debug = debug
        self._debug_printed_normalizer = False
        
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using stored statistics.
        
        Clamps normalized values to [-5, 5] to handle dimensions that had
        near-zero variance during training but vary at evaluation.
        """
        if self.normalizer is not None:
            # Use minimum std of 0.1 to avoid division by tiny values
            std = np.maximum(self.normalizer['obs_std'], 0.1)
            normalized = (obs - self.normalizer['obs_mean']) / std
            # Clamp to reasonable range
            return np.clip(normalized, -5.0, 5.0)
        return obs
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action from [-1, 1] back to original scale.
        
        Uses min-max denormalization matching the dataset normalization.
        """
        if self.normalizer is not None:
            # Min-max denormalization: action = (normalized + 1) / 2 * (max - min) + min
            range_val = self.normalizer['action_max'] - self.normalizer['action_min'] + 1e-6
            return (action + 1.0) / 2.0 * range_val + self.normalizer['action_min']
        return action
    
    def _debug_log_normalizer(self):
        """Print normalizer stats once at the start of evaluation."""
        if not self.debug or self._debug_printed_normalizer:
            return
        self._debug_printed_normalizer = True
        
        print("\n" + "=" * 60)
        print("DEBUG: Normalizer Statistics")
        print("=" * 60)
        if self.normalizer is not None:
            print(f"  action_min: {self.normalizer['action_min']}")
            print(f"  action_max: {self.normalizer['action_max']}")
            print(f"  obs_mean:   {self.normalizer['obs_mean'][:5]}... (first 5)")
            print(f"  obs_std:    {self.normalizer['obs_std'][:5]}... (first 5)")
            # Show dims with very small std (potential problem dims)
            small_std_dims = np.where(self.normalizer['obs_std'] < 0.01)[0]
            if len(small_std_dims) > 0:
                print(f"  WARNING: {len(small_std_dims)} dims have std < 0.01: {small_std_dims[:10]}...")
        else:
            print("  WARNING: No normalizer loaded!")
        print("=" * 60 + "\n")
    
    def _debug_log_replan(self, step: int, replan_num: int, obs_raw: np.ndarray, 
                          obs_norm: np.ndarray, action_raw: np.ndarray, 
                          action_unnorm: np.ndarray, queue_len: int):
        """Log debug info at each replan."""
        if not self.debug:
            return
        
        print(f"\n--- DEBUG: Replan #{replan_num} at step {step} ---")
        print(f"  Observation (raw):        min={obs_raw.min():.4f}, max={obs_raw.max():.4f}, mean={obs_raw.mean():.4f}")
        print(f"  Observation (normalized): min={obs_norm.min():.4f}, max={obs_norm.max():.4f}, mean={obs_norm.mean():.4f}")
        
        # Per-dimension analysis: find extreme normalized values
        # obs_norm shape is (obs_horizon, obs_dim), flatten to find extremes
        obs_norm_flat = obs_norm.flatten() if obs_norm.ndim > 1 else obs_norm
        obs_raw_flat = obs_raw.flatten() if obs_raw.ndim > 1 else obs_raw
        
        # Get the last frame's normalized obs for dimension analysis
        last_frame_norm = obs_norm[-1] if obs_norm.ndim > 1 else obs_norm
        last_frame_raw = obs_raw[-1] if obs_raw.ndim > 1 else obs_raw
        
        # Find top 5 most extreme dimensions
        abs_norm = np.abs(last_frame_norm)
        extreme_dims = np.argsort(abs_norm)[-5:][::-1]  # Top 5, descending
        
        print(f"  Top 5 extreme normalized dims:")
        for dim in extreme_dims:
            raw_val = last_frame_raw[dim]
            norm_val = last_frame_norm[dim]
            if self.normalizer is not None:
                mean_val = self.normalizer['obs_mean'][dim]
                std_val = self.normalizer['obs_std'][dim]
                print(f"    dim {dim:2d}: raw={raw_val:8.4f}, norm={norm_val:8.2f} (mean={mean_val:.4f}, std={std_val:.6f})")
            else:
                print(f"    dim {dim:2d}: raw={raw_val:8.4f}, norm={norm_val:8.2f}")
        
        print(f"  Model output ([-1,1]):    min={action_raw.min():.4f}, max={action_raw.max():.4f}, mean={action_raw.mean():.4f}")
        print(f"  Action (unnormalized):    min={action_unnorm.min():.4f}, max={action_unnorm.max():.4f}, mean={action_unnorm.mean():.4f}")
        print(f"  First action: {action_unnorm[0]}")
        print(f"  Action queue length: {queue_len}")
    
    def evaluate_episode(
        self,
        max_steps: int = 500,
        render: bool = False,
        save_video: bool = False,
    ) -> Tuple[bool, float, int, List[np.ndarray]]:
        """
        Evaluate a single episode.
        
        Args:
            max_steps: Maximum steps per episode
            render: Whether to render
            save_video: Whether to save video frames
            
        Returns:
            success: Whether task was successful
            total_reward: Cumulative reward
            steps: Number of steps taken
            frames: Video frames if save_video=True
        """
        obs = self.env.reset()
        
        # Log normalizer stats once
        self._debug_log_normalizer()
        
        total_reward = 0.0
        steps = 0
        success = False
        frames = []
        replan_count = 0
        
        # Action queue for temporal action chunking
        action_queue = []
        
        # Observation history for obs_horizon > 1
        obs_history = []
        
        while steps < max_steps:
            # Get observation
            obs_state = obs["state"]
            obs_normalized = self.normalize_obs(obs_state)
            
            # Maintain observation history
            obs_history.append(obs_normalized)
            if len(obs_history) > self.obs_horizon:
                obs_history.pop(0)
            
            # Pad history if we don't have enough observations yet
            while len(obs_history) < self.obs_horizon:
                obs_history.insert(0, obs_history[0])
            
            # Stack observations: (obs_horizon, obs_dim)
            obs_stacked = np.stack(obs_history, axis=0)
            obs_tensor = torch.FloatTensor(obs_stacked).unsqueeze(0)  # (1, obs_horizon, obs_dim)
            
            # Get new actions if queue is empty
            if len(action_queue) == 0:
                replan_count += 1
                # Sample action sequence from policy
                with torch.no_grad():
                    action_seq = self.policy.sample(obs_tensor)  # (1, T, action_dim)
                    action_seq_raw = action_seq[0].cpu().numpy()  # (T, action_dim)
                
                # Unnormalize and add to queue
                action_seq_unnorm = self.unnormalize_action(action_seq_raw)
                
                # Debug logging
                self._debug_log_replan(
                    step=steps,
                    replan_num=replan_count,
                    obs_raw=obs_state,
                    obs_norm=obs_stacked,
                    action_raw=action_seq_raw,
                    action_unnorm=action_seq_unnorm,
                    queue_len=min(self.action_horizon, len(action_seq_unnorm)),
                )
                
                # Only use first action_horizon actions
                for i in range(min(self.action_horizon, len(action_seq_unnorm))):
                    action_queue.append(action_seq_unnorm[i])
            
            # Execute next action
            action = action_queue.pop(0)
            obs, reward, done, info = self.env.step(action)
            
            total_reward += reward
            steps += 1
            
            if save_video and "images" in obs:
                frames.append(obs["images"].get("agentview", None))
            
            if render:
                self.env.render()
            
            # Mark success and break early
            if info.get("success", False):
                success = True
                break
            
            if done:
                break
        
        return success, total_reward, steps, frames
    
    def evaluate_n_episodes(
        self,
        n_episodes: int = 50,
        render: bool = False,
        save_videos: bool = False,
        video_dir: str = "eval_videos",
        n_videos: int = 5,
    ) -> Tuple[Dict, List[Dict]]:
        """
        Evaluate policy over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render
            save_videos: Whether to save videos
            video_dir: Directory to save videos
            n_videos: Number of videos to save
            
        Returns:
            metrics: Dict with success_rate, avg_reward, avg_steps
            results: List of per-episode results
        """
        results = []
        successes = []
        
        if save_videos:
            video_path = Path(video_dir)
            video_path.mkdir(parents=True, exist_ok=True)
        
        for ep in tqdm(range(n_episodes), desc="Evaluating"):
            save_video = save_videos and ep < n_videos
            
            success, reward, steps, frames = self.evaluate_episode(
                render=render,
                save_video=save_video,
            )
            
            results.append({
                "episode": ep,
                "success": success,
                "reward": reward,
                "steps": steps,
            })
            successes.append(success)
            
            # Print live results
            success_rate = np.mean(successes) * 100
            print(f"  Ep {ep+1}: {'✓' if success else '✗'} | Reward: {reward:.2f} | Steps: {steps} | Success Rate: {success_rate:.1f}%")
            
            # Save video immediately
            if save_video and frames:
                self._save_video(frames, video_path / f"episode_{ep}.mp4")
        
        # Compute metrics
        rewards = [r["reward"] for r in results]
        steps_list = [r["steps"] for r in results]
        
        metrics = {
            "n_episodes": n_episodes,
            "success_rate": np.mean(successes),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_steps": np.mean(steps_list),
            "std_steps": np.std(steps_list),
        }
        
        return metrics, results
    
    def _save_video(self, frames: List[np.ndarray], path: Path):
        """Save frames as video."""
        try:
            import imageio
            # Use 60 fps for smoother playback
            imageio.mimsave(str(path), frames, fps=60)
            print(f"  → Saved video: {path}")
        except Exception as e:
            print(f"Failed to save video: {e}")
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Episodes: {metrics['n_episodes']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print(f"Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Avg Steps: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
        print("=" * 50)
    
    def save_metrics(self, metrics: Dict, path: str):
        """Save metrics to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()}
        
        with open(path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"Saved metrics to {path}")


def visualize_results(results: List[Dict], save_path: str = None):
    """
    Visualize evaluation results.
    
    Args:
        results: List of per-episode results
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        
        episodes = [r["episode"] for r in results]
        successes = [r["success"] for r in results]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        # Success/Failure per episode (binary)
        colors = ['green' if s else 'red' for s in successes]
        axes[0].bar(episodes, [1] * len(episodes), color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel("Episode")
        axes[0].set_yticks([])
        axes[0].set_title("Episode Outcomes (green=success, red=failure)")
        
        # Cumulative success rate
        cumulative_success = np.cumsum(successes) / (np.arange(len(successes)) + 1)
        axes[1].plot(episodes, cumulative_success, 'b-', linewidth=2, marker='o', markersize=4)
        axes[1].axhline(y=np.mean(successes), color='r', linestyle='--', label=f'Final: {np.mean(successes):.1%}')
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Success Rate")
        axes[1].set_title("Cumulative Success Rate")
        axes[1].legend()
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping visualization")
