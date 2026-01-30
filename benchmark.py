"""
Benchmark comparison: Our Diffusion Policy vs robomimic's implementation.

Usage:
    uv run python benchmark.py --ours-checkpoint checkpoints/policy.pt --n-episodes 50
    uv run python benchmark.py --robomimic --n-episodes 50  # Run robomimic baseline
"""

import argparse
import json
from pathlib import Path


def eval_our_policy(checkpoint: str, n_episodes: int = 50):
    """Evaluate our diffusion policy implementation."""
    from policy import DiffusionPolicy
    from envs.robosuite_wrapper import RobosuiteWrapper
    from eval.evaluate import PolicyEvaluator
    from data.dataset import load_normalizer
    
    print("=" * 60)
    print("EVALUATING: Our Diffusion Policy")
    print("=" * 60)
    
    # Load policy
    policy = DiffusionPolicy(
        hidden_dim=256,
        action_dim=7,
        obs_dim=32,
        action_horizon=16,
        n_diffusion_steps=100,
    )
    policy.load(checkpoint)
    
    # Load normalizer
    normalizer_path = Path(checkpoint).parent / "normalizer.npz"
    normalizer = load_normalizer(str(normalizer_path))
    
    # Create environment
    env = RobosuiteWrapper(
        env_name="Lift",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    
    # Evaluate
    evaluator = PolicyEvaluator(
        env=env,
        policy=policy,
        normalizer=normalizer,
        action_horizon=8,
        pred_horizon=16,
    )
    
    metrics, results = evaluator.evaluate_n_episodes(n_episodes=n_episodes)
    env.close()
    
    return {
        "method": "Ours (DiT)",
        "success_rate": metrics["success_rate"],
        "avg_reward": metrics["avg_reward"],
        "avg_steps": metrics["avg_steps"],
        "n_episodes": n_episodes,
    }


def eval_robomimic_policy(checkpoint: str = None, n_episodes: int = 50):
    """Evaluate robomimic's diffusion policy implementation."""
    try:
        import robomimic
        import robomimic.utils.file_utils as FileUtils
        from robomimic.utils.train_utils import rollout_generator
    except ImportError:
        print("robomimic not installed. Install with: uv add robomimic")
        return None
    
    print("=" * 60)
    print("EVALUATING: robomimic Diffusion Policy")
    print("=" * 60)
    
    if checkpoint is None:
        print("No robomimic checkpoint provided. Train one first with:")
        print("  uv run python -m robomimic.scripts.train --config robomimic/exps/templates/diffusion_policy.json --dataset data/lift_low_dim.hdf5")
        return None
    
    # Load robomimic policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint)
    
    # Create environment from checkpoint config
    env_meta = ckpt_dict["env_metadata"]
    env = robomimic.utils.env_utils.create_env_from_metadata(env_meta)
    
    # Evaluate
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
        print(f"  Ep {ep+1}: {'✓' if success else '✗'} | Reward: {total_reward:.2f}")
    
    env.close()
    
    import numpy as np
    return {
        "method": "robomimic Diffusion Policy",
        "success_rate": np.mean(successes),
        "avg_reward": np.mean(rewards),
        "n_episodes": n_episodes,
    }


def print_comparison(results: list):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Method':<30} {'Success Rate':<15} {'Avg Reward':<15}")
    print("-" * 60)
    for r in results:
        if r:
            print(f"{r['method']:<30} {r['success_rate']*100:.1f}%{'':<10} {r['avg_reward']:.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark diffusion policy implementations")
    parser.add_argument("--ours-checkpoint", type=str, default="checkpoints/policy.pt",
                       help="Path to our policy checkpoint")
    parser.add_argument("--robomimic-checkpoint", type=str, default=None,
                       help="Path to robomimic policy checkpoint")
    parser.add_argument("--n-episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--robomimic", action="store_true",
                       help="Evaluate robomimic baseline")
    parser.add_argument("--ours-only", action="store_true",
                       help="Only evaluate our implementation")
    args = parser.parse_args()
    
    results = []
    
    # Evaluate our policy
    if not args.robomimic or args.ours_only:
        ours_result = eval_our_policy(args.ours_checkpoint, args.n_episodes)
        results.append(ours_result)
    
    # Evaluate robomimic (if requested and checkpoint provided)
    if args.robomimic and args.robomimic_checkpoint:
        robomimic_result = eval_robomimic_policy(args.robomimic_checkpoint, args.n_episodes)
        results.append(robomimic_result)
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output_path = Path("eval_results/benchmark.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
