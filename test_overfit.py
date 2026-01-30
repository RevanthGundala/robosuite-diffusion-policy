#!/usr/bin/env python
"""Test overfitting on 10 examples from Can dataset."""

import torch
import numpy as np
import h5py
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from policy import DiffusionPolicy, SuperSimpleModel

# Load only 10 demos from can.hdf5
print("Loading 10 demos from can.hdf5...")
demos = []
with h5py.File('data/can.hdf5', 'r') as f:
    for i in range(10):  # Only 10 demos
        demo_key = f'demo_{i}'
        if f'data/{demo_key}' in f:
            states = f[f'data/{demo_key}/states'][:]
            actions = f[f'data/{demo_key}/actions'][:]
            demos.append({'obs': states, 'actions': actions})

print(f"Loaded {len(demos)} demos")
print(f"Obs shape: {demos[0]['obs'].shape}, Action shape: {demos[0]['actions'].shape}")

# Compute normalization stats from these 10 demos
all_obs = np.concatenate([d['obs'] for d in demos])
all_actions = np.concatenate([d['actions'] for d in demos])

obs_mean = all_obs.mean(axis=0)
obs_std = all_obs.std(axis=0) + 1e-6
action_mean = all_actions.mean(axis=0)
action_std = all_actions.std(axis=0) + 1e-6

print(f"Total samples: {len(all_obs)}")

# Create simple dataset - just (obs, action_chunk) pairs
pred_horizon = 16
samples = []
for demo in demos:
    T = len(demo['obs'])
    for t in range(T - pred_horizon):
        obs = demo['obs'][t]
        actions = demo['actions'][t:t+pred_horizon]
        
        # Normalize
        obs_norm = (obs - obs_mean) / obs_std
        actions_norm = (actions - action_mean) / action_std
        
        samples.append((
            torch.FloatTensor(obs_norm),
            torch.FloatTensor(actions_norm)
        ))

print(f"Created {len(samples)} training samples")

# Use only 10 samples for overfitting test
samples = samples[:10]
print(f"Using only {len(samples)} samples for overfitting test")

from torch.utils.data import DataLoader, TensorDataset

obs_tensor = torch.stack([s[0] for s in samples])
actions_tensor = torch.stack([s[1] for s in samples])

# Step 1: Verify data is correct
print("\n=== Data Verification ===")
print("Obs shape:", obs_tensor.shape)
print("Actions shape:", actions_tensor.shape)
print("Obs sample (first 5 dims):", obs_tensor[0, :5])
print("Actions sample (first timestep):", actions_tensor[0, 0, :])
print("Obs has NaN:", torch.isnan(obs_tensor).any().item())
print("Actions has NaN:", torch.isnan(actions_tensor).any().item())
print("Obs range:", obs_tensor.min().item(), "to", obs_tensor.max().item())
print("Actions range:", actions_tensor.min().item(), "to", actions_tensor.max().item())
print("=========================\n")

dataset = TensorDataset(obs_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # batch = all samples

# Create policy
obs_dim = demos[0]['obs'].shape[1]  # 71 for Can
action_dim = demos[0]['actions'].shape[1]  # 7

# Use DiffusionPolicy with DiT model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Warmup settings
warmup_steps = 500  # Warmup for 500 steps

policy = DiffusionPolicy(
    hidden_dim=256,
    action_dim=action_dim,
    obs_dim=obs_dim,
    action_horizon=pred_horizon,
    n_diffusion_steps=100,
    lr=1e-3,
    device=device,
    warmup_steps=warmup_steps,
)
model = policy.model
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

print(f"\nTraining DiT DiffusionPolicy on {len(samples)} samples until loss < 0.01...")
print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
print(f"Using LR warmup for {warmup_steps} steps")

# Train until loss is near 0
import torch.nn.functional as F
from tqdm import tqdm

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Learning rate scheduler with warmup
def lr_lambda(step):
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
global_step = 0

max_epochs = 10000
target_loss = 0.01  # Same as MLP target

for epoch in range(max_epochs):
    epoch_loss = 0.0
    for obs, actions in dataloader:
        obs = obs.to(device)
        actions = actions.to(device)
        batch_size = actions.shape[0]
        
        # Use RANDOM timesteps (proper diffusion training)
        t = torch.randint(0, policy.n_diffusion_steps, (batch_size,), device=device)
        noisy_actions, noise = policy.forward_process(actions, t)
        noise_pred = model(noisy_actions, t, obs)
        loss = F.mse_loss(noise_pred, noise)
        
        # Debug: Check loss computation on first epoch
        if epoch == 0:
            print("\n=== Loss Computation Debug ===")
            print("Noise shape:", noise.shape)
            print("Noise_pred shape:", noise_pred.shape)
            print("Noise sample:", noise[0, 0, :].detach().cpu().numpy().round(3))
            print("Pred sample:", noise_pred[0, 0, :].detach().cpu().numpy().round(3))
            print("MSE Loss:", loss.item())
            print("Noise_pred std:", noise_pred.std().item(), "(should be ~1.0 if learning)")
            print("Noise_pred mean:", noise_pred.mean().item(), "(should be ~0.0 if learning)")
            print("==============================\n")
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Update learning rate with warmup
        global_step += 1
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    
    if epoch % 100 == 0 or avg_loss < target_loss:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
    
    if avg_loss < target_loss:
        print(f"\n✅ Reached target loss {target_loss} at epoch {epoch+1}!")
        break

if avg_loss >= target_loss:
    print(f"\n⚠️ Did not reach target loss. Final loss: {avg_loss:.6f}")

# Test: can we reproduce the training samples?
print("\n=== Testing on training data ===")
policy.model.eval()

total_mse = 0.0
for i in range(len(samples)):
    test_obs = obs_tensor[i:i+1]
    test_actions_true = actions_tensor[i].numpy()
    
    with torch.no_grad():
        pred_actions = policy.sample(test_obs)[0].cpu().numpy()
    
    mse = ((pred_actions - test_actions_true) ** 2).mean()
    total_mse += mse
    
    if i < 3:  # Show first 3 samples
        print(f"\nSample {i}:")
        print(f"  True actions[0]: {test_actions_true[0].round(2)}")
        print(f"  Pred actions[0]: {pred_actions[0].round(2)}")
        print(f"  MSE: {mse:.4f}")

avg_mse = total_mse / len(samples)
print(f"\n=== Overall Results ===")
print(f"Average MSE across {len(samples)} samples: {avg_mse:.4f}")

if avg_mse < 0.1:
    print("✅ Model CAN overfit - training is working correctly")
else:
    print("⚠️ Model CANNOT overfit - something is wrong with training")
