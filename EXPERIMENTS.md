# Diffusion Policy Experiments

Goal: Break through 0.1 loss barrier (modeling after NVIDIA GROOT)

## Baseline Configuration
- **Task**: Can (robosuite)
- **Architecture**: DiT (Diffusion Transformer)
- **Hidden dim**: 256
- **Layers**: 4
- **Heads**: 8
- **obs_horizon**: 1 (single frame)
- **action_horizon**: 16
- **diffusion_steps**: 100
- **Scheduler**: DDPM with cosine schedule (squaredcos_cap_v2)
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Gradient clipping**: 1.0

## Experiments

### Experiment 1: Increase Transformer Depth (4 → 16 layers)
**Rationale**: GROOT uses 32 layers. Our 4 layers likely insufficient capacity.

**Changes**:
- `n_layers`: 4 → 16

**Result**: 
- Loss improved but still stuck around ~0.12-0.13
- Model now has 23.4M params (up from ~6M)

---

### Experiment 2: Multi-frame Observation (obs_horizon=4)
**Rationale**: GROOT uses multi-frame observations with temporal structure. Single frame lacks context.

**Changes**:
- `obs_horizon`: 1 → 4
- Dataset returns `(obs_horizon, obs_dim)` instead of flattening
- Added per-timestep observation encoder with learned positional embeddings

**Result**:
- Combined with Experiment 1
- Loss still around ~0.12-0.13 range
- Model can now see temporal context but not breaking 0.1

---

### Experiment 3: Cross-attention on ALL layers (disable interleaving)
**Rationale**: With interleaving, only 8/16 layers see observations. Full cross-attention = stronger conditioning.

**Changes**:
- `interleave_cross_attn`: True → False
- All 16 layers now have cross-attention to observations

**Result** (DDPM):
- Train loss: ~0.16 at epoch 4, val_loss: ~0.157
- Loss dropping to ~0.10-0.11 range by epoch 5
- Improvement but still plateauing around 0.1

---

### Experiment 4: Flow Matching (instead of DDPM)
**Rationale**: GROOT uses Flow Matching which predicts velocity v = x1 - x0 along straight path. Smoother gradients than noise prediction.

**Changes**:
- Linear interpolation: x_t = (1-t) * x0 + t * noise
- Predict velocity instead of noise
- Euler integration at inference

**Result**:
- Loss on different scale (~0.3-0.5 for velocity vs ~0.1-0.2 for noise)
- At epoch 16, still at ~0.27-0.38 loss
- **Did NOT help** - convergence slower than DDPM
- **REVERTED** - went back to DDPM

---

## Current Best Configuration (100% Success on Can Task)
- 16 layers, 512 hidden dim, 8 heads
- obs_horizon=4 with per-timestep encoding
- Cross-attention on ALL layers (no interleaving)
- DDPM scheduler with cosine schedule
- **Use `can_converted.hdf5`** (robomimic obs format, 23 dims)
- **Match controller config** from HDF5 env_args (OSC_POSE)
- Min-max action normalization to [-1, 1]
- Training: 76 epochs
- **Result: 100% success rate, 32.88 avg reward**

## Completed Experiments

### Experiment 5: Increase Hidden Dimension (256 → 512)
**Rationale**: GROOT uses 1024 hidden dim. 256 may be underpowered.

**Result**: Still plateauing around 0.10-0.17, ~90M params now. ❌ FAILED

---

### Experiment 6: Min-Max Action Normalization to [-1, 1]
**Rationale**: Diffusion models expect bounded [-1, 1] targets because the noise schedule is designed for that range. Z-score normalization produces unbounded values (could be +3 std, -2 std, etc.) which breaks the noise schedule assumptions.

**Changes**:
- Changed action normalization from z-score (`(x - mean) / std`) to min-max (`2 * (x - min) / (max - min) - 1`)
- Actions now guaranteed to be in [-1, 1] range

**Result**:
- ✅ **MAJOR BREAKTHROUGH!** Loss now dropping to 0.02-0.06 range (previously stuck at 0.10-0.17)
- Training loss at epoch 11: 0.022-0.070 (well below 0.1 barrier!)
- Loss may plateau at ~0.03-0.06 which is expected (irreducible error from multimodal actions)

**Status**: ✅ SUCCESS - Fixed the normalization issue

---

### Experiment 7: Fix Observation Format Mismatch (0% → 100% Success!)
**Date**: 2026-01-29

**Problem**: Despite low training loss (~0.04), evaluation showed 0% success rate on PickPlaceCan task.

**Root Cause Analysis**:
1. **Wrong observation data**: Training used raw MuJoCo states (71 dims) from `can.hdf5` instead of semantic observations (23 dims) from `can_converted.hdf5`
2. **Observation key mismatch**: HDF5 uses `object` key, but live robosuite env uses `object-state`
3. **Controller config mismatch**: Default controller differs from the OSC_POSE controller used during data collection

**Diagnosis Process**:
- Added debug logging to trace observation values during inference
- Found normalized observations had extreme values (-80 to +50) - way outside expected range
- Created `debug_obs_mismatch.py` to compare training data vs simulator observations
- Discovered dimension mismatch: training data had 71 dims, should have been 23 dims
- Even replaying exact demo actions failed (0.5 reward) due to environment randomization

**Fixes Applied**:

1. **`data/dataset.py`**: Load from `obs` group with semantic keys instead of raw `states`
   ```python
   obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
   # Concatenate to get 23-dim observation vector
   ```

2. **`envs/robosuite_wrapper.py`**: 
   - Use `object-state` key (live env name differs from HDF5)
   - Added exact controller config matching training data:
   ```python
   controller_config = {
       "type": "BASIC",
       "body_parts": {
           "right": {
               "type": "OSC_POSE",
               "kp": 150, "damping": 1,
               "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
               "gripper": {"type": "GRIP"}
               # ... other params matching HDF5 env_args
           }
       }
   }
   ```

3. **`train_modal.py`**: Load `{task}_converted.hdf5` files with proper robomimic format

**Results**:

| Metric | Before Fix | After Fix (5 epochs) | After Fix (76 epochs) |
|--------|------------|----------------------|----------------------|
| Success Rate | 0% | 20% | **100%** |
| Avg Reward | 0.16 | 12.16 | **32.88** |
| Avg Steps | 500 | 425 | **162** |
| Obs Dimension | 71 (wrong) | 23 (correct) | 23 (correct) |

**Status**: ✅ **MAJOR SUCCESS** - Task now solved with 100% success rate!

---

## Key Learnings (Updated)

1. **Diffusion models need bounded targets**: The noise schedule assumes data in ~[-1, 1] range. Z-score normalization breaks this.
2. **Loss floor is expected**: ~0.02-0.06 is good for robot manipulation - there's irreducible error from multimodal action distributions.
3. **Evaluate on task success, not just loss**: Low loss doesn't mean task success - observation/action format must match exactly between training and evaluation.
4. **Observation format is critical**: Training on raw simulator states vs semantic observations makes a huge difference. Always use the `obs` group from robomimic converted datasets.
5. **Controller config matters**: The action space interpretation depends on the controller. OSC_POSE with specific gains produces different behavior than default controllers.
6. **Debug systematically**: Compare training data vs live environment observations dimension-by-dimension to find mismatches.

---

### Other Ideas (Not Tried)
1. **Relax gradient clipping** (1.0 → 5.0) - may be limiting updates on hard samples
2. **State-relative actions** - predict deltas from current pose, not absolute
3. **Data analysis** - check for outliers, normalization issues
4. **Increase diffusion steps** (100 → 200-500)
5. **Lower learning rate** with longer training

---

## GROOT Architecture Reference (for comparison)
- 32 transformer layers (vs our 16)
- 1024 hidden dim (vs our 512)
- 2B parameter VLM backbone for observation encoding
- Flow Matching (didn't work for us)
- State-relative action prediction
- Multi-frame observations with temporal structure
