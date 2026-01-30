import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

from diffusers.models.attention import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, SinusoidalPositionalEmbedding
from diffusers.schedulers import DDPMScheduler

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class TimestepEncoder(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.t_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.t_emb = TimestepEmbedding(in_channels=256, time_embed_dim=out_dim)
    
    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        return self.t_emb(self.t_proj(timesteps).to(dtype))

class AdaLayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_dim, emb_dim * 2)
        self.norm = nn.LayerNorm(normalized_shape=emb_dim, elementwise_affine=False)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        t_emb = self.linear(self.silu(t_emb))
        scale, shift = t_emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            n_heads: int,
            dim_head: int,
            hidden_dim: Optional[int] = None,
            cross_attn_dim: int = None,  # None = self-attention only, not None = has cross-attention
            max_seq_len: int = 512,
    ):
        super().__init__()
        self.has_cross_attn = cross_attn_dim is not None
        
        # Positional embedding (applied inside block after norm, before attention)
        self.pos_embed = SinusoidalPositionalEmbedding(embed_dim=hidden_dim, max_seq_length=max_seq_len)
        
        # Self-attention
        self.norm1 = AdaLayerNorm(emb_dim=hidden_dim)
        self.attn1 = Attention(
            query_dim=hidden_dim,
            heads=n_heads,
            dim_head=dim_head,
        )  # Self-attention - no cross_attention_dim
        
        # Cross-attention (only if cross_attn_dim provided)
        if self.has_cross_attn:
            self.norm2 = AdaLayerNorm(emb_dim=hidden_dim)
            self.attn2 = Attention(
                query_dim=hidden_dim, 
                heads=n_heads,
                dim_head=dim_head,
                cross_attention_dim=cross_attn_dim,
            )
        
        # FFN
        self.norm3 = AdaLayerNorm(emb_dim=hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor = None):
        # Self-attention (apply pos_embed to normalized states, residual preserves original x)
        norm_x = self.norm1(x, timesteps)
        norm_x = self.pos_embed(norm_x)
        x = x + self.attn1(norm_x)
        
        # Cross-attention (only if this layer has it)
        if self.has_cross_attn and encoder_hidden_states is not None:
            x = x + self.attn2(self.norm2(x, timesteps), encoder_hidden_states=encoder_hidden_states)
        
        # FFN
        x = x + self.ff(self.norm3(x, timesteps))
        return x 
    
class DiT(nn.Module): 
    def __init__(
        self, 
        hidden_dim: int, 
        action_dim: int, 
        obs_dim: int,
        action_horizon: int = 16,
        obs_horizon: int = 1,
        n_layers: int = 16,
        n_heads: int = 8,
        interleave_cross_attn: bool = True,
    ):
        super().__init__()
        self.timestep_encoder = TimestepEncoder(emb_dim=hidden_dim, out_dim=hidden_dim)
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.interleave_cross_attn = interleave_cross_attn
        dim_head = hidden_dim // n_heads
        
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # Per-timestep observation encoder: encodes each obs frame independently
        # Input: (B, obs_horizon, obs_dim) -> Output: (B, obs_horizon, hidden_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learned positional embedding for observation timesteps
        self.obs_pos_embed = nn.Embedding(obs_horizon, hidden_dim)
        
        # Interleaved layers: EVEN layers have cross-attention (0,2,4...), ODD layers self-attn only
        # Or all layers have cross-attention if interleave_cross_attn=False
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim, 
                n_heads=n_heads, 
                dim_head=dim_head, 
                cross_attn_dim=hidden_dim if (not interleave_cross_attn or i % 2 == 0) else None,
            )
            for i in range(n_layers)
        ])
        
        # AdaLN-Zero output block (from DiT paper)
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(hidden_dim, 2 * hidden_dim)  # For shift and scale
        self.proj_out_2 = nn.Linear(hidden_dim, action_dim)  # Final projection
        
        # Print model size
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DiT initialized: {n_layers} layers, {n_heads} heads, {hidden_dim} dim, {n_params:,} params")

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor):
        """
        Args:
            noisy_action: (B, T, action_dim) - noisy action sequence
            timestep: (B,) - diffusion timestep
            obs: (B, obs_horizon, obs_dim) or (B, obs_dim) - observation(s)
        Returns:
            noise_pred: (B, T, action_dim) - predicted noise
        """
        # Embed noisy actions
        t_emb = self.timestep_encoder(timestep)
        x = self.action_embed(noisy_action)
        
        # Handle both single-frame (B, obs_dim) and multi-frame (B, obs_horizon, obs_dim) observations
        if obs.dim() == 2:
            # Single frame: (B, obs_dim) -> (B, 1, obs_dim)
            obs = obs.unsqueeze(1)
        
        # Per-timestep encoding: apply MLP to each observation frame independently
        # (B, obs_horizon, obs_dim) -> (B, obs_horizon, hidden_dim)
        encoder_hidden_states = self.obs_encoder(obs)
        
        # Add positional embeddings to distinguish observation timesteps
        B, T_obs, _ = encoder_hidden_states.shape
        obs_pos_ids = torch.arange(T_obs, device=obs.device)
        encoder_hidden_states = encoder_hidden_states + self.obs_pos_embed(obs_pos_ids)
        
        # Pass through transformer layers with timestep and obs conditioning
        for layer in self.layers:
            x = layer(x, t_emb, encoder_hidden_states)
        
        # AdaLN-Zero output processing (applies timestep conditioning at output)
        shift, scale = self.proj_out_1(F.silu(t_emb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)

class DiffusionPolicy:
    def __init__(
        self, 
        hidden_dim: int = 512,
        action_dim: int = 7,  # Panda: 6-DOF arm + 1 gripper
        obs_dim: int = 32,    # Robosuite proprio + object state (per frame)
        action_horizon: int = 16,
        obs_horizon: int = 1,
        n_layers: int = 16,
        n_diffusion_steps: int = 100,
        device: str = None,
        epochs: int = 100,
        lr: float = 1e-4,
        warmup_steps: int = 0,
        interleave_cross_attn: bool = True,
        use_flow_matching: bool = False,  # DDPM works better than Flow Matching
    ):
        self.device = device if device is not None else get_device()
        self.n_diffusion_steps = n_diffusion_steps
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.interleave_cross_attn = interleave_cross_attn
        self.use_flow_matching = use_flow_matching
        self.lr = lr
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        
        # Initialize model
        self.model = DiT(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            obs_dim=obs_dim,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            n_layers=n_layers,
            interleave_cross_attn=interleave_cross_attn,
        ).to(self.device)
        
        if use_flow_matching:
            # Flow Matching: no scheduler needed, we use simple linear interpolation
            # Model predicts velocity v = x1 - x0
            print("Using Flow Matching (velocity prediction)")
            self.noise_scheduler = None
        else:
            # Use DDPMScheduler from diffusers
            print("Using DDPM (noise prediction)")
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=n_diffusion_steps,
                beta_schedule="squaredcos_cap_v2",  # cosine schedule, better than linear
                clip_sample=True,
                prediction_type="epsilon",  # predict noise
            )
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Add noise to clean actions x0 at timestep t.
        
        For Flow Matching: x_t = (1-t) * x0 + t * noise (linear interpolation)
        For DDPM: Uses scheduler's add_noise method
        
        Args:
            x0: (B, T, action_dim) - clean action sequence
            t: (B,) - timesteps (integer for DDPM, will convert to float for FM)
        Returns:
            x_t: (B, T, action_dim) - noisy actions
            target: (B, T, action_dim) - the target (velocity for FM, noise for DDPM)
        """
        noise = torch.randn_like(x0)
        
        if self.use_flow_matching:
            # Flow Matching: linear interpolation
            # t is integer timestep [0, n_diffusion_steps), convert to float [0, 1]
            t_float = t.float() / self.n_diffusion_steps
            t_float = t_float[:, None, None]  # (B, 1, 1) for broadcasting
            
            # x_t = (1 - t) * x0 + t * noise
            x_t = (1 - t_float) * x0 + t_float * noise
            
            # Target is the velocity: v = noise - x0 (direction from data to noise)
            # At inference we integrate backwards: x_{t-dt} = x_t - dt * v
            velocity = noise - x0
            return x_t, velocity
        else:
            # DDPM: Use scheduler's add_noise method
            x_t = self.noise_scheduler.add_noise(x0, noise, t)
            return x_t, noise
    
    def train(self, dataloader, epochs: int = None, lr: float = None, val_dataloader=None, checkpoint_path=None):
        """
        Train the diffusion policy.
        
        Args:
            dataloader: yields (obs, actions) batches
            epochs: number of training epochs (overrides self.epochs if provided)
            lr: learning rate (overrides self.lr if provided)
            val_dataloader: optional validation dataloader
            checkpoint_path: path to save best checkpoint
            
        Returns:
            dict with training history (train_losses, val_losses, best_val_loss)
        """
        import tqdm
        
        # Use provided values or fall back to instance values
        epochs = epochs if epochs is not None else self.epochs
        lr = lr if lr is not None else self.lr
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # OneCycleLR: warmup + cosine annealing (built-in PyTorch scheduler)
        total_steps = epochs * len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.05,  # 5% warmup
            anneal_strategy='cos',  # Cosine annealing
            div_factor=25,  # Initial LR = max_lr / 25
            final_div_factor=100,  # Final LR = max_lr / 100
        )
        global_step = 0
        print(f"OneCycleLR scheduler: {total_steps} total steps, 5% warmup")
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
        }

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            tqdm_loader = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (obs, actions) in enumerate(tqdm_loader):
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                batch_size = actions.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=self.device)
                
                # Add noise to actions (returns velocity target for FM, noise for DDPM)
                noisy_actions, target = self.forward_process(actions, t)
                
                # Predict velocity (FM) or noise (DDPM)
                pred = self.model(noisy_actions, t, obs)
                
                # MSE loss between predicted and target
                loss = F.mse_loss(pred, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate
                global_step += 1
                
                epoch_loss += loss.item()
                tqdm_loader.set_postfix(loss=loss.item())
            
            avg_loss = epoch_loss / len(dataloader)
            history['train_losses'].append(avg_loss)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                history['val_losses'].append(val_loss)
                
                # Save best checkpoint
                if val_loss < history['best_val_loss']:
                    history['best_val_loss'] = val_loss
                    if checkpoint_path is not None:
                        self.save(checkpoint_path)
                        print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.6f})")
                
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        return history
    
    @torch.no_grad()
    def _validate(self, val_dataloader):
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        for obs, actions in val_dataloader:
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            batch_size = actions.shape[0]
            
            t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=self.device)
            noisy_actions, target = self.forward_process(actions, t)
            pred = self.model(noisy_actions, t, obs)
            loss = F.mse_loss(pred, target)
            total_loss += loss.item()
        
        return total_loss / len(val_dataloader)
    
    @torch.no_grad()
    def sample(self, obs: torch.Tensor, n_inference_steps: int = None) -> torch.Tensor:
        """
        Sample actions given observation.
        
        For Flow Matching: Euler integration from noise to data
        For DDPM: Standard reverse diffusion process
        
        Args:
            obs: (B, obs_dim) or (B, obs_horizon, obs_dim) - observation
            n_inference_steps: Number of inference steps (default: n_diffusion_steps)
        Returns:
            actions: (B, T, action_dim) - denoised action sequence
        """
        self.model.eval()
        batch_size = obs.shape[0] if obs.dim() >= 2 else 1
        obs = obs.to(self.device)
        
        if n_inference_steps is None:
            n_inference_steps = self.n_diffusion_steps
        
        # Start from pure noise
        x = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device)
        
        if self.use_flow_matching:
            # Flow Matching: Euler integration from noise (t=1) to data (t=0)
            # x_{t-dt} = x_t - dt * v(x_t, t)
            dt = 1.0 / n_inference_steps
            
            for i in range(n_inference_steps):
                # Current time: t = 1 - i/n_inference_steps (going from 1 to 0)
                t = 1.0 - i / n_inference_steps
                # Convert to integer timestep for model input
                t_int = int(t * self.n_diffusion_steps)
                t_batch = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)
                
                # Predict velocity
                velocity = self.model(x, t_batch, obs)
                
                # Euler step: x_{t-dt} = x_t - dt * v
                x = x - dt * velocity
            
            return x
        else:
            # DDPM: Standard reverse diffusion
            self.noise_scheduler.set_timesteps(n_inference_steps)
            
            for t in self.noise_scheduler.timesteps:
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                noise_pred = self.model(x, t_batch, obs)
                x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
            
            return x
    
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'action_dim': self.action_dim,
                'obs_dim': self.obs_dim,
                'action_horizon': self.action_horizon,
                'obs_horizon': self.obs_horizon,
                'n_layers': self.n_layers,
                'n_diffusion_steps': self.n_diffusion_steps,
                'interleave_cross_attn': self.interleave_cross_attn,
                'use_flow_matching': self.use_flow_matching,
            },
            'scheduler_config': self.noise_scheduler.config if self.noise_scheduler else None,
        }
        
        torch.save(checkpoint, path)

        print(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model to
            
        Returns:
            DiffusionPolicy instance
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        if device is None:
            device = get_device()
        
        # Create policy with saved config (with backwards compatibility)
        policy = cls(
            hidden_dim=config['hidden_dim'],
            action_dim=config['action_dim'],
            obs_dim=config['obs_dim'],
            action_horizon=config['action_horizon'],
            obs_horizon=config.get('obs_horizon', 1),
            n_layers=config.get('n_layers', 16),
            n_diffusion_steps=config['n_diffusion_steps'],
            interleave_cross_attn=config.get('interleave_cross_attn', True),
            use_flow_matching=config.get('use_flow_matching', False),  # Default False for old checkpoints
            device=device,
        )
        
        # Load model weights
        policy.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from {path}")
        return policy


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class SuperSimpleModel(nn.Module):
    """Simple MLP for debugging - to verify training pipeline works."""
    def __init__(self, obs_dim: int, action_dim: int, action_horizon: int, n_timesteps: int = 100):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
        # Timestep embedding (sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            nn.Embedding(n_timesteps, 128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        # Main network - now includes timestep embedding
        input_dim = obs_dim + action_dim * action_horizon + 256  # +256 for timestep
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim * action_horizon),
        )
    
    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor):
        B = obs.shape[0]
        # Embed timestep
        t_emb = self.time_embed(timestep)  # (B, 256)
        # Flatten noisy actions
        noisy_flat = noisy_action.reshape(B, -1)
        # Concatenate obs, noisy_actions, and timestep embedding
        x = torch.cat([obs, noisy_flat, t_emb], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.action_horizon, self.action_dim)