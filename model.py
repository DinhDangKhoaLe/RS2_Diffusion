#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

import torch
import torch.nn as nn
import math
from typing import Union
import torch.nn.functional as F

from hypnettorch.hnets import HMLP

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)
    


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

# --- A. Trajectory Encoder (Compresses Action Trajectory -> Latent Z) ---
# class TrajectoryEncoder(nn.Module):
#     def __init__(self, state_dim, action_dim, pred_horizon, latent_dim):
#         super().__init__()
#         # Calculate input dimension: (State + Action) * Time Steps
#         input_dim = (state_dim + action_dim) * pred_horizon
#         # Simple MLP encoder architecture
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#         )
        
#         # Two heads: one for Mean (mu), one for Log Variance (logvar)
#         self.fc_mu = nn.Linear(512, latent_dim)
#         self.fc_logvar = nn.Linear(512, latent_dim)

#     def forward(self, states, actions):
#         """
#         Args:
#             states: Tensor of shape (B, T, state_dim)
#             actions: Tensor of shape (B, T, action_dim)
#         """
#         B = states.shape[0]
        
#         # 1. Concatenate state and action along the feature dimension (dim=-1)
#         # Result shape: (B, T, state_dim + action_dim)
#         trajectory = torch.cat([states, actions], dim=-1)
        
#         # 2. Flatten the time and feature dimensions
#         # Result shape: (B, T * (state_dim + action_dim))
#         x = trajectory.reshape(B, -1)
        
#         h = self.net(x)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar   

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + eps * std
#         else:
#             return mu
        
# -------------------------------------------------
# Basic TCN Block (Causal, Dilated, Residual)
# -------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]  # remove future padding (causal)
        out = self.relu(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = self.relu(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# -------------------------------------------------
# TCN Encoder (stacked dilated blocks)
# -------------------------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, kernel_size=3):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    in_ch,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.transpose(1, 2)  # -> (B, D, T)
        out = self.network(x)  # -> (B, hidden_dim, T)

        # take last time step (causal summary)
        return out[:, :, -1]   # -> (B, hidden_dim)
        

# -------------------------------------------------
# Trajectory Encoder with Separate State/Action TCNs
# -------------------------------------------------
class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        pred_horizon,
        latent_dim,
        tcn_hidden=256,
        tcn_layers=4,
    ):
        super().__init__()

        # Separate temporal encoders
        self.state_tcn = TCNEncoder(
            input_dim=state_dim,
            hidden_dim=tcn_hidden,
            num_layers=tcn_layers,
        )

        self.action_tcn = TCNEncoder(
            input_dim=action_dim,
            hidden_dim=tcn_hidden,
            num_layers=tcn_layers,
        )

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * tcn_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, states, actions):
        """
        states:  (B, T, state_dim)
        actions: (B, T, action_dim)
        """

        state_embed = self.state_tcn(states)     # (B, H)
        action_embed = self.action_tcn(actions)  # (B, H)

        fused = torch.cat([state_embed, action_embed], dim=-1)
        h = self.mlp(fused)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

        
class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_emb_dim=None):
        super().__init__()
        self.out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, 3, padding=1)
        
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, self.out_channels)
            
        self.norm2 = nn.GroupNorm(8, self.out_channels)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1)
        
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv1d(in_channels, self.out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        
        if time_emb is not None:
            # Project time_emb and broadcast over sequence length
            t_emb = self.time_emb_proj(F.silu(time_emb))
            h = h + t_emb[:, :, None]
            
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class TransformerBlock1D(nn.Module):
    """
    A Transformer block for 1D sequences with Cross-Attention.
    Correctly handles cases where cond_dim != model_channels.
    """
    def __init__(self, dim, n_heads, cond_dim, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 1. Self-Attention (Processing the Latent)
                # Inputs: x, x, x (All have dimension 'dim')
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True),
                
                # 2. Cross-Attention (Injecting the Condition)
                # Query: x (dimension 'dim')
                # Key/Value: context (dimension 'cond_dim')
                # We must specify kdim and vdim!
                nn.LayerNorm(dim),
                nn.MultiheadAttention(
                    embed_dim=dim, 
                    num_heads=n_heads, 
                    batch_first=True,
                    kdim=cond_dim,   # <--- FIX: Tell layer Key dim is cond_dim
                    vdim=cond_dim    # <--- FIX: Tell layer Value dim is cond_dim
                ),
                
                # 3. Feed Forward
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            ]))

    def forward(self, x, context):
        b, c, t = x.shape
        x_in = x.permute(0, 2, 1) # (B, T, C)
        
        # Ensure context has a sequence dimension for attention
        if context.ndim == 2:
            context = context.unsqueeze(1) # (B, 1, cond_dim)

        for ln_1, self_attn, ln_2, cross_attn, ln_3, ff in self.layers:
            # 1. Self Attention
            norm_x = ln_1(x_in)
            attn_out, _ = self_attn(norm_x, norm_x, norm_x)
            x_in = x_in + attn_out
            
            # 2. Cross Attention
            norm_x = ln_2(x_in)
            # Query=norm_x (dim=32), Key/Val=context (dim=10)
            # The layer now handles the projection from 10 -> 32 internally
            attn_out, _ = cross_attn(norm_x, context, context)
            x_in = x_in + attn_out
            
            # 3. Feed Forward
            out = ff(ln_3(x_in))
            x_in = x_in + out
            
        return x_in.permute(0, 2, 1)

class LatentDiffusionUNet(nn.Module):
    def __init__(self,
        input_dim,          # Dimension of the latent vector 'z' (e.g., 256)
        cond_dim=4,         # "Conditioning Dimension 4"
        model_channels=32,  # "Channels 32"
        channel_mults=(1, 2, 4), # "Channel Multipliers {1, 2, 4}"
        attn_levels=(0, 1), # "Attention Levels {0, 1}"
        n_res_blocks=1,     # "Number of Residual Blocks 1"
        n_heads=2,          # "Number of Heads 2"
        tf_layers=1,        # "Transformer Layers 1"
    ):
        super().__init__()
        
        # 1. Input Processing
        # We treat the input latent vector (B, input_dim) as a sequence (B, 1, input_dim)
        # or (B, input_dim, 1). To use 1D Conv properly, we map:
        # Input: (B, input_dim) -> Treat as (B, 1, input_dim) sequence
        # Channels = 1 (feature channel), Length = input_dim.
        
        self.input_dim = input_dim
        self.model_channels = model_channels
        
        # Time Embeddings
        time_embed_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial Conv
        self.input_conv = nn.Conv1d(1, model_channels, 3, padding=1)

        # Downsampling Stack
        self.down_blocks = nn.ModuleList([])
        ch = model_channels
        dims = [model_channels]
        
        curr_res = 1 # We track "resolution" logic as levels
        
        for i, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            for _ in range(n_res_blocks):
                layers = [ResnetBlock1D(ch, out_ch, time_embed_dim)]
                ch = out_ch
                
                # Check if we add Attention at this level
                if i in attn_levels:
                    layers.append(TransformerBlock1D(ch, n_heads, cond_dim, depth=tf_layers))
                
                self.down_blocks.append(nn.ModuleList(layers))
                dims.append(ch)
            
            # Downsample (except last level)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample1d(ch)]))
                dims.append(ch)

        # Middle Block
        self.mid_block1 = ResnetBlock1D(ch, ch, time_embed_dim)
        self.mid_attn = TransformerBlock1D(ch, n_heads, cond_dim, depth=tf_layers)
        self.mid_block2 = ResnetBlock1D(ch, ch, time_embed_dim)

        # Upsampling Stack
        self.up_blocks = nn.ModuleList([])
        
        # Reverse the multipliers for upsampling
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = model_channels * mult
            # Map the reversed index i back to the original level index for attention check
            original_level_idx = len(channel_mults) - 1 - i
            
            for _ in range(n_res_blocks + 1): # +1 for the skip connection
                layers = [ResnetBlock1D(ch + dims.pop(), out_ch, time_embed_dim)]
                ch = out_ch
                
                if original_level_idx in attn_levels:
                    layers.append(TransformerBlock1D(ch, n_heads, cond_dim, depth=tf_layers))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            if original_level_idx != 0:
                self.up_blocks.append(Upsample1d(ch))

        # Final Output
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, 1, 3, padding=1)

    def forward(self, x, timesteps, context):
        """
        x: (B, input_dim) -> Noisy Latent
        timesteps: (B,) -> Noise levels
        context: (B, cond_dim) -> Conditioning vector
        """
        # Shape Handling: (B, D) -> (B, 1, D)
        # 1 Channel, Length = D
        x = x.unsqueeze(1) 
        
        t = self.time_mlp(timesteps)
        
        h = self.input_conv(x)
        hs = [h]
        
        # --- Down ---
        for module in self.down_blocks:
            if isinstance(module, nn.ModuleList): # ResNet + Attn
                for layer in module:
                    if isinstance(layer, ResnetBlock1D):
                        h = layer(h, t)
                    elif isinstance(layer, TransformerBlock1D):
                        h = layer(h, context)
                hs.append(h)
            else: # Downsample
                h = module(h)
                hs.append(h)

        # --- Mid ---
        h = self.mid_block1(h, t)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t)

        # --- Up ---
        for module in self.up_blocks:
            if isinstance(module, nn.ModuleList): # ResNet + Attn
                # Concatenate Skip Connection
                h_skip = hs.pop()
                # Ensure shapes match (handle potential odd/even rounding issues if input_dim is weird)
                if h.shape[-1] != h_skip.shape[-1]:
                    h = F.interpolate(h, size=h_skip.shape[-1], mode='nearest')
                    
                h = torch.cat((h, h_skip), dim=1)
                
                for layer in module:
                    if isinstance(layer, ResnetBlock1D):
                        h = layer(h, t)
                    elif isinstance(layer, TransformerBlock1D):
                        h = layer(h, context)
            else: # Upsample
                h = module(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        
        # Shape Back: (B, 1, D) -> (B, D)
        return h.squeeze(1)
 
class LatentUnet1D(nn.Module):
    """
    Adapts the ConditionalUnet1D to diffuse a static latent vector 'z' 
    instead of a temporal action trajectory.
    
    We treat the latent vector (size: latent_dim) as a sequence of length 'latent_dim'
    with 1 channel.
    """
    def __init__(self,
        latent_dim, # This replaces input_dim. Size of the latent z.
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8
        ):
        super().__init__()
        
        # We treat the latent vector as a sequence with 1 channel.
        # So the 'input_dim' to the first Conv1d is 1.
        input_channels = 1 
        
        all_dims = [input_channels] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        # 1. Diffusion Step Encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # 2. Global Conditioning (State)
        cond_dim = dsed + global_cond_dim

        # 3. UNet Backbone (Same as before, but operating on reshaped z)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        # 4. Final projection back to 1 channel
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_channels, 1),
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        sample: (B, latent_dim) -> The Noisy Latent 'z'
        timestep: (B,)
        global_cond: (B, global_cond_dim) -> The Observation/State
        """
        
        # --- Reshape Flat Latent to Sequence ---
        # Input (B, latent_dim) -> (B, 1, latent_dim)
        # We treat the latent vector as a sequence of length 'latent_dim' with 1 channel
        x = sample.unsqueeze(1) 
        
        # --- Time & Condition Encoding ---
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # --- UNet Pass ---
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # --- Reshape back ---
        # (B, 1, latent_dim) -> (B, latent_dim)
        x = x.squeeze(1)
        return x

# --- C. HMLP Wrapper for HypnetTorch ---

class HyperNetwork(nn.Module):
    def __init__(self, latent_dim, param_shapes):
        super().__init__()
        self.param_shapes = param_shapes
        self.keys_list = list(param_shapes.keys())
        self.shapes_list = list(param_shapes.values())
        
        # Initialize HMLP
        self.hnet = HMLP(
            target_shapes=self.shapes_list,
            cond_in_size=latent_dim,
            layers=(400, 400),      
        )

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)
        Returns:
            params_dict: dict[str, Tensor] with shape (B, *param_shape)
        """
        params = self.hnet(cond_input=z)

        # Normalize to batch-first list
        if isinstance(params[0], torch.Tensor):
            params = [params]  # B = 1

        # params: List[B] of List[num_params]
        params = list(zip(*params))  # param-first

        return {
            key: torch.stack(p, dim=0)
            for key, p in zip(self.keys_list, params)
        }
        
    # def forward(self, z):
    #     """
    #     Args:
    #         z: (B, latent_dim)
    #     Returns:
    #         params_dict: dict of tensors with shape (B, *param_shape)
    #     """
    #     # HMLP returns a list of tensors: [tensor(B, shape1), tensor(B, shape2), ...]
    #     params = self.hnet(cond_input=z)
        
    #     # Ensure params is a list of tensors, not a list of lists
    #     # If HMLP returns a list of lists, this dictionary comprehension 
    #     # should be optimized using torch.cat or similar batch operations.
    #     return {
    #         key: p for key, p in zip(self.keys_list, params)
    #     }

class FunctionalPolicy:
    @staticmethod
    def apply(obs, params_dict):
        # Handle both (B, Dim) and (B, T, Dim)
        if obs.dim() == 3:
            # Trajectory Mode: (Batch, Time, Dim)
            einsum_str = 'bti,boi->bto'
            # We must unsqueeze bias to (B, 1, Out) to broadcast over Time
            bias_op = lambda b: b.unsqueeze(1)
        else:
            # Single Step Mode: (Batch, Dim)
            einsum_str = 'bi,boi->bo'
            # No change needed for bias
            bias_op = lambda b: b

        x = obs
        
        # Layer 1
        w1 = params_dict['fc1.weight'] 
        b1 = params_dict['fc1.bias']   
        # Apply bias_op(b1) to handle the broadcasting
        x = torch.einsum(einsum_str, x, w1) + bias_op(b1)
        x = F.relu(x)
        
        # Layer 2
        w2 = params_dict['fc2.weight'] 
        b2 = params_dict['fc2.bias']   
        x = torch.einsum(einsum_str, x, w2) + bias_op(b2)
        x = F.relu(x)
        
        # Layer 3
        w3 = params_dict['fc3.weight']
        b3 = params_dict['fc3.bias']
        x = torch.einsum(einsum_str, x, w3) + bias_op(b3)
        
        # return torch.tanh(x)
        return x