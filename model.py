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
class TrajectoryEncoder(nn.Module):
    # def __init__(self, action_dim, pred_horizon, latent_dim):
    #     super().__init__()
    #     input_dim = action_dim * pred_horizon
    #     self.net = nn.Sequential(
    #         nn.Linear(input_dim, 256),
    #         nn.Mish(),
    #         nn.Linear(256, 128),
    #         nn.Mish(),
    #         nn.Linear(128, latent_dim)
    #     )
    # def forward(self, actions):
    #     # actions: (B, T, D)
    #     B = actions.shape[0]
    #     flat_actions = actions.reshape(B, -1)
    #     return self.net(flat_actions)
    
    def __init__(self, action_dim, pred_horizon, latent_dim):
        super().__init__()
        # Simple MLP encoder architecture
        input_dim = action_dim * pred_horizon
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        # Two heads: one for Mean (mu), one for Log Variance (logvar)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, actions):
        # actions: (B, T, D) -> (B, T*D)
        B = actions.shape[0]
        x = actions.reshape(B, -1)
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The "Reparameterization Trick":
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
 
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

# class HyperNetwork(nn.Module):
#     """
#     Decodes the Latent 'z' into the weights of the Policy Network.
#     """
#     def __init__(self, latent_dim, param_shapes):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.param_shapes = param_shapes
        
#         # Calculate total number of parameters to generate
#         self.total_params = 0
#         for shape in param_shapes.values():
#             prod = 1
#             for dim in shape: prod *= dim
#             self.total_params += prod
            
#         # A simple MLP decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.Mish(),
#             nn.Linear(512, 1024),
#             nn.Mish(),
#             nn.Linear(1024, self.total_params)
#         )

#     def forward(self, z):
#         # z: (B, latent_dim)
#         flat_params = self.decoder(z)
        
#         # Reshape flat params into dictionary of weights
#         params_dict = {}
#         curr_idx = 0
#         for name, shape in self.param_shapes.items():
#             num_p = 1
#             for dim in shape: num_p *= dim
            
#             p = flat_params[:, curr_idx : curr_idx + num_p]
#             params_dict[name] = p.view(-1, *shape)
#             curr_idx += num_p
            
#         return params_dict
    
# --- C. HMLP Wrapper for HypnetTorch ---
class HyperNetwork(nn.Module):
    """
    Wraps hypnettorch.hnets.HMLP.
    Conditions on 'z' (latent_dim=256) to generate policy weights.
    Structure: [100, 100] hidden layers.
    """
    def __init__(self, latent_dim, param_shapes):
        super().__init__()
        self.param_shapes = param_shapes
        # Convert dictionary shapes to list of shapes for HMLP
        self.shapes_list = list(param_shapes.values())
        self.keys_list = list(param_shapes.keys())
        
        # Initialize HMLP
        self.hnet = HMLP(
            target_shapes=self.shapes_list,
            cond_in_size=latent_dim,
            layers=[100, 100],
            activation_fn=torch.nn.ReLU(),
            uncond_in_size=0,
            use_batch_norm=False
        )

    def forward(self, z):
        # z: (B, latent_dim)
        
        # HMLP forward returns: [ [p1_s1, p2_s1...], [p1_s2, p2_s2...], ... ]
        # Inner lists are parameters for a single sample.
        # Outer list is the batch.
        params_per_sample = self.hnet.forward(cond_input=z)
        
        # We need to Transpose this to: [ [p1_s1, p1_s2...], [p2_s1, p2_s2...], ... ]
        # zip(*list) is the standard python idiom for transposing lists of lists.
        params_per_layer = list(zip(*params_per_sample))
        
        params_dict = {}
        for i, key in enumerate(self.keys_list):
            # params_per_layer[i] is now a tuple of tensors for this specific weight across the batch
            # Stack them to get (Batch, ...)
            params_dict[key] = torch.stack(params_per_layer[i], dim=0)
            
        return params_dict

class FunctionalPolicy:
    @staticmethod
    def apply(obs, params_dict):
        # Layer 1
        w1 = params_dict['fc1.weight'] 
        b1 = params_dict['fc1.bias']   
        x = torch.einsum('bi,boi->bo', obs, w1) + b1
        x = F.mish(x) # Or F.relu
        
        # Layer 2
        w2 = params_dict['fc2.weight'] 
        b2 = params_dict['fc2.bias']   
        x = torch.einsum('bi,boi->bo', x, w2) + b2
        x = F.mish(x)
        
        # Layer 3 (Output)
        w3 = params_dict['fc3.weight']
        b3 = params_dict['fc3.bias']
        x = torch.einsum('bi,boi->bo', x, w3) + b3
        
        return x