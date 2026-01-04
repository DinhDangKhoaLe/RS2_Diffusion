import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import click

from model import ConditionalUnet1D, FunctionalPolicy, TrajectoryEncoder, LatentUnet1D, HyperNetwork
from dataset_pusht_state import PushTStateDataset
from vision_encoder import get_resnet, replace_bn_with_gn

@click.command()
@click.option('-i', '--dataset_path', required=True)
@click.option('-o', '--checkpoint_dir', default='ckpt', help='Directory to save checkpoints')
def main(dataset_path,checkpoint_dir):
    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    
    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape", batch['action'].shape)
    
    # --- Define Policy Architecture (Target for HyperNet) ---
    # We want a policy that takes (Obs) and outputs (Action Trajectory)
    # Input: obs_dim
    # Output: action_dim
    
    obs_dim = 5
    action_dim = 2
    latent_dim = 256  # Size of 'z'
    hidden_dim = 256 # 256 neurons per layer

    policy_input_dim = obs_horizon*obs_dim
    policy_output_dim = pred_horizon*action_dim
    
    # 2 Hidden Layers (Input -> FC1 -> FC2 -> FC3 -> Output)
    policy_shapes = {
        'fc1.weight': (hidden_dim, policy_input_dim),
        'fc1.bias':   (hidden_dim,),
        
        'fc2.weight': (hidden_dim, hidden_dim),
        'fc2.bias':   (hidden_dim,),
        
        'fc3.weight': (policy_output_dim, hidden_dim),
        'fc3.bias':   (policy_output_dim,)
    }
    
    # --- Initialize Networks ---
    device = torch.device('cuda')
    
    # 1. Encoder (Trainable): Action -> z
    encoder = TrajectoryEncoder(action_dim, pred_horizon, latent_dim).to(device)
    
    # 2. Diffusion Model (Trainable): Noisy z -> Noise
    noise_pred_net = LatentUnet1D(
        latent_dim=latent_dim,
        global_cond_dim=obs_dim*obs_horizon, # Condition on Obs
        down_dims=[256, 512, 1024] # Smaller Unet for latent vector
    ).to(device)
    
    # 3. HyperNetwork (Trainable): z -> Weights
    hypernet = HyperNetwork(latent_dim, policy_shapes).to(device)
    
    # --- Scheduler ---
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    
    # Training Config
    phase1_epochs = 300  # Train Autoencoder
    phase2_epochs = 1000  # Train Diffusion
    
    # A small constant to prevent posterior collapse (where the encoder ignores the input).
    kl_weight = 1e-8
    
    # ==========================================================================
    # PHASE 1: REPRESENTATION LEARNING (Autoencoder)
    # Train Encoder + Hypernet to reconstruct actions. Ignore Diffusion.s
    # ==========================================================================
    
    print(f"\n=== Starting Phase 1: Representation Learning ({phase1_epochs} Epochs) ===")
    
    optimizer_p1 = torch.optim.AdamW(
        list(encoder.parameters()) + list(hypernet.parameters()), 
        lr=3e-4, weight_decay=1e-6
    )
    
    lr_scheduler_p1 = get_scheduler(
        'cosine', optimizer=optimizer_p1, 
        num_warmup_steps=500, num_training_steps=len(dataloader)*phase1_epochs
    )
    
    with tqdm(range(phase1_epochs), desc='Phase 1 Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            for nbatch in tqdm(dataloader, desc='Batch', leave=False):
                # Data
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]
                obs_cond = nobs[:,:obs_horizon,:].flatten(start_dim=1)

                # Forward
                # 1. Variational Encoding
                mu, logvar = encoder(naction)
                
                # 2. Reparameterize (Sample z)
                z = encoder.reparameterize(mu, logvar)
                
                # 3. Decode (Hypernet -> Policy -> Action)
                weights = hypernet(z)
                pred_action_flat = FunctionalPolicy.apply(obs_cond, weights)
                pred_action = pred_action_flat.reshape(B, pred_horizon, action_dim)

                # --- ELBO LOSS CALCULATION ---
                
                # A. Reconstruction Loss (Likelihood)
                recon_loss = F.mse_loss(pred_action, naction)

                # B. KL Divergence Loss
                # D_KL( q(z|x) || N(0,1) )
                # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # Normalize by batch size implies average KL per batch
                kl_loss = kl_loss / B 

                # Total Loss (Negative ELBO)
                loss = recon_loss + (kl_weight * kl_loss)

                # Optimize
                optimizer_p1.zero_grad()
                loss.backward()
                optimizer_p1.step()
                lr_scheduler_p1.step()

                epoch_loss.append(loss.item())
            
            avg_loss = np.mean(epoch_loss)
            tglobal.set_postfix(recon_loss=avg_loss)
            
            # Save Phase 1 Checkpoint
            if (epoch_idx + 1) % 50 == 0:
                torch.save({
                    'encoder': encoder.state_dict(),
                    'hypernet': hypernet.state_dict(),
                }, f"{checkpoint_dir}/phase1_epoch_{epoch_idx+1}.pth")

    print("Phase 1 Complete. Latent Space Learned.")
    
    # ==========================================================================
    # TRANSITION: FREEZE REPRESENTATION
    # ==========================================================================
    
    # Freeze Encoder and HyperNet
    for param in encoder.parameters(): param.requires_grad = False
    for param in hypernet.parameters(): param.requires_grad = False
    
    # Put them in eval mode (important for things like Dropout/BatchNorm)
    encoder.eval()
    hypernet.eval()

    # ==========================================================================
    # PHASE 2: LATENT DIFFUSION
    # Train UNet to generate 'z'. Encoder/HyperNet are frozen.
    # ==========================================================================
    
    print(f"\n=== Starting Phase 2: Latent Diffusion ({phase2_epochs} Epochs) ===")

    optimizer_p2 = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
    
    ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
    
    lr_scheduler_p2 = get_scheduler(
        'cosine', optimizer=optimizer_p2, 
        num_warmup_steps=500, num_training_steps=len(dataloader)*phase2_epochs
    )

    with tqdm(range(phase2_epochs), desc='Phase 2 Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            for nbatch in tqdm(dataloader, desc='Batch', leave=False):
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]
                obs_cond = nobs[:,:obs_horizon,:].flatten(start_dim=1)

                # 1. Get Ground Truth 'z' (No Gradients here!)
                with torch.no_grad():
                    mu, logvar = encoder(naction)
                    z_gt = encoder.reparameterize(mu, logvar)
                    
                    # *Optional*: Scale z_gt to ensure it's roughly unit variance 
                    # if the KL weight was very low. (Standard LDM practice)
                    # z_gt = z_gt * 0.18215 # Typical scaling factor in Stable Diffusion, but check your variance stats first!
                    # For now, we will leave scaling out unless training is unstable.

                # 2. Diffusion Process
                noise = torch.randn_like(z_gt)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                ).long()
                
                noisy_z = noise_scheduler.add_noise(z_gt, noise, timesteps)
                
                # 3. Predict Noise
                noise_pred = noise_pred_net(noisy_z, timesteps, global_cond=obs_cond)
                
                # 4. Loss & Optimize
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer_p2.zero_grad()
                loss.backward()
                optimizer_p2.step()
                lr_scheduler_p2.step()
                
                ema.step(noise_pred_net.parameters())
                epoch_loss.append(loss.item())

            tglobal.set_postfix(diff_loss=np.mean(epoch_loss))

            # Save Phase 2 Checkpoint (Full Model)
            if (epoch_idx + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch_idx + 1,
                    'diffusion_model': noise_pred_net.state_dict(),
                    'encoder': encoder.state_dict(),
                    'hypernet': hypernet.state_dict(),
                    'stats': dataset.stats
                }, f"{checkpoint_dir}/phase2_epoch_{epoch_idx+1}.pth")

    print("Training Complete.")


if __name__ == '__main__':
    main()