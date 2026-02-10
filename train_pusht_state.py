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

from model import ConditionalUnet1D, FunctionalPolicy, TrajectoryEncoder, LatentUnet1D, HyperNetwork, LatentDiffusionUNet
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
    action_horizon = 16
    
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
        batch_size=128,
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
    print("batch['state'].shape:", batch['state'].shape)
    print("batch['action'].shape", batch['action'].shape)
    
    # --- Define Policy Architecture (Target for HyperNet) ---
    # We want a policy that takes (Obs) and outputs (Action Trajectory)
    # Input: obs_dim
    # Output: action_dim
    
    obs_dim = 5
    state_dim = 5
    action_dim = 2
    latent_dim = 256  # Size of 'z'
    hidden_dim = 256 # 256 neurons per layer
    
    # 2 Hidden Layers (Input -> FC1 -> FC2 -> FC3 -> Output)
    policy_shapes = {
        'fc1.weight': (hidden_dim, state_dim),
        'fc1.bias':   (hidden_dim,),
        
        'fc2.weight': (hidden_dim, hidden_dim),
        'fc2.bias':   (hidden_dim,),
        
        'fc3.weight': (action_dim, hidden_dim),
        'fc3.bias':   (action_dim,)
    }
    
    # --- Initialize Networks ---
    device = torch.device('cuda')
    
    # 1. Encoder (Trainable): Action -> z
    encoder = TrajectoryEncoder(state_dim, action_dim, pred_horizon, latent_dim).to(device)
    
    # 2. Diffusion Model (Trainable): Noisy z -> Noise
    # noise_pred_net = LatentUnet1D(
    #     latent_dim=latent_dim,
    #     global_cond_dim=obs_dim*obs_horizon, # Condition on Obs
    # ).to(device)

    noise_pred_net = LatentUnet1D(
        latent_dim=latent_dim,
        global_cond_dim=obs_dim*obs_horizon, # Condition on Obs
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
    phase1_epochs = 1  # Train Autoencoder
    phase2_epochs = 500  # Train Diffusion
    
    # A small constant to prevent posterior collapse (where the encoder ignores the input).
    kl_weight = 1e-10
    
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
                nstate = nbatch['state'].to(device)
                naction = nbatch['action'].to(device)
                B, _, _ = nstate.shape

                # Forward
                # 1. Variational Encoding
                mu, logvar = encoder(nstate, naction)
                
                # 2. Reparameterize (Sample z)
                z = encoder.reparameterize(mu, logvar)
                
                # 3. Decode (Hypernet -> Policy -> Action)
                weights = hypernet(z)
                pred_action = FunctionalPolicy.apply(nstate, weights)

                # --- ELBO LOSS CALCULATION ---
                # A. Reconstruction Loss (Likelihood)
                # recon_loss = F.mse_loss(pred_action, naction)
                recon_loss = F.mse_loss(pred_action, naction, reduction='sum') / B

                # B. KL Divergence Loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / B

                # Total Loss (Negative ELBO)
                loss = recon_loss + (kl_weight * kl_loss)

                # --- Optimization ---
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
    
    print("\nFreezing VAE and HyperNetwork for Phase 2...")
    # Freeze Encoder and HyperNet
    for param in encoder.parameters(): param.requires_grad = False
    for param in hypernet.parameters(): param.requires_grad = False
    
    # Put them in eval mode
    encoder.eval()
    hypernet.eval()

    # ==========================================================================
    # PHASE 2: LATENT DIFFUSION
    # Train UNet to generate 'z' (Skill) from 'obs' (Context).
    # ==========================================================================
    
    print(f"=== Starting Phase 2: Latent Diffusion ({phase2_epochs} Epochs) ===")

    # Initialize Optimizer for Diffusion Model (Noise Predictor)
    optimizer_p2 = torch.optim.AdamW(noise_pred_net.parameters(), lr=3e-4, weight_decay=1e-6)
    
    # Initialize EMA (Exponential Moving Average) for stable sampling
    # ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
    ema = EMAModel(noise_pred_net, power=0.75)

    
    # Cosine Scheduler
    lr_scheduler_p2 = get_scheduler(
        'cosine', optimizer=optimizer_p2, 
        num_warmup_steps=500, num_training_steps=len(dataloader)*phase2_epochs
    )

    with tqdm(range(phase2_epochs), desc='Phase 2 Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            
            for nbatch in tqdm(dataloader, desc='Batch', leave=False):
                # Data
                nobs = nbatch['obs'].to(device)
                nstate = nbatch['state'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]
                
                # Conditioning Context: The starting observation(s)
                # Flatten (B, Obs_Horizon, Dim) -> (B, Obs_Horizon * Dim)
                obs_cond = nobs[:,:obs_horizon,:].flatten(start_dim=1)

                # --- 1. Get Ground Truth 'z' (Latent Skill) ---
                # We use the frozen Encoder to extract the "Perfect Skill" for this trajectory.
                with torch.no_grad():
                    # CRITICAL UPDATE: Pass both State and Action to encoder
                    mu, logvar = encoder(nstate, naction)
                    z_gt = encoder.reparameterize(mu, logvar)
                    
                # --- 2. Diffusion Forward Process ---
                # Sample noise to add to the latent
                noise = torch.randn_like(z_gt)
                
                # Sample random timesteps for each batch item
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                ).long()
                
                # Add noise to the clean latent according to the schedule
                noisy_z = noise_scheduler.add_noise(z_gt, noise, timesteps)
                
                # --- 3. Predict Noise ---
                # The UNet tries to predict the noise added, conditioned on the current observation
                noise_pred = noise_pred_net(noisy_z, timesteps, obs_cond)
                
                # --- 4. Loss & Optimize ---
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer_p2.zero_grad()
                loss.backward()
                optimizer_p2.step()
                lr_scheduler_p2.step()
                
                # Update EMA model
                ema.step(noise_pred_net)
                
                epoch_loss.append(loss.item())

            # Logging
            avg_loss = np.mean(epoch_loss)
            tglobal.set_postfix(diff_loss=avg_loss)

            # Save Phase 2 Checkpoint
            if (epoch_idx + 1) % 100 == 0:
                # # 1. Create a copy of the model with EMA weights applied
                # # We don't want to overwrite the training model, so we copy the weights temporarily
                # # ema.store(noise_pred_net.parameters()) # Save original weights
                # ema.store(noise_pred_net)           # Save original weights

                # # ema.copy_to(noise_pred_net.parameters()) # Load EMA weights into model
                # ema.copy_to(noise_pred_net)         # Load EMA weights

                # # Get the state_dict of the EMA-averaged model
                # ema_model_state_dict = noise_pred_net.state_dict()
                
                # # Restore original weights to continue training correctly
                # # ema.restore(noise_pred_net.parameters())
                # ema.restore(noise_pred_net)         # Restore original weights


                torch.save({
                    'epoch': epoch_idx + 1,
                    'diffusion_model': noise_pred_net.state_dict(), # The current training weights
                    # 'ema_model': ema.model.state_dict(),              # The smoothed EMA weights
                    'encoder': encoder.state_dict(),
                    'hypernet': hypernet.state_dict(),
                    'stats': dataset.stats
                }, f"{checkpoint_dir}/phase2_epoch_{epoch_idx+1}.pth")

    print("Training Complete. Model Ready for Inference.")


if __name__ == '__main__':
    main()