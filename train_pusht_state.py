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

from model import ConditionalUnet1D
from dataset_pusht_state import PushTStateDataset
from vision_encoder import get_resnet, replace_bn_with_gn

@click.command()
@click.option('-i', '--dataset_path', required=True)
@click.option('-o', '--checkpoint_dir', default='ckpt/state', help='Directory to save checkpoints')

def main(dataset_path,checkpoint_dir):
    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    device = torch.device('cuda')

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
    print("batch['action'].shape", batch['action'].shape)

    #### Construct STATE only
    obs_dim = 5
    action_dim = 2

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    with torch.no_grad():
        # example inputs
        noised_action = torch.randn((1, pred_horizon, action_dim))
        obs = torch.zeros((1, obs_horizon, obs_dim))
        diffusion_iter = torch.zeros((1,))

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = noise_pred_net(
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))

        # illustration of removing noise
        # the actual noise removal is performed by NoiseScheduler
        # and is dependent on the diffusion noise schedule
        denoised_action = noised_action - noise
    
    # --- Scheduler ---
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    noise_pred_net.to(device)

    # Training Config
    num_epochs = 200

    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)
    
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            if (epoch_idx + 1) % 1 == 0:
                checkpoint_path = f"{checkpoint_dir}/pusht_checkpoint_epoch_{epoch_idx + 1}.pth"
                torch.save({
                    'epoch': epoch_idx + 1,
                    'model_state_dict': noise_pred_net.state_dict(),
                    'loss': epoch_loss,
                    'stats': stats
                }, checkpoint_path)
    
    checkpoint_path = f"{checkpoint_dir}/pusht_checkpoint_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': noise_pred_net.state_dict(),
        'loss': epoch_loss,
        'stats': stats
    }, checkpoint_path)

if __name__ == '__main__':
    main()
