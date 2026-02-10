
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import os
import click
from skvideo.io import vwrite

# Import your modules
# Ensure these files are in the same directory or python path
from envir_pusht import PushTEnv
from model import FunctionalPolicy, LatentUnet1D, HyperNetwork, LatentDiffusionUNet
from dataset_pusht import normalize_data, unnormalize_data

@click.command()
@click.option('-i', '--checkpoint_path', required=True, help='Path to the .pth checkpoint')
@click.option('-n', '--num_videos', default=10, help='Number of videos to generate')
@click.option('-o', '--video_folder', default='videos', help='Folder to save videos')
def main(checkpoint_path, num_videos, video_folder):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Parameters (Must match Training) ---
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 16  # How many steps to execute the generated weights
    action_dim = 2
    obs_dim = 5
    state_dim = 5       # Input to the functional policy
    latent_dim = 256    # Updated to 256 based on previous phase 2 training
    hidden_dim = 256    # MLP Hidden Size
    
    # Context dim for Diffusion (Flattened observation history)
    global_cond_dim = obs_horizon * obs_dim 
    
    # Define the architecture of the Policy the HyperNetwork generates
    # CRITICAL: Must match the shapes used in Training (State -> Action)
    policy_shapes = {
        'fc1.weight': (hidden_dim, state_dim),
        'fc1.bias':   (hidden_dim,),
        'fc2.weight': (hidden_dim, hidden_dim),
        'fc2.bias':   (hidden_dim,),
        'fc3.weight': (action_dim, hidden_dim), # Output is 2 (action_dim)
        'fc3.bias':   (action_dim,)
    }
    
    # --- 2. Initialize Models ---
    # The Diffusion Model (Generates Latent Z)
    # noise_pred_net = LatentUnet1D(
    #     latent_dim=latent_dim,
    #     global_cond_dim=global_cond_dim, # Condition on history
    # ).to(device)

    noise_pred_net = LatentDiffusionUNet(
        input_dim=latent_dim,
        cond_dim=obs_dim*obs_horizon, # Condition on Obs
    ).to(device)
    
    # The HyperNetwork (Decodes Z -> Policy Weights)
    hypernet = HyperNetwork(latent_dim, policy_shapes).to(device)
    
    # --- 3. Load Checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    noise_pred_net.load_state_dict(checkpoint['diffusion_model'])
    hypernet.load_state_dict(checkpoint['hypernet'])
    stats = checkpoint['stats']
    
    noise_pred_net.eval()
    hypernet.eval()
    
    # Scheduler
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # --- 4. Evaluation Loop ---
    max_steps = 500
    env = PushTEnv()
    number_success_video = 0
    
    for video_idx in range(num_videos):
        seed = np.random.randint(201, 100000)
        env.seed(seed)
        obs, info = env.reset() # (obs_dim,)

        # History Queue for Conditioning
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        imgs = [env.render(mode='rgb_array')]
        rewards = []
        done = False
        step_idx = 0
        
        # Progress bar for this episode
        pbar = tqdm(total=max_steps, desc=f"Video {video_idx+1}", leave=False)
        
        while not done and step_idx < max_steps:
            # A. Prepare Conditioning (Observation History)
            obs_seq = np.stack(obs_deque) # (obs_horizon, obs_dim)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs_tensor = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            
            # Flatten: (1, obs_horizon * obs_dim)
            obs_cond = nobs_tensor.unsqueeze(0).flatten(start_dim=1)

            # B. Generate Policy Weights (The "Skill")
            with torch.no_grad():    
                # 1. Sample Noise
                latent_z = torch.randn((1, latent_dim), device=device)
                
                # 2. Diffusion Denoising
                noise_scheduler.set_timesteps(num_diffusion_iters)
                for k in noise_scheduler.timesteps:
                    # Model needs batched, CUDA, float timesteps
                    t_model = k.expand(latent_z.shape[0]).to(device).float()

                    # Scheduler needs scalar timestep (CPU is fine)
                    t_sched = k

                    noise_pred = noise_pred_net(
                        x=latent_z,
                        timesteps=t_model,
                        context=obs_cond
                    )

                    latent_z = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=t_sched,
                        sample=latent_z
                    ).prev_sample
                
                # 3. Decode Z -> Weights
                weights = hypernet(latent_z)
            
            # C. Execute Policy (Closed-Loop for action_horizon steps)
            # We use the SAME weights for 'action_horizon' steps (Temporal Consistency)
            for _ in range(action_horizon):
                if done or step_idx >= max_steps:
                    break
                
                # 1. Prepare Current State for Policy
                # Note: FunctionalPolicy expects (Batch, State_Dim)
                curr_state = obs_deque[-1] # Get most recent obs
                n_curr_state = normalize_data(curr_state, stats=stats['state'])
                n_curr_state_tensor = torch.from_numpy(n_curr_state).to(device, dtype=torch.float32).unsqueeze(0)
                
                # 2. Query Policy
                with torch.no_grad():
                    # Apply weights to CURRENT state
                    n_action_tensor = FunctionalPolicy.apply(n_curr_state_tensor, weights)
                
                # 3. Unnormalize Action
                n_action = n_action_tensor.cpu().numpy()[0] # (action_dim,)
                action = unnormalize_data(n_action, stats=stats['action'])
                
                # 4. Step Environment
                obs, reward, done, info = env.step(action)
                
                # 5. Update History & stats
                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))
                step_idx += 1
                pbar.update(1)
                
                if reward >= 1.0: # Success condition usually 1.0 for PushT
                    done = True

        pbar.close()
        
        # Save Video
        max_reward = max(rewards) if rewards else 0
        print(f'Video {video_idx+1} Score: {max_reward:.4f}')
        
        if max_reward > 0.95: # Success threshold
            number_success_video += 1
            filename = f'video_{video_idx+1}_success.mp4'
        else: 
            filename = f'video_{video_idx+1}_failed.mp4'
            
        video_path = os.path.join(video_folder, filename)
        vwrite(video_path, imgs)
        print(f'Saved to {video_path}')
        
    print(f'\nFinal Success Rate: {number_success_video / num_videos * 100:.2f}%')

if __name__ == '__main__':
    main()
