#@markdown ### **Imports**
# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import os
# env import
from skvideo.io import vwrite
from envir_pusht import PushTEnv
from model import FunctionalPolicy, LatentUnet1D, HyperNetwork
from vision_encoder import get_resnet, replace_bn_with_gn
from dataset_pusht import normalize_data, unnormalize_data
import click

@click.command()
@click.option('-i', '--checkpoint_path', required=True)
@click.option('-n', '--num_videos', default=10, help='Number of videos to generate')
@click.option('-o', '--video_folder', default='videos', help='Folder to save videos')
def main(checkpoint_path, num_videos, video_folder):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Dataset path {checkpoint_path} does not exist.")
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    # device transfer
    device = torch.device('cuda')
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    action_dim = 2
    obs_dim = 5
    latent_dim = 128
    
    policy_input_dim = obs_horizon * obs_dim
    policy_output_dim = pred_horizon * action_dim
    
    # Define the shape of the weights our HyperNetwork generates
    policy_shapes = {
        'fc1.weight': (128, policy_input_dim),
        'fc1.bias':   (128,),
        'fc2.weight': (128, 128),
        'fc2.bias':   (128,),
        'fc3.weight': (policy_output_dim, 128),
        'fc3.bias':   (policy_output_dim,)
    }
    
    # --- 2. Initialize Models ---
    # The Diffusion Model (Generates Latent Z)
    noise_pred_net = LatentUnet1D(
        latent_dim=latent_dim,
        global_cond_dim=policy_input_dim,
        down_dims=[64, 128, 256] 
    ).to(device)
    
    # The HyperNetwork (Decodes Z -> Policy Weights)
    hypernet = HyperNetwork(latent_dim, policy_shapes).to(device)
    
    # --- 3. Load Checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dicts
    noise_pred_net.load_state_dict(checkpoint['diffusion_model'])
    hypernet.load_state_dict(checkpoint['hypernet'])
    stats = checkpoint['stats']
    
    noise_pred_net.eval()
    hypernet.eval()
    
    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # limit enviornment interaction to 200 steps before termination
    max_steps = 300
    env = PushTEnv()
    # number of videos that reach the goal
    number_success_video = 0
    
    for video_idx in range(num_videos):
        # use a seed >200 to avoid initial states seen in the training dataset
        seed = np.random.randint(201, 25536)
        print("Seed:",seed)
        env.seed(seed)
        
        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        imgs = [env.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc=f"Eval PushTImageEnv Video {video_idx+1}/{num_videos}") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # Tensorize
                nobs_tensor = torch.from_numpy(nobs).to(device, dtype=torch.float32)
                # Flatten obs for condition: (1, obs_horizon * obs_dim)
                obs_cond = nobs_tensor.unsqueeze(0).flatten(start_dim=1)

                with torch.no_grad():    
                    # 1. Start with random noise for Latent Z
                    latent_z = torch.randn((B, latent_dim), device=device)
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                    # 2. Denoise Z

                    for k in noise_scheduler.timesteps:
                        noise_pred = noise_pred_net(
                            sample=latent_z,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        latent_z = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=latent_z
                        ).prev_sample
                    
                    # 3. Decode Z -> Weights using HyperNetwork
                    weights = hypernet(latent_z)
                    # 4. Execute Functional Policy
                    # This applies the generated weights to the observation
                    naction_flat = FunctionalPolicy.apply(obs_cond, weights)
                    
                    # Reshape to (B, Horizon, Action_Dim)
                    naction = naction_flat.reshape(B, pred_horizon, action_dim)

                # --- EXECUTION ---
                # Unnormalize
                naction = naction.detach().to('cpu').numpy()[0] # (Horizon, Action_Dim)
                action_pred = unnormalize_data(naction, stats=stats['action'])
                
                # Receding Horizon Control
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]

                for i in range(len(action)):
                    obs, reward, done, info = env.step(action[i])
                    obs_deque.append(obs)
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        print(f'Score for video {video_idx+1}: ', max(rewards))
        if max(rewards) == 1:
            number_success_video += 1
            video_path = os.path.join(video_folder, f'video_{video_idx+1}_succeeded.mp4')
        else: 
            video_path = os.path.join(video_folder, f'video_{video_idx+1}_failed.mp4')
        vwrite(video_path, imgs)
        print(f'Video saved to {video_path}')
        
    # Calculate and print success rate
    success_rate = number_success_video / num_videos
    print(f'Success rate: {success_rate * 100:.2f}%')

if __name__ == '__main__':
    main()
