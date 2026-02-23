
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import os
from skvideo.io import vwrite

# Import your modules
# Ensure these files are in the same directory or python path
from envir_pusht import PushTEnv
from model import ConditionalUnet1D
from dataset_pusht import normalize_data, unnormalize_data
import click

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
    action_horizon = 8  # How many steps to execute the generated weights
    action_dim = 2
    obs_dim = 5

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon, # Condition on Obs
    ).to(device)

    # --- 3. Load Checkpoint ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint['stats']
        
    # Scheduler
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # --- 4. Evaluation Loop ---
    max_steps = 300
    env = PushTEnv()
    number_success_video = 0
    
    for video_idx in range(num_videos):
        seed = np.random.randint(201, 100000)
        seed = seed +1
        env.seed(seed)
        obs, info = env.reset() # (obs_dim,)

        # History Queue for Conditioning
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        imgs = [env.render(mode='rgb_array')]
        rewards = []
        done = False
        step_idx = 0
        
        with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
        
        # Save Video
        print(f'Score for video {video_idx+1}: ', max(rewards))
        if max(rewards) >= 0.9:
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
