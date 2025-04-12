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
from envir_pusht import PushTImageEnv
from model import ConditionalUnet1D
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
    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    # Create EMA models
    ema_nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
    })
    # Move models to GPU
    ema_nets = ema_nets.to(device)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    ema_nets.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint['stats']
    
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
    env = PushTImageEnv()
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
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # nimages = nimages.clone().detach().to(device, dtype=torch.float32)
                    image_features = ema_nets['vision_encoder'](nimages)
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                    noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                    naction = noisy_action
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        noise_pred = ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                naction = naction.detach().to('cpu').numpy()
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])
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
