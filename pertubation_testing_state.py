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
from model import ConditionalUnet1D
from vision_encoder import get_resnet, replace_bn_with_gn
from dataset_pusht import normalize_data, unnormalize_data
import click
import pygame
import cv2
from pymunk.vec2d import Vec2d
import time


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
    
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    # Move models to GPU
    noise_pred_net = noise_pred_net.to(device)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint['stats']
    
    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    max_steps = 500
    step_idx = 0
    
    env = PushTEnv()
    clock = pygame.time.Clock()
    done = False
    # limit enviornment interaction to 200 steps before termination
    
    pygame.display.set_caption(f'perturbation_testing')
    # seed = np.random.randint(201, 25536)
    seed = 12157
    env.seed(seed)
    # get first observation
    obs, info = env.reset()
    imgs = [env.render(mode='rgb_array')]
    
    while not done:
        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        B = 1
        # stack the last obs_horizon number of observations
        obs_seq = np.stack(obs_deque)

        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats['obs'])

        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
        
        # infer action
        with torch.no_grad():
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
            noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
            naction = noisy_action
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                noise_pred = noise_pred_net(
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
            imgs.append(env.render(mode='rgb_array'))
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        env.block.position += Vec2d(0, -20)  # Move up
                    elif event.key == pygame.K_s:
                        env.block.position += Vec2d(0, 20)  # Move down
                    elif event.key == pygame.K_a:
                        env.block.position += Vec2d(-20, 0)  # Move left
                    elif event.key == pygame.K_d:
                        env.block.position += Vec2d(20, 0)  # Move right
                    elif event.key == pygame.K_q:
                        env.block.angle -= 0.2  # Rotate counter-clockwise
                    elif event.key == pygame.K_e:
                        env.block.angle += 0.2  # Rotate clockwise
                    elif event.key == pygame.K_c:
                        video_path = os.path.join(video_folder, f'pertubation_testing.mp4')
                        vwrite(video_path, imgs)
                        exit(0)
                clock.tick(10)
                step_idx += 1
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    video_path = os.path.join(video_folder, f'pertubation_testing.mp4')
    vwrite(video_path, imgs)
        
if __name__ == '__main__':
    main()
