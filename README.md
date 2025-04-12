# RS2_Diffusion_Policy

## Acknowledgement
This code was based on Diffusion Policy [paper](https://diffusion-policy.cs.columbia.edu/#paper) and notebooks for [vision-based environment](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing).

The code has been modified to allow users to collect their own data in simulation instead of using the data provided by the authors.

The system has been tested on Ubuntu 22.04 and 20.04.

## Installation
To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU.
First install Conda Environment. We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation.

This command will create the virtual environment named "RS2Diffusion" and install all the packages.
```console
$ conda env create -f conda_environment.yaml
```

## Creating Training Data Set
Activate the environment:
```console
(base)$ conda activate RS2diffusion
```
Run the data collection script:
```console
(RS2diffusion)$ python3 collect_data_pusht.py -o data/pusht_data.zarr
```

Instruction for collecting data in the simulation: 

- Hover mouse close to the blue circle to start.

- Push the T block into the green area. 

- The episode will automatically terminate if the task is succeeded.

- Press "Q" to exit and save the data.

- Press "R" to retry.

NOTE: The dataset should contain approximately 200 demonstrations, which takes around an hour to collect. You can exit and save by pressing 'Q' and resume data collection later if needed. To continue collecting data, rerun the command above and ensure you use the same dataset name.

[This Dataset](https://drive.google.com/drive/folders/1LSFfpA6JL-Ugn6-Hid7w1qFYJZ88rBVC?usp=sharing) was collected using 200 demonstrations.


## Training 
The model will train on the input dataset and save checkpoints in the 'ckpt' folder.
Training will take approximately a few hours, depending on your GPU.
A GPU with more than 8GB of VRAM is recommended.
This code has been tested on NVIDIA RTX 3060 and 2080 Ti, where the training time is approximately 1 hour.
```console
(RS2diffusion)$ python3 train_pusht.py -i data/pusht_data.zarr
```
[This Model](https://drive.google.com/file/d/1sJOWmka15V7nL71jXH9Qs31aAw2zF9Bm/view?usp=sharing) was trained for 200 epochs. 


## Inference
```console
(RS2diffusion)$ python3 inference_pusht.py -i ckpt/pusht_checkpoint_final.pth
```
Note: When the simulation finishes, it will generate 10 MP4 files (by default) in the "videos" folder for evaluation. The evaluation data will not be the same as the training data.
A run is considered a success if the model pushes the block to cover at least 90% of the goal within 300 steps. Each video filename will include either 'failed' or 'succeeded' to indicate the outcome

If you want to generate more or less videos to evaluate:
```console
(RS2diffusion)$ python3 inference_pusht.py -i ckpt/pusht_checkpoint_final.pth -n "number_videos"
```

## Pertubation testing
```console
(RS2diffusion)$ python3 pertubation_testing.py -i ckpt/pusht_checkpoint_final.pth
```
Instruction for pertubating the T-block in the simulation: 

- Press "W", "S", "A", or "D" to move the block upward, downward, to the left, or to the right, respectively. These controls apply within the simulation application, not in the terminal.

- Press "C" to exit.

- The simulation will stop and save automatically after 500 steps or when the task is completed.

- Note: The diffusion policy does not complete the task with a 100% success rate, so in some cases, it may fail to recover after you perturb the block. It is better to perturb the block when the task is nearly completed.
