# Multi-direction and Multi-scale Pyramid in Transformer for Video-based Pedestrian Retrieval

## Getting Started

### Requirements

Here is a brief instruction for installing the experimental environment.

```
# install virtual envs
$ conda create --prefix ./env python=3.8 -y
$ conda activate ./env -y


$ conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install conda packages
$ conda install scipy pandas opencv pillow matplotlib tensorboard -y
$ conda install -c conda-forge einops -y

# Install pip packages
pip install timm

```

### Create Simple Config System (No YACS)

Create file: simple_config.py in the PiT root directory

### Create Evaluation Script

evaluate_mevid.py in the PiT root directory.

## Run the Evaluation

```
# Run evaluation
$ python evaluate_mevid.py

```
