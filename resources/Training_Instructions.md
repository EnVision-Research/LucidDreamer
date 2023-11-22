# Preparations

## Cloning the Repository
The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:EnVision-Research/LucidDreamer.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/EnVision-Research/LucidDreamer.git --recursive
```
## Setup
Our default, provided install method is based on Conda package.
Firstly, you need to create an virtual environment and install the submodoules we provide. (slightly difference from original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting))
```shell
conda env create --file environment.yml
conda activate LucidDreamer
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
```
Then, you need to install [Point-E](https://github.com/openai/point-e) follow the instruction under this repo:
```shell
https://github.com/openai/point-e
```

# Running
We will provide a detailed guideline of our implementation about the description of each hyperparameter and how to tune them later. Now, we release 9 config files for you to evaluate the effectiveness of our framework (all configs can be trained in a single RTX3090).

Firstly, you may need to change ```model_key:``` in the ```configs\<config_file>.yaml``` to link the local Pretrained Diffusion Models ( [Stable Diffusion 2.1-base](https://github.com/Stability-AI/StableDiffusion) in default)


Then, you can simply use:

```shell
python train.py --opt <path to config file>
```
or you can see an exmaple and use the script we provide after you identify the visualable GPU:
```shell
bash train.sh
```
We provide config files in ```configs\``` that serve for different tasks:

Text-to-3D generation:
```shell
axe.yaml
bagel.yaml
cat_armor.yaml
crown.yaml
football_helmet.yaml
hamburger.yaml
white_hair_ironman.yaml
zombie_joker.yaml
```

Personalized Text-to-3D:
```shell
ts_lora.yaml
```
You can also use your own LoRA thourgh modify the: ```LoRA_path:```
