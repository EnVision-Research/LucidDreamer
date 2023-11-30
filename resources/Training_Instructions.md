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
conda create -n LucidDreamer python=3.9.16 cudatoolkit=11.8
conda activate LucidDreamer
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
```
# Running
We will provide a detailed guideline of our implementation about the description of each hyperparameter and how to tune them later. Now, we release 9 config files for you to evaluate the effectiveness of our framework (all configs can be trained in a single RTX3090).

The pre-trained model will be downloaded automatically. You can also change ```model_key:``` in the ```configs\<config_file>.yaml``` to link the local Pretrained Diffusion Models ( [Stable Diffusion 2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) in default)


Then, you can use:

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
