# Gradio Demo

- For online demo, please click [**here**](https://huggingface.co/spaces/haodongli/LucidDreamer).
- However, it is highly advisable to run the demo locally, as the live demo may take a considerable amount of time.
- Below is a straightforward guide for building the Gradio demo locally :smile:

## Cloning the :hugs: Repository

```shell
git lfs install
git clone https://huggingface.co/spaces/haodongli/LucidDreamer
cd LucidDreamer
```

## Setup

- Please **skip** this step if you have already built conda env. `LucidDreamer` following [this guide](Training_Instructions.md).

```shell
conda create -n LD_Demo python=3.9.16 cudatoolkit=11.8 -y
conda activate LD_Demo
pip install -r pre-requirements.txt
pip install -r requirements.txt
pip install git+https://github.com/YixunLiang/simple-knn.git
pip install git+https://github.com/YixunLiang/diff-gaussian-rasterization.git
```

## Running

```shell
LD_CUDA=0 # select a graphic card
python app.py --cuda $LD_CUDA
```
