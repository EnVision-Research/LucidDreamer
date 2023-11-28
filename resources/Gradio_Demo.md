# Gradio Demo

- For online demo, please click [**here**](https://huggingface.co/spaces/haodongli/LucidDreamer).
- However, it is highly advisable to run the demo locally, as the live demo may take a considerable amount of time.
- Below is a straightforward guide for building the Gradio demo locally :smile:

## Cloning the HuggingFace Repository

```shell
git lfs install
git clone https://huggingface.co/spaces/haodongli/LucidDreamer.git
cd LucidDreamer
```

## Setup

```shell
conda create -n LD_Demo python=3.9.16 cudatoolkit=11.8
conda activate LD_Demo
pip install -r pre-requirements.txt
pip install -r requirements.txt
```

## Running

```shell
LD_CUDA=0 # select a graphic card
python gradio_demo.py --cuda $LD_CUDA
```