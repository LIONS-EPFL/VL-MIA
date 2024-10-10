# 🔦 Membership Inference Attacks against Large Vision-Language Models

This repository provides an official implementation Membership Inference Attacks against Large Vision-Language Models.

## 🔍 Overview

We explore membership inference attack(MIA) on VLLMs(large vision-language models):
- We release the first benchmark tailored for detecting training data in VLLMs, called Vision Language MIA (VL-MIA). 
- We also perform the first individual image or description MIAs on VLLMs in a cross-modal manner. 
- We propose a target-free MIA metric, MaxRényi-K%, and its modified target-based ModRényi.

## 🎞️ VL-MIA Datasets

The **VL-MIA datasets** serve as a benchmark designed to evaluate membership inference attack (MIA) methods for large vision language models. Access our **VL-MIA datasets** directly on [image](https://huggingface.co/datasets/JaineLi/VL-MIA-image) and [text](https://huggingface.co/datasets/JaineLi/VL-MIA-text) . 

#### Loading the Datasets

```python
from datasets import load_dataset

text_len = 64 # 16,32,64

img_subset = "img_Flickr" # or img_dalle
text_subset = "llava_v15_gpt_text" # or minigpt4_stage2_text

image_dataset = load_dataset("JaineLi/VL-MIA-image", subset, split='train')
text_dataset = load_dataset("JaineLi/VL-MIA-text", subset, split=f"length_{text_len}")
```

* *Label 0*: Refers to the unseen data during pretraining. *Label 1*: Refers to the seen data.


## 🚀 Run MIA
Use the following command to prepare the enviroment after navigating to the repo folder:
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Use the following command to run the MIA:
```bash
python run_with_img.py --gpu_id 0 --num_gen_token 32 --dataset img_Flickr
```
```bash
python run_with_text.py --gpu_id 0 --text_len 32 --dataset llava_v15_gpt_text
```


## Cite as:

```
@inproceedings{zhan2024mia,
  author = {Li*, Zhan and Wu*, Yongtao and Chen*, Yihang and Tonin, Francesco and Abad Rocamora, Elias and Cevher, Volkan},

  title = {Membership Inference Attacks against Large Vision-Language Models},

  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},

  year = {2024}
}
```