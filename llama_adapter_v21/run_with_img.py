import argparse
import os
import random
import glob
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
from PIL import Image

from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, ColorJitter 
from scipy.stats import entropy
import statistics

import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
import zlib
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

import llama
import cv2
import sys
sys.path.insert(0,'../')

from metric_util import get_text_metric, get_img_metric, get_meta_metrics, convert, save_output
from eval import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--llama_path",
                        default = "")
    parser.add_argument("--num_gen_token", type=int, default=32)
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--dataset", type=str, default='img_Flickr')
    parser.add_argument("--output_dir", type=str, default="image_MIA")
    args = parser.parse_args()
    return args


def generate_text(model, img, text, gpu_id, num_gen_token):

    device = 'cuda:{}'.format(gpu_id)
    prompt = llama.format_prompt(text)
    img = preprocess(img).unsqueeze(0).to(device)

    output_text = model.generate(img, [prompt], max_gen_len = num_gen_token, temperature = 0, device = device)[0]

    return output_text

def evaluate_data(model, test_data, text, gpu_id, num_gen_token):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data

    for ex in tqdm(test_data): 

        description = generate_text(model, ex['image'], text, gpu_id, num_gen_token)
        # description = ''
        new_ex = inference(model, ex['image'], text, description, ex, gpu_id)

        all_output.append(new_ex)

    return all_output


def inference(model, img_path, text, description, ex, gpu_id):
    goal_parts = ['inst_desp','inst','desp']
    all_pred = {}

    if isinstance(img_path, Image.Image):
        image = img_path.convert('RGB')  
    else:
        image = Image.open(img_path).convert('RGB')

    # Define the transformations
    transform1 = RandomResizedCrop(size=(256, 256))
    aug1 = transform1(image)

    transform2 = RandomRotation(degrees=45)
    aug2 = transform2(image)

    transform3 = RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25))
    aug3 = transform3(image)

    transform4 = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    aug4 = transform4(image)
    
    for part in goal_parts:
        pred = {}
        metrics = mod_infer(model, image, text, description, gpu_id, part)
        metrics1 = mod_infer(model, aug1, text, description, gpu_id, part)
        metrics2 = mod_infer(model, aug2, text, description, gpu_id, part)
        metrics3 = mod_infer(model, aug3, text, description, gpu_id, part)
        metrics4 = mod_infer(model, aug4, text, description, gpu_id, part)

        aug1_prob = metrics1['log_probs']
        aug2_prob = metrics2['log_probs']
        aug3_prob = metrics3['log_probs']
        aug4_prob = metrics4['log_probs']

        ppl = metrics["ppl"]
        all_prob = metrics["all_prob"]
        p1_likelihood = metrics["loss"]
        entropies = metrics["entropies"]
        mod_entropy = metrics["modified_entropies"]
        max_p = metrics["max_prob"]
        org_prob = metrics["probabilities"]
        log_probs = metrics["log_probs"]
        gap_p = metrics["gap_prob"]
        renyi_05 = metrics["renyi_05"]
        renyi_2 = metrics["renyi_2"]

        mod_renyi_05 = metrics["mod_renyi_05"]
        mod_renyi_2 = metrics["mod_renyi_2"]

        pred = get_img_metric(ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05, renyi_2, log_probs, aug1_prob, aug2_prob, aug3_prob, aug4_prob,mod_renyi_05, mod_renyi_2)

        all_pred[part] = pred

    ex["pred"] = all_pred

    torch.cuda.empty_cache()

    return ex

@torch.inference_mode()
def logits_forward(model, tokens, visual_query):
    _bsz, seqlen = tokens.shape

    h = model.llama.tok_embeddings(tokens)
    freqs_cis = model.llama.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[:seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

    for layer in model.llama.layers[:-1 * model.query_layer]:
        h = layer(h, 0, freqs_cis, mask)

    adapter = model.adapter_query.weight.reshape(model.query_layer, model.query_len, -1).unsqueeze(1)
    adapter_index = 0
    for layer in model.llama.layers[-1 * model.query_layer:]:
        dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
        dynamic_adapter = dynamic_adapter + visual_query
        h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
        adapter_index = adapter_index + 1

    h = model.llama.norm(h)
    output = model.llama.output(h)
    # output = output[:, :-1, :]

    assert model.llama.vocab_size == 32000

    return output


def mod_infer(model, img, instruction, description, gpu_id, goal):
    device='cuda:{}'.format(gpu_id)

    img = preprocess(img).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        visual_query = model.forward_visual(img)

    prompt = llama.format_prompt(instruction) + description

    prompt_t = model.tokenizer.encode(prompt, bos=True, eos=False)

    tokens = torch.tensor(prompt_t).to(device).long().unsqueeze(0) 

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = logits_forward(model, tokens, visual_query)

    input_ids = tokens[0]
    
    descp_encoding = model.tokenizer.encode(description, bos=False, eos=False)

    goal_slice_dict = {
        'inst_desp' : slice(0, None),
        'inst' : slice(0,-len(descp_encoding)),
        'desp' : slice(-len(descp_encoding),None)
        } 

    target_slice = goal_slice_dict[goal]

    logits_slice = logits[0,target_slice,:]

    input_ids = input_ids[target_slice]

    probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)
    
    return get_meta_metrics(input_ids, probabilities, log_probabilities)

# ========================================
#             Model Initialization
# ========================================

if __name__ == '__main__':

    args = parse_args()
    llama_dir = args.llama_path
    device = 'cuda:{}'.format(args.gpu_id)

    # choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
    model, preprocess = llama.load("LORA-BIAS-7B-v21", llama_dir, llama_type="7B", device=device)
    model.eval()
    
    dataset = load_dataset("JaineLi/VL-MIA-image", args.dataset, split='train')
    data = convert_huggingface_data_to_list_dic(dataset)

    output_dir = f"{args.output_dir}/{args.dataset}/gen_{args.num_gen_token}_tokens"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info('=======Initialization Finished=======')

    text = "Please introduce this painting."

    all_output = evaluate_data(model, data, text, args.gpu_id, args.num_gen_token)

    fig_fpr_tpr_img(all_output, output_dir)



