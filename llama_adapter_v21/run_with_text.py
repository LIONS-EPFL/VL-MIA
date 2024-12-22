import argparse
import os
import random
import glob
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

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
from metric_util import get_text_metric, get_img_metric, save_output, convert, get_meta_metrics
from eval import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--llama_path",
                        default = "")
    parser.add_argument("--gpu_id",
                        type=int,
                        default=0,
                        help="specify the gpu to load the model.")
    parser.add_argument("--text_len", type=int, default=32)
    parser.add_argument('--dataset', type=str, default="llava_v15_gpt_text")
    parser.add_argument('--output_dir', type=str, default="text_MIA")
    args = parser.parse_args()
    return args


def evaluate_data(model, test_data, col_name, gpu_id):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data

    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model, text, ex, gpu_id)
        all_output.append(new_ex)

    return all_output


def inference(model, text, ex, gpu_id):
    pred = {}

    metrics = mod_infer(model,text,gpu_id)
    metrics_lower = mod_infer(model,text.lower(),gpu_id)
    
    ppl = metrics["ppl"]
    all_prob = metrics["all_prob"]
    p1_likelihood = metrics["loss"]
    entropies = metrics["entropies"]
    mod_entropy = metrics["modified_entropies"]
    max_p = metrics["max_prob"]
    org_prob = metrics["probabilities"]
    gap_p = metrics["gap_prob"]
    renyi_05 = metrics["renyi_05"]
    renyi_2 = metrics["renyi_2"]
    mod_renyi_05 = metrics["mod_renyi_05"]
    mod_renyi_2 = metrics["mod_renyi_2"]

    ppl_lower = metrics_lower["ppl"]

    pred = get_text_metric(ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05, renyi_2, text, ppl_lower,mod_renyi_05, mod_renyi_2)

    ex["pred"] = pred

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

def mod_infer(model, text, gpu_id):
    device='cuda:{}'.format(gpu_id)

    img = Image.new('RGB', (1024, 1024), color = 'black')
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        visual_query = model.forward_visual(img)

    prompt = llama.format_prompt("") + text

    prompt_t = model.tokenizer.encode(prompt, bos=True, eos=False)

    tokens = torch.tensor(prompt_t).to(device).long().unsqueeze(0) 

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = logits_forward(model, tokens, visual_query)

    input_ids = tokens[0]

    descp_encoding = model.tokenizer.encode(text, bos=False, eos=False)

    target_slice = slice(-len(descp_encoding)-10,None)

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

    text_len = args.text_len
    output_dir = f"{args.output_dir}/length_{text_len}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("JaineLi/VL-MIA-text", args.dataset, split=f"length_{text_len}")
    data = convert_huggingface_data_to_list_dic(dataset)

    logging.info('=======Initialization Finished=======')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_output = evaluate_data(model, data, 'input', args.gpu_id)

    fig_fpr_tpr(all_output, output_dir)



