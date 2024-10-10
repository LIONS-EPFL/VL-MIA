import argparse
import torch
import math

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re

import torch.nn as nn
import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from eval import *
import pdb
from datasets import load_dataset

import sys
# sys.path.insert(0, '../')
from metric_util import get_text_metric, get_img_metric, save_output, get_meta_metrics

def shuffle_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)    
    shuffled_sentence = ' '.join(words)
    return shuffled_sentence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="text_MIA")
    parser.add_argument('--dataset', type=str, default="llava_v15_gpt_text")
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--text_len", type=int, default=32)
    args = parser.parse_args()
    return args


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def evaluate_data(model, image_processor, conv_mode, test_data, col_name, gpu_id):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data

    for ex in tqdm(test_data): 
        img = Image.new('RGB', (1024, 1024), color = 'black')
        text = ex[col_name]
        new_ex = inference(model, image_processor, conv_mode, img, text, ex, gpu_id)

        all_output.append(new_ex)

    return all_output

def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode

def inference(model, vis_processor, conv_mode, img, text, ex, gpu_id):
    
    pred = {}
    
    metrics = mod_infer(model, vis_processor, conv_mode, img, text, gpu_id)
    metrics_lower = mod_infer(model, vis_processor, conv_mode, img, text.lower(), gpu_id)

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

    return ex


def mod_infer(model, image_processor, conv_mode, img, description, gpu_id):
    device='cuda:{}'.format(gpu_id)

    qs = ""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], description)
    prompt = conv.get_prompt()[:-4]

    images = [img]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda(gpu_id)
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            images=images_tensor,
            image_sizes=image_sizes
        )
    
    logits = outputs.logits
    goal_slice_dict = {
        'img' : slice(len(prompt_chunks[0]),-len(prompt_chunks[-1])+1),
        # 'inst' : slice(-len(prompt_chunks[-1])+3,-5),  # wo sys tokens
        'inst' : slice(-len(prompt_chunks[-1])+1,None), # include sys tokens
        } 

    img_loss_slice = logits[0, goal_slice_dict['img'].start-1:goal_slice_dict['img'].stop-1, :]
    img_target_np = torch.nn.functional.softmax(img_loss_slice, dim=-1).cpu().numpy()
    max_indices = np.argmax(img_target_np, axis=-1)
    img_max_input_id = torch.from_numpy(max_indices).to(device)

    tensor_a = torch.tensor(prompt_chunks[0]).to(device) if not isinstance(prompt_chunks[0], torch.Tensor) else prompt_chunks[0]
    tensor_b = torch.tensor(prompt_chunks[-1][1:]).to(device) if not isinstance(prompt_chunks[-1][1:], torch.Tensor) else prompt_chunks[-1][1:]

    mix_input_ids = torch.cat([tensor_a, img_max_input_id, tensor_b], dim=0)

    target_slice = goal_slice_dict['inst']

    logits_slice = logits[0,target_slice,:]

    if target_slice.stop is None:
        loss_slice = logits[0, target_slice.start - 1 : -1, :]
    else:
        loss_slice = logits[0, target_slice.start - 1 : target_slice.stop - 1, :]

    input_ids = mix_input_ids[target_slice]

    # loss = nn.CrossEntropyLoss()(loss_slice, input_ids)

    probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)
    
    return get_meta_metrics(input_ids, probabilities, log_probabilities)


# ========================================
#             Model Initialization
# ========================================

if __name__ == '__main__':

    args = parse_args()

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, gpu_id = args.gpu_id
    )
    conv_mode = load_conversation_template(model_name)

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    text_len = args.text_len
    output_dir = f"{args.output_dir}/length_{text_len}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("JaineLi/VL-MIA-text", args.dataset, split=f"length_{text_len}")
    data = convert_huggingface_data_to_list_dic(dataset)

    logging.info('=======Initialization Finished=======')

    all_output = evaluate_data(model, image_processor, conv_mode, data, 'input', args.gpu_id)

    fig_fpr_tpr(all_output, output_dir)



