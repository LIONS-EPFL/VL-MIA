import argparse
import torch

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

from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, ColorJitter 
from scipy.stats import entropy
import statistics

import torch.nn as nn
import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
import zlib
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from eval import *

import sys
# sys.path.insert(0, '../')
from metric_util import get_text_metric, get_img_metric, save_output, convert, get_meta_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_gen_token", type=int, default=32)
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--dataset", type=str, default='img_Flickr')
    parser.add_argument("--output_dir", type=str, default="image_MIA")
    parser.add_argument("--severity", type=int, default=6)
    args = parser.parse_args()
    return args


def load_image(image_file):
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
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

def generate_text(model, image_processor, conv_mode, img, text, gpu_id, num_gen_token):
    qs = text
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
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images([img])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda(gpu_id)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_token,
            use_cache=True,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output_text

def evaluate_data(model, image_processor, conv_mode, test_data, text, gpu_id, num_gen_token):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data

    for ex in tqdm(test_data): 
        description = generate_text(model, image_processor, conv_mode, ex['image'], text, gpu_id, num_gen_token)
        # description = ''
        new_ex = inference(model, image_processor, conv_mode, ex['image'], text, description, ex, gpu_id)

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

def inference(model, vis_processor, conv_mode, img_path, text, description, ex, gpu_id):
    goal_parts = ['img','inst_desp','inst','desp']
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
        metrics = mod_infer(model, vis_processor, conv_mode, image, text, description, gpu_id, part)
        metrics1 = mod_infer(model, vis_processor, conv_mode, aug1, text, description, gpu_id, part)
        metrics2 = mod_infer(model, vis_processor, conv_mode, aug2, text, description, gpu_id, part)
        metrics3 = mod_infer(model, vis_processor, conv_mode, aug3, text, description, gpu_id, part)
        metrics4 = mod_infer(model, vis_processor, conv_mode, aug4, text, description, gpu_id, part)


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


def mod_infer(model, image_processor, conv_mode, img, instruction, description, gpu_id, goal):
    device='cuda:{}'.format(gpu_id)

    qs = instruction
    # qs = ''
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
    
    descp_encoding = tokenizer(description, return_tensors="pt", add_special_tokens = False).to(device).input_ids

    logits = outputs.logits
    goal_slice_dict = {
        'img' : slice(len(prompt_chunks[0]),-len(prompt_chunks[-1])+1),
        'inst_desp' : slice(-len(prompt_chunks[-1])+1,None),
        'inst' : slice(-len(prompt_chunks[-1])+1,-descp_encoding.shape[1]),
        'desp' : slice(-descp_encoding.shape[1],None)
        } 

    img_loss_slice = logits[0, goal_slice_dict['img'].start-1:goal_slice_dict['img'].stop-1, :]
    img_target_np = torch.nn.functional.softmax(img_loss_slice, dim=-1).cpu().numpy()
    max_indices = np.argmax(img_target_np, axis=-1)
    img_max_input_id = torch.from_numpy(max_indices).to(device)

    tensor_a = torch.tensor(prompt_chunks[0]).to(device) if not isinstance(prompt_chunks[0], torch.Tensor) else prompt_chunks[0]
    tensor_b = torch.tensor(prompt_chunks[-1][1:]).to(device) if not isinstance(prompt_chunks[-1][1:], torch.Tensor) else prompt_chunks[-1][1:]

    mix_input_ids = torch.cat([tensor_a, img_max_input_id, tensor_b], dim=0)

    target_slice = goal_slice_dict[goal]

    logits_slice = logits[0,target_slice,:]

    input_ids = mix_input_ids[target_slice]

    probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)
    
    return get_meta_metrics(input_ids, probabilities, log_probabilities)

# ========================================
#             Model Initialization
# ========================================

if __name__ == '__main__':

    args = parse_args()
    num_gen_token = args.num_gen_token
    dataset = args.dataset

    #For corruption
    severity = args.severity

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

    dataset = load_dataset("JaineLi/VL-MIA-image", dataset, split='train')
    data = convert_huggingface_data_to_list_dic(dataset)

    output_dir = f"{args.output_dir}/{args.dataset}/gen_{num_gen_token}_tokens"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info('=======Initialization Finished=======')

    text = 'Describe this image concisely.'

    all_output = evaluate_data(model, image_processor, conv_mode, data, text, args.gpu_id, num_gen_token)

    fig_fpr_tpr_img(all_output, output_dir)
