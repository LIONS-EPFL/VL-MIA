import argparse
import os
import random
import glob
import logging
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from fastchat import model as fmodel

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.interact import Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

import sys
sys.path.insert(0,'../')
from metric_util import get_text_metric, get_img_metric, save_output, convert, get_meta_metrics
from eval import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path",
                        default = "eval_configs/minigpt4_eval_local.yaml",
                        help="path to configuration file.")
    parser.add_argument("--num_gen_token", type=int, default=32)
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--dataset", type=str, default='img_Flickr')
    parser.add_argument("--output_dir", type=str, default="image_MIA")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def generate_text(model, vis_processor, img, text, gpu_id, num_gen_token):
    chat = Interact(model, vis_processor, device='cuda:{}'.format(gpu_id))

    img_list = []

    chat_state = CONV_VISION.copy()

    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)

    chat.ask(text, chat_state)

    gen_ = chat.get_generate_output(conv=chat_state,
                                img_list=img_list,
                                max_new_tokens = num_gen_token,
                                do_sample=False
                            )

    output_text = chat.model.llama_tokenizer.decode(gen_[0], skip_special_tokens=True)

    return output_text

def evaluate_data(model, vis_processor, test_data, text, gpu_id, num_gen_token):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data

    for ex in tqdm(test_data): 

        description = generate_text(model, vis_processor, ex['image'], text, gpu_id, num_gen_token)
        # description = ''
        new_ex = inference(model, vis_processor, ex['image'], text, description, ex, gpu_id)

        all_output.append(new_ex)

    return all_output

def load_conversation_template(template_name):
        conv_template = fmodel.get_conversation_template(template_name)
        if conv_template.name == 'zero_shot':
            conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
            conv_template.sep = '\n'
        elif conv_template.name == 'llama-2':
            conv_template.sep2 = conv_template.sep2.strip()
        return conv_template

def inference(model, vis_processor, img_path, text, description, ex, gpu_id):
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
        metrics = mod_infer(model, vis_processor, image, text, description, gpu_id, part)
        metrics1 = mod_infer(model, vis_processor, aug1, text, description, gpu_id, part)
        metrics2 = mod_infer(model, vis_processor, aug2, text, description, gpu_id, part)
        metrics3 = mod_infer(model, vis_processor, aug3, text, description, gpu_id, part)
        metrics4 = mod_infer(model, vis_processor, aug4, text, description, gpu_id, part)
        
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


def mod_infer(model, vis_processor, img_path, instruction, description, gpu_id, goal):
    device='cuda:{}'.format(gpu_id)

    chat = Interact(model, vis_processor, device=device)

    logging.info('=======Chat established, load images=======')

    img_list = []

    chat_state = CONV_VISION.copy()

    llm_message = chat.upload_img(img_path, chat_state, img_list)
    chat.encode_img(img_list)

    chat.ask(instruction, chat_state)

    chat_state.append_message(chat_state.roles[1], None)

    chat_state.append_message(description, None)
    
    # print(chat_state.get_prompt())

    outputs, input_ids, seg_tokens = chat.get_output_by_emb(conv=chat_state,
                                img_list=img_list
                            )
    
    descp_encoding = chat.model.llama_tokenizer(description, return_tensors="pt", add_special_tokens = False).to(chat.device).input_ids

    logits = outputs.logits
    goal_slice_dict = {
        'img' : slice(seg_tokens[0].shape[1],-seg_tokens[1].shape[1]),
        # '<img>' : slice(-seg_tokens[-1].shape[1],-seg_tokens[-1].shape[1]+3),
        'inst_desp' : slice(-seg_tokens[-1].shape[1],None),
        'inst' : slice(-seg_tokens[-1].shape[1],-descp_encoding.shape[1]),
        'desp' : slice(-descp_encoding.shape[1],None)
        } 

    img_loss_slice = logits[0, goal_slice_dict['img'].start-1:goal_slice_dict['img'].stop-1, :]
    img_target_np = torch.nn.functional.softmax(img_loss_slice, dim=-1).cpu().numpy()
    max_indices = np.argmax(img_target_np, axis=-1)
    img_max_input_id = torch.from_numpy(max_indices).to(device)

    mix_input_ids = torch.cat([seg_tokens[0][0], img_max_input_id, seg_tokens[1][0]], dim=0)

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
        
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
            'pretrain_llama2': CONV_VISION_LLama2}

    logging.info('=======Initializing Chat=======')
    args = parse_args()
    cfg = Config(args)
    # print(args.cfg_path)
    model_config = cfg.model_cfg

    # print(model_config)
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(
        args.gpu_id))
    num_gen_token = args.num_gen_token

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)

    dataset = load_dataset("JaineLi/VL-MIA-image", args.dataset, split='train')
    data = convert_huggingface_data_to_list_dic(dataset)

    # data = data[:10]

    output_dir = f"{args.output_dir}/{args.dataset}/gen_{args.num_gen_token}_tokens"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info('=======Initialization Finished=======')

    text = 'Describe this image in detail.'

    all_output = evaluate_data(model, vis_processor, data, text, args.gpu_id, num_gen_token)

    fig_fpr_tpr_img(all_output, output_dir)



