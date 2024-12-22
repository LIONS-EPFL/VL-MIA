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

import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

import sys
sys.path.insert(0,'../')
from metric_util import get_text_metric, get_img_metric, get_meta_metrics, convert, save_output
from eval import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path",
                        default = "eval_configs/minigpt4_eval_local.yaml",
                        help="path to configuration file.")
    parser.add_argument("--gpu_id",
                        type=int,
                        default=0,
                        help="specify the gpu to load the model.")
    parser.add_argument('--output_dir', type=str, default="text_MIA")
    parser.add_argument('--dataset', type=str, default="minigpt4_stage2_text")
    parser.add_argument("--text_len", type=int, default=32)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def evaluate_data(model, vis_processor, test_data, col_name, gpu_id):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model, vis_processor, text, ex, gpu_id)
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

def inference(model, vis_processor, text, ex, gpu_id):
    pred = {}

    metrics = mod_infer(model,vis_processor,text,gpu_id)
    metrics_lower = mod_infer(model,vis_processor,text.lower(),gpu_id)

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


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def mod_infer(model, vis_processor, user_message, gpu_id):

    chat = Interact(model, vis_processor, device='cuda:{}'.format(gpu_id))

    logging.info('=======Chat established, load images=======')


    img = Image.new('RGB', (1024, 1024), color = 'black')
    img_list = []

    chat_state = CONV_VISION.copy()

    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)

    chat.ask("", chat_state)

    chat_state.append_message(chat_state.roles[1], None)

    chat_state.append_message(user_message, None)
    
    outputs, input_ids, seg_tokens = chat.get_output_by_emb(conv=chat_state,
                                img_list=img_list
                            )
    
    logits = outputs.logits
    
    descp_encoding = chat.model.llama_tokenizer(user_message, return_tensors="pt", add_special_tokens = False).to(chat.device).input_ids

    goal_slice_dict = {
        'img' : slice(seg_tokens[0].shape[1],-seg_tokens[1].shape[1]),
        'inst_desp' : slice(-seg_tokens[-1].shape[1],None),
        'inst' : slice(-seg_tokens[-1].shape[1],-descp_encoding.shape[1]),
        'desp' : slice(-descp_encoding.shape[1],None)
        } 
    
    goal_slice = goal_slice_dict['inst_desp']

    input_ids = input_ids[0][goal_slice]
    
    logits_slice = logits[0,goal_slice,:]

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
    
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    text_len = args.text_len
    output_dir = f"{args.output_dir}/length_{text_len}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("JaineLi/VL-MIA-text", args.dataset, split=f"length_{text_len}")
    data = convert_huggingface_data_to_list_dic(dataset)
    # data = data[:10]

    logging.info('=======Initialization Finished=======')

    all_output = evaluate_data(model, vis_processor, data, 'input', args.gpu_id)

    fig_fpr_tpr(all_output, output_dir)



