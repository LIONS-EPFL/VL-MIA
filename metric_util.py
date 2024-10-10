import numpy as np
import zlib
from scipy.stats import entropy
import statistics
import json
import pdb
import torch
from PIL import Image

def get_text_metric(ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05, renyi_2, text, ppl_lower, mod_renyi_05, mod_renyi_2):
    pred = {}

    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    
    pred["ppl"] = ppl
    pred["ppl/zlib"] = np.log(ppl)/zlib_entropy
    pred["ppl/lowercase_ppl"] = - (np.log(ppl_lower) / np.log(ppl)).item()

    # mink
    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_length = int(len(all_prob)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -1* np.mean(topk_prob).item()

    pred["Modified_entropy"] = np.nanmean(mod_entropy).item()

    pred["Modified_renyi_05"] = np.nanmean(mod_renyi_05).item()

    pred["Modified_renyi_2"] = np.nanmean(mod_renyi_2).item()

    pred["Max_Prob_Gap"] = -np.mean(gap_p).item()

    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_05"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_1"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_2"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[-k_length:]
        pred[f"Max_{ratio*100}% renyi_inf"] = np.mean(topk_prob).item()


    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05)[:k_length]
        pred[f"Min_{ratio*100}% renyi_05"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[:k_length]
        pred[f"Min_{ratio*100}% renyi_1"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2)[:k_length]
        pred[f"Min_{ratio*100}% renyi_2"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[:k_length]
        pred[f"Min_{ratio*100}% renyi_inf"] = np.mean(topk_prob).item()

    return pred

def kl_divergence(p, log_p, log_q):
    kl_div = np.sum(p * (log_p - log_q))
    return kl_div

def get_img_metric(ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05, renyi_2, log_probs, aug1_prob, aug2_prob, aug3_prob, aug4_prob, mod_renyi_05, mod_renyi_2):
    pred = {}

    kl_1 = kl_divergence(org_prob.cpu().numpy(), log_probs.cpu().numpy(), aug1_prob.cpu().numpy()).mean()
    kl_2 = kl_divergence(org_prob.cpu().numpy(), log_probs.cpu().numpy(), aug2_prob.cpu().numpy()).mean()
    kl_3 = kl_divergence(org_prob.cpu().numpy(), log_probs.cpu().numpy(), aug3_prob.cpu().numpy()).mean()
    kl_4 = kl_divergence(org_prob.cpu().numpy(), log_probs.cpu().numpy(), aug4_prob.cpu().numpy()).mean()
    
    pred['aug_kl'] = -statistics.mean([kl_1,kl_2,kl_3,kl_4])

    pred["ppl"] = ppl

    # mink
    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_length = int(len(all_prob)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -1* np.mean(topk_prob).item()

    pred["Modified_entropy"] = np.nanmean(mod_entropy).item()

    pred["Modified_renyi_05"] = np.nanmean(mod_renyi_05).item()

    pred["Modified_renyi_2"] = np.nanmean(mod_renyi_2).item()

    pred["Max_Prob_Gap"] = -np.mean(gap_p).item()

    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_05"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_1"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_2"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[-k_length:]
        pred[f"Max_{ratio*100}% renyi_inf"] = np.mean(topk_prob).item()

    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05)[:k_length]
        pred[f"Min_{ratio*100}% renyi_05"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[:k_length]
        pred[f"Min_{ratio*100}% renyi_1"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2)[:k_length]
        pred[f"Min_{ratio*100}% renyi_2"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[:k_length]
        pred[f"Min_{ratio*100}% renyi_inf"] = np.mean(topk_prob).item()

    return pred

def convert(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64, float)):
        return float(obj)
    elif isinstance(obj, (np.int16, np.int32, np.int64, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(item) for item in obj)
    elif isinstance(obj, set):
        return list(convert(item) for item in obj)
    else:
        return obj

def save_output(data, filename):
    converted_data = convert(data)
    with open(filename, 'w') as f:
        json.dump(converted_data, f, indent=4) 


def get_meta_metrics(input_ids, probabilities, log_probabilities):
    entropies = []
    all_prob = []
    modified_entropies = []
    max_prob = []
    gap_prob = []
    renyi_05 = []
    renyi_2 = []
    losses = []
    modified_entropies_alpha05 = []
    modified_entropies_alpha2 = []
    epsilon = 1e-10

    input_ids_processed = input_ids[1:]  # Exclude the first token for processing
    for i, token_id in enumerate(input_ids_processed):
        token_probs = probabilities[i, :]  # Get the probability distribution for the i-th token
        token_probs = token_probs.clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :]  # Log probabilities for entropy
        token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)

        entropy = -(token_probs * token_log_probs).sum().item()  # Calculate entropy
        entropies.append(entropy)

        token_probs_safe = torch.clamp(token_probs, min=epsilon, max=1-epsilon)

        alpha = 0.5
        renyi_05_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_05.append(renyi_05_)
        alpha = 2
        renyi_2_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_2.append(renyi_2_)

        max_p = token_log_probs.max().item()
        second_p = token_log_probs[token_log_probs != token_log_probs.max()].max().item()
        gap_p = max_p - second_p
        gap_prob.append(gap_p)
        max_prob.append(max_p)

        mink_p = token_log_probs[token_id].item()
        all_prob.append(mink_p)

        cross_entropy_loss = -mink_p
        losses.append(cross_entropy_loss)

        # Modified entropy
        p_y = token_probs_safe[token_id].item()
        modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_safe)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
        modified_entropies.append(modified_entropy)

        token_probs_remaining = torch.cat((token_probs_safe[:token_id], token_probs_safe[token_id+1:]))
        
        for alpha in [0.5,2]:
            entropy = - (1 / abs(1 - alpha)) * (
                (1-p_y)* p_y**(abs(1-alpha))\
                    - (1-p_y)
                    + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                    - torch.sum(token_probs_remaining)
                    ).item() 
            if alpha==0.5:
                modified_entropies_alpha05.append(entropy)
            if alpha==2:
                modified_entropies_alpha2.append(entropy)

    loss = np.nanmean(losses)
    # loss = torch.tensor(loss)

    return {
        "ppl": np.exp(loss),
        "all_prob": all_prob,
        "loss": loss,
        "entropies": entropies,
        "modified_entropies": modified_entropies,
        "max_prob": max_prob,
        "probabilities": probabilities,
        "log_probs" : log_probabilities,
        "gap_prob": gap_prob,
        "renyi_05": renyi_05,
        "renyi_2": renyi_2,
        "mod_renyi_05" : modified_entropies_alpha05,
        "mod_renyi_2" : modified_entropies_alpha2
    }

