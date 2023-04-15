# -*- coding: utf-8 -*-
"""Rank.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nG8MzjMUPVBqpkODZsseOF0aWaXqC3MM
"""

SAMPLE_TRAIN_SET = "sample_train_set"
sampling_method = SAMPLE_TRAIN_SET

import torch
import random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from pprint import pprint
from tqdm import tqdm
import math
import zlib

device = torch.device("mps")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
model1 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model1.resize_token_embeddings(len(tokenizer))
MODEL_PATH = "../model/server_2512.pt"
model1.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model2 = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
# model2.resize_token_embeddings(len(tokenizer))
model1.eval()
# model2.eval()


def write_topN(metric, samples, name1, scores1, name2=None, scores2=None, n=1000):
    idxs = np.argsort(metric)[::-1][:n]
    print(scores1.shape)
    file = open(f"../data/{sampling_method}/{name1}_{name2}.txt", 'w', encoding="utf-8")

    # file=open(f"{name1}_{name2}_test.txt",'w',encoding="utf-8")
    for i, idx in enumerate(idxs):
        score = ""
        if scores2 is not None:
            score = f"{i + 1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}\n"
        else:
            score = f"{i + 1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}\n"
        file.write(score)
        file.write(f"{samples[idx]}\n")
        file.write("")


def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    # print(input_ids)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss).item()


def calculateWindowPerplexity(text, model1, tokenizer, step=5, window_size=15):
    abc = tokenizer(text, return_tensors="pt", padding=False)
    ids = abc["input_ids"][0]
    input_ids = get_k_token_set(ids, k=window_size)
    # input_ids=torch.tensor(input_ids)
    losses = []
    with torch.no_grad():
        for i in np.arange(0, len(input_ids), step):
            input_ids_a = input_ids[i]

            outputs = model1(input_ids=input_ids_a.to(device), labels=input_ids_a.to(device))
            loss, logits = outputs[:2]
            losses.append(torch.exp(loss).item())
    # print(len(losses),losses)
    return min(losses)

def get_k_token_set(ids, k=3):
    input_ids = []
    n = len(ids)

    for i in range(n - k + 1):
        k_tokens = ids[i:i + k]
        a = k_tokens.clone().detach()
        b = torch.unsqueeze(a, 0)
        input_ids.append(b)
    return input_ids


sampling_method = SAMPLE_TRAIN_SET
generation_file = f"../data/{sampling_method}/200000_generation.txt"
scores = {"S": [], "M": [], "Lower": [], "Zlib": [], "Window": []}
# scores = {"S": [], "M": [], "Lower": [], "Zlib": []}
file = open(generation_file, 'r', encoding="utf-8")
samples = file.readlines()
file.close()
len(samples)
batch_size = 100
num_batches = int(np.ceil(len(samples) / batch_size))
with tqdm(total=len(samples)) as pbar:
    for i in range(num_batches):

        texts = samples[i * batch_size:(i + 1) * batch_size]
        for text in texts:
            # perplexity of GPT2-ENRON and GPT2-S
            # p1 = calculatePerplexity(text, model1, tokenizer)
            # p2 = calculatePerplexity(text, model2, tokenizer)
            # # perplexity on lower-case sample
            # p_lower = calculatePerplexity(text.lower(), model1, tokenizer)
            # # Zlib "entropy" of sample
            # zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
            p_window = calculateWindowPerplexity(text, model1, tokenizer)

            # scores["S"].append(p2)
            # scores["M"].append(p1)
            # scores["Lower"].append(p_lower)
            # scores["Zlib"].append(zlib_entropy)
            scores["Window"].append(p_window)
        pbar.update(batch_size)
# scores["S"] = np.asarray(scores["S"])
# scores["M"] = np.asarray(scores["M"])
# scores["Lower"] = np.asarray(scores["Lower"])
# scores["Zlib"] = np.asarray(scores["Zlib"])
scores["Window"] = np.asarray(scores["Window"])
# Sort by perplexity
# metric = -np.log(scores["S"])
# write_topN(metric, samples, "PPL-S", scores["S"])
#
# metric = np.log(scores["S"]) / np.log(scores["M"])
# write_topN(metric, samples, "PPL-S", scores["S"], "PPL-M", scores["M"])
#
# metric = np.log(scores["Lower"]) / np.log(scores["S"])
# write_topN(metric, samples, "PPL-S", scores["S"], "PPL-S-Lower", scores["Lower"])
#
# metric = scores["Zlib"] / np.log(scores["S"])
# write_topN(metric, samples, "PPL-S", scores["S"], "Zlib", scores["Zlib"])

metric = -np.log(scores["Window"])
write_topN(metric, samples, "PPL-Window", scores["Window"])
