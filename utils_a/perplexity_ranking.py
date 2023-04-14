import torch
import random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import zlib


def random_top_k(N, batch_size, device, tokenizer):
    print(f"using device: {device}")
    seq_len = 100
    top_k = 40
    samples = []
    scores = {"S": [], "M": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(N / batch_size))
    with tqdm(total=N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            prompt = "If you have any further questions, please give me a call at"
            prompts = ["<|startoftext|>"] * batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)

            # batch generation
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                min_length=input_len + seq_len,
                do_sample=True,
                top_k=top_k,
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:
                # perplexity of GPT2-ENRON and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer)
                p2 = calculatePerplexity(text, model2, tokenizer)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), model1, tokenizer)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["S"].append(p2)
                scores["M"].append(p1)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)

            pbar.update(batch_size)
    scores["S"] = np.asarray(scores["S"])
    scores["M"] = np.asarray(scores["M"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])
    # Sort by perplexity
    metric = -np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"])

    metric = np.log(scores["M"]) / np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"], "PPL-M", scores["M"])

    metric = np.log(scores["Lower"]) / np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"], "PPL-S-Lower", scores["Lower"])

    metric = scores["zlib"] / np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"], "Zlib", scores["zlib"])


def write_topN(metric, samples, name1, scores1, name2=None, scores2=None, n=100):
    idxs = np.argsort(metric)[::-1][:n]
    print(scores1.shape)
    file = open(f"/content/drive/MyDrive/FLLM/data/sample_train_set/{name1}_{name2}.txt", 'w', encoding="utf-8")

    # file=open(f"{name1}_{name2}_test.txt",'w',encoding="utf-8")
    for i, idx in enumerate(idxs):
        score = ""
        if scores2 is not None:
            score = f"{i + 1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}\n"
        else:
            score = f"{i + 1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}\n"
        file.write(score)
        file.write(f"{samples[idx]}\n")
        file.write("\n")

def main_sample_train_content(N,batch_size,device,tokenizer,sampling_pool):
    print(f"using device: {device}")
    seq_len = 100
    top_k = 40
    samples = []
    scores = {"S": [], "M": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(N / batch_size))
    with tqdm(total=N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            prompts=[]
            input_len = 10
            input_ids = []
            attention_mask = []

            while len(input_ids) < batch_size:
                # take some random words in common crawl
                r = np.random.randint(0, len(sampling_pool))
                prompt = " ".join(sampling_pool[r:r+100].split(" ")[1:-1])

                # make sure we get the same number of tokens for each prompt to enable batching
                inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                if len(inputs['input_ids'][0]) == input_len:
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])

            inputs = {'input_ids': torch.stack(input_ids),
                      'attention_mask': torch.stack(attention_mask)}

            # the actual truncated prompts
            prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            print(prompts)
            # batch generation
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                min_length=input_len+seq_len,
                do_sample=True,
                top_k=top_k,
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:
                # perplexity of GPT2-ENRON and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer)
                p2 = calculatePerplexity(text, model2, tokenizer)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), model1, tokenizer)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["S"].append(p2)
                scores["M"].append(p1)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)

            pbar.update(batch_size)
    scores["S"] = np.asarray(scores["S"])
    scores["M"] = np.asarray(scores["M"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])
    # Sort by perplexity
    print(1)
    metric = -np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"])
    print(2)
    metric = np.log(scores["M"]) / np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"], "PPL-M", scores["M"])
    print(3)
    metric = np.log(scores["Lower"]) / np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"], "PPL-S-Lower", scores["Lower"])
    print(4)
    metric = scores["zlib"] / np.log(scores["S"])
    write_topN(metric, samples, "PPL-S", scores["S"], "Zlib", scores["zlib"])

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    return torch.exp(loss).item()
