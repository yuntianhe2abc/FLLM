from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import re
PERPLEXITY_RANKING_PATH = "../data/perplexity_ranking/"
TRAIN_DATA_FILE = "../data/train_sentences/all_sentences.txt"
SCORE_DICTIONARIES_PATH = "../result/score_dictionaries/"
SCORE_STATISTICS_PATH="../result/score_statistics/"
ranking_method="PPL-XL_Zlib"

def generate2(model, tokenizer, device, seq_len, batch_size, num_samples, prompt, top_k=50, temperature=0.8):
    samples = []
    num_batches = int(np.ceil(num_samples / batch_size))
    with tqdm(total=num_samples) as pbar:
        for i in range(num_batches):
            prompts = [prompt] * batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True,
                # top_p=1,
                top_k=top_k,
                temperature=temperature

            )
            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            for text in texts:
                samples.append(text)
            pbar.update(batch_size)
    return samples


def encode(tokenizer, sentences):
    encodings = []
    for a in sentences:
        encoding = tokenizer(a, return_tensors="pt", padding=False, truncation=True, max_length=128)
        encodings.append(encoding["input_ids"][0].tolist())
    return encodings


def get_k_token_set(sentence_encoding, k=3):
    k_tokens_list = []
    n = len(sentence_encoding)
    for i in range(n - k + 1):
        k_tokens = sentence_encoding[i:i + k]
        k_tokens_list.append(k_tokens)
    return k_tokens_list

#
# def load_clients_train_data(clients, folder_path):
#     clients_data = []
#     start, end = clients
#     for i in range(start, end + 1):
#         file_name = f"client_{i}.txt"
#         file_path = folder_path + file_name
#         file = open(file_path, 'r')
#         a = file.readlines()
#         clients_data.extend(a)
#     return clients_data

def number_sentence_preprocessor(Str):
  new_Str=Str
  numbers = re.findall(r'\d+', Str)
  if len(numbers)>0:
    for number in numbers:
      if len(number) >= 4:
        number_digits_with_space = ' '.join(number)
        number_digits_with_space=' '+number_digits_with_space+' '
        new_Str=new_Str.replace(number,number_digits_with_space)
  return new_Str

def read_perplexity_ranking(file_path):
    sentences = []
    file = open(file_path, 'r')
    result1 = file.readlines()
    for i in range(1000):
        sentences.append(result1[i * 3 + 2])
    return sentences


def get_intersection(k_set1, k_set2):
    """
    given k-tokens-set for two sentences, count how many k-tokens set do they have in common
    Args:
    Returns:
        Count of common and unique k-token-set
    """
    common_list = []
    count = 0
    for a in k_set1:
        for b in k_set2:
            if a == b:
                if a not in common_list:
                    count += 1
                    common_list.append(a)

    return count, common_list


# def calculate_scores(generated_encodings, email_encodings, tokenizer, k=3):
#     """
#     given k-tokens-set for two sentences, count how many k-tokens set do they have in common
#     Args:
#     Returns:
#         scores np array
#     """
#     m = len(generated_encodings)
#     n = len(email_encodings)
#     count = 0
#     scores = np.zeros((m, n), dtype=float)
#     for i in range(m):
#         count += 1
#         if count % 40 == 0:
#             print(count)
#         x = generated_encodings[i]
#         k_set1 = get_k_token_set(x)
#         for j in range(n):
#             y = email_encodings[j]
#             k_set2 = get_k_token_set(y)
#             score, _ = get_intersection(k_set1, k_set2)
#             scores[i][j] = score
#     return scores

def retrieve_top_similar_sentences(generated_sentence, train_data, tokenizer,  num_of_return=5,k=3):
    """
    Given a sentence, find most k-tokens matched sentences from training set
    Args:
    Returns:
        scores np array
    """
    common_lists=[]
    scores=[]
    number_split_generated_sentence=number_sentence_preprocessor(generated_sentence)
    generated_sentence_encoding = tokenizer(number_split_generated_sentence, return_tensors="pt", padding=False, truncation=True, max_length=128)
    train_data_encoding= encode(tokenizer, train_data)
    n = len(train_data)

    k_set1 = get_k_token_set(generated_sentence_encoding,k)
    for i in range(n):
        k_set2 = get_k_token_set(train_data_encoding[i],k)
        score, common_list = get_intersection(k_set1, k_set2)
        if score>0:
            scores.append(score)
            common_lists.append(tokenizer.batch_decode(common_list, skip_special_tokens=True))

    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k],reverse=True)

    top_indices=sorted_indices[:num_of_return]
    for i in top_indices:
        print((scores[i],train_data[i]))
        print(common_lists[i])
    return scores

def write_ground_truth(scores, generated_data, email_data, path, top_n=1000):
    scores_flatten = np.ravel(scores)
    sorted_indices = np.argsort(scores_flatten)
    reverse_sorted_indices = sorted_indices[::-1]
    largest_indices = reverse_sorted_indices[:top_n]
    with open(path, 'w') as f:
        for index in largest_indices:
            j, k = np.unravel_index(index, scores.shape)
            f.write(f"{scores[j][k]}\n")
            f.write(f"{generated_data[j]}\n")
            f.write(f"{email_data[j]}\n")
            f.write("\n")
    f.close()


def load_train_data():
    file = open(TRAIN_DATA_FILE, 'r')
    train_data = file.readlines()
    file.close()
    return train_data


def search_train_data(string, all_train_data):
    count = 0
    for x in all_train_data:
        if string in x:
            count += 1
    return count


def score_statistics(scores, store_path):
    score_count = {}
    scores_flatten = np.ravel(scores)
    sorted_indices = np.argsort(scores_flatten)
    reverse_sorted_indices = sorted_indices[::-1]
    for index in reverse_sorted_indices:
        j, k = np.unravel_index(index, scores.shape)
        score = scores[j][k]
        score_count[score] = score_count.get(score, 0) + 1
    with open(store_path, 'wb') as fp:
        pickle.dump(score_count, fp)
        print('Score statistics saved successfully!')
    return score_count


def write_top_scores_dict(generated_encodings, email_encodings, scores, store_path, top_n=2000):
    top_scores = {}
    common_k_set = {}
    scores_flatten = np.ravel(scores)
    sorted_indices = np.argsort(scores_flatten)
    reverse_sorted_indices = sorted_indices[::-1]
    count = 0
    for index in reverse_sorted_indices:
        if count < top_n and scores_flatten[index] > 0:

            j, k = np.unravel_index(index, scores.shape)
            _, common_list = get_intersection(get_k_token_set(generated_encodings[j]),
                                              get_k_token_set(email_encodings[k]))
            existed_match = common_k_set.get(j, [])

            common_tri = intersection(common_list, existed_match)
            if len(common_tri) < (len(common_list) / 2):
                res = [i for i in common_list if i not in existed_match]
                existed_match.extend(res)
                common_k_set[j] = existed_match
                key = (j, k)
                top_scores[key] = scores[j][k]

    with open(store_path, 'wb') as fp:
        pickle.dump(top_scores, fp)
        print('dictionary saved successfully to file')


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
