import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from utils_a import *

PERPLEXITY_RANKING_PATH = "../data/perplexity_ranking/"
TRAIN_DATA_FOLDER_PATH = "../data/train_sentences/"
SCORE_DICTIONARIES_PATH = "../result/score_dictionaries/"
SCORE_STATISTICS_PATH="../result/score_statistics/"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
ppl_ranking_method = "PPL-XL_Zlib"
file_path = f"{PERPLEXITY_RANKING_PATH}{ppl_ranking_method}.txt"
clients = (1, 1)
email_data = load_clients_train_data(clients, TRAIN_DATA_FOLDER_PATH)
generated_data = read_perplexity_ranking(file_path)

generated_encodings = encode(tokenizer, generated_data)
email_encodings = encode(tokenizer, email_data)
print("finish encoding")

start, end = clients
print("start score")
scores = calculate_scores(generated_encodings, email_encodings,tokenizer)
top_score_dict_path = f"{SCORE_DICTIONARIES_PATH}clients_{start}_{end}_{ppl_ranking_method}.pkl"
write_top_scores_dict(generated_encodings,email_encodings,scores, top_score_dict_path, 1000)
score_statistics_path= f"{SCORE_STATISTICS_PATH}clients_{start}_{end}_{ppl_ranking_method}.pkl"
score_statistics(scores,score_statistics_path)