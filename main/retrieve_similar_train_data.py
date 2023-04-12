import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from utils_a import *

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
ppl_ranking_method = "PPL-XL_Zlib"
file_path = f"{PERPLEXITY_RANKING_PATH}{ppl_ranking_method}.txt"
# clients = (1, 1)

generated_data = read_perplexity_ranking(file_path)
# retrieve_top_similar_sentences(generated_data[7],train_data,tokenizer)
# train_data_encoding= encode(tokenizer, train_data)
# print("encoding end")
# write_pickle(train_data_encoding,TRAIN_DATA_ENCODINGS_FILE)
train_data_encodings = read_pickle(TRAIN_DATA_ENCODINGS_FILE)
results = []
for x in generated_data:
    result = retrieve_top_similar_sentences(x, train_data_encodings, tokenizer)
    results.append(result)
# write results to file
ranking_method = ZLIB
generation_method = RANDOM_TOP_K
file_path = f"{SIMILAR_SENTENCES_PATH}/RANDOM_TOP_K/{ranking_method}.txt"
file = open(file_path, 'w')

for count, values in enumerate(results):
    sample = values[0]
    top_5 = values[1]
    file.write(f"{count}\n")
    file.write(f"Sample: {sample}\n")

    for count_1, top_sentence in enumerate(top_5):
        score, train_sentence, common_tokens = result[1]
        file.write(f"{count_1}. score:{score}  common tokens:  {common_tokens}\n")
        file.write(f"{train_sentence}\n")
        file.write("\n")

file.close()
# generated_encodings = encode(tokenizer, generated_data)
# email_encodings = encode(tokenizer, train_data)

# # scores = calculate_scores(generated_encodings, email_encodings,tokenizer)
# # top_score_dict_path = f"{SCORE_DICTIONARIES_PATH}clients_{start}_{end}_{ppl_ranking_method}.pkl"
# # write_top_scores_dict(generated_encodings,email_encodings,scores, top_score_dict_path, 1000)
# score_statistics_path= f"{SCORE_STATISTICS_PATH}clients_{start}_{end}_{ppl_ranking_method}.pkl"
# score_statistics(scores,score_statistics_path)
