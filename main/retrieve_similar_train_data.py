import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from utils_a import *

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")

ranking_method = ZLIB
generation_method = RANDOM_TOP_K
file_path = f"{GENERATION_PATH}/{generation_method}/{ranking_method}.txt"
# clients = (1, 1)

generated_data = read_perplexity_ranking(file_path)
# retrieve_top_similar_sentences(generated_data[7],train_data,tokenizer)
# train_data_encoding= encode(tokenizer, train_data)
# print("encoding end")
# write_pickle(train_data_encoding,TRAIN_DATA_ENCODINGS_FILE)
train_data_encodings = read_pickle(TRAIN_DATA_ENCODINGS_FILE)
results = []
count=0
for x in generated_data:
    count+=1
    if count%10==0:
        print(count)
    result = retrieve_top_similar_sentences(x, train_data_encodings, tokenizer)
    results.append(result)
# write results to file

file_path = f"{SIMILAR_SENTENCES_PATH}/RANDOM_TOP_K/{ranking_method}.txt"
file = open(file_path, 'w')

for count, result in enumerate(results):
    sample = result[0]
    print("shoudl be 6:", len(result))

    file.write(f"Sample [ {count} ]: {sample}\n")
    file.write(f"\tTop 5 similar sentences in training set: \n")
    for i in range(1,6):
        score, train_sentence, common_tokens=result[i]
        file.write(f"\t--{i}-- {train_sentence}")
        file.write(f"\tscore:{score}  common tokens:  {common_tokens}\n")

    file.write("\n")
file.close()
# generated_encodings = encode(tokenizer, generated_data)
# email_encodings = encode(tokenizer, train_data)

# # scores = calculate_scores(generated_encodings, email_encodings,tokenizer)
# # top_score_dict_path = f"{SCORE_DICTIONARIES_PATH}clients_{start}_{end}_{ppl_ranking_method}.pkl"
# # write_top_scores_dict(generated_encodings,email_encodings,scores, top_score_dict_path, 1000)
# score_statistics_path= f"{SCORE_STATISTICS_PATH}clients_{start}_{end}_{ppl_ranking_method}.pkl"
# score_statistics(scores,score_statistics_path)
