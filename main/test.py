import pickle
from utils_a import *
from transformers import  GPT2TokenizerFast
def print_memorization_ground_truth(generated_data,email_data,file_path):
    result=[]
    generated_encodings = encode(tokenizer, generated_data)
    email_encodings = encode(tokenizer, email_data)
    fp=open(file_path, 'rb')
    top_scores = pickle.load(fp)
    print(len(top_scores.keys()))
    fp.close()
    for i, key in enumerate(top_scores):
        j,k=key
        if top_scores[key]>5:

            result.append((generated_data[j],email_data[k],top_scores[key]))
            print(generated_data[j])
            print(email_data[k])
            _, common_list = get_intersection(get_k_token_set(generated_encodings[j]),
                                              get_k_token_set(email_encodings[k]))
            texts = tokenizer.batch_decode(common_list, skip_special_tokens=True)

            print(texts)

    return result

# clients=(1,1)
file_path = f"{PERPLEXITY_RANKING_PATH}{ranking_method}.txt"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
email_data = load_clients_train_data()
generated_data = read_perplexity_ranking(file_path)

file_path=f"{SCORE_DICTIONARIES_PATH}clients_{clients[0]}_{clients[1]}_{ranking_method}.pkl"

result=print_memorization_ground_truth(generated_data,email_data,file_path)

