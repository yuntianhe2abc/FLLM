import re

import numpy as np
import time

from fuzzywuzzy import fuzz
from transformers import GPT2TokenizerFast

file = open(f"../data/train_sentences/all_sentences.txt", 'r', encoding="utf-8")
lines=file.readlines()
count=0
for x in lines:
    if "http://football" in x:
        count+=1
print(count)
count1=0
file1 = open(f"../data/sample_train_set/200000_generation_origin.txt", 'r', encoding="utf-8")
lines1=file1.readlines()

def delete_common_string(x):
    if re.search('http://football', x) is None:
        print(f"should not be None : {x}")
    start = re.search('http://football', x).span()[0]

    end = re.search('league', x).span()[1]
    a = re.search('/FONT/FONTBRBRA HREF', x)
    if a is not None:
        start = a.span()[0]
    new_x = x[:start] + x[end:]
    return new_x

for i,x in enumerate(lines1):
    if "http://football" in x and "fantasy.sportsline.com" in x and "scriptplayers" in x and "elink" in x and "league" in x:
        lines1[i]=delete_common_string(x)

new_200000_file = open(f"../data/sample_train_set/200000_generation.txt", 'w', encoding="utf-8")
for line in lines1:
    new_200000_file.write(line)
new_200000_file.close()
    # print(x)
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
#                                               pad_token="<|pad|>")
# sentence="Jimi Hendrix Experience 0909 0909 0909 "
# a=tokenizer.encode(sentence)
