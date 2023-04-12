import pickle
with open("../result/score_dictionaries/clients_1_1_PPL-XL_Zlib.pkl", 'rb') as fp:
    top_scores=pickle.load(fp)
    print(top_scores)
fp.close()

# a={}
# b=a.get(2,[])
# print(b)
# a[2].extend([2,43,56])
# print(b)