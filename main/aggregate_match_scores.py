from utils_a import *
a="http://www.energycentral.com"
b=search_train_data(a,load_all_train_data(f"{TRAIN_DATA_FOLDER_PATH}all_sentences.txt"))
print(b)