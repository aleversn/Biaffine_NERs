# %%
import json

def is_entity(filename, entity):
    with open(filename) as f:
        data = [json.loads(line) for line in f]
    en_list = [token for token in entity]
    length = len(en_list)
    exists = False
    for item in data:
        text, label = item['text'], item['label']
        for idx in range(len(text)):
            if en_list == text[idx: idx + length]:
                exists = True
                if label[idx] != 'O':
                    return (text, idx, idx + length, label[idx])
    return exists
            

is_entity('/home/lpc/repos/CNNNER/datasets/weibo/train.jsonl', '吃货')

# %%
