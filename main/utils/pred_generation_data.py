# %%
import sys
sys.path.append("../../")
import json
from transformers import BertTokenizer, BertConfig
from main.predictor.fusion_ner_predictor import FusionNERPredictor

LABEL_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/weibo_DA/labels_fusion.json'
SOURCE_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/weibo_DA/1000/train_1000_synthetic.jsonl'
SAVE_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/weibo_DA/1000/train_1000_synthetic.jsonl1'

tokenizer = BertTokenizer.from_pretrained(
    "/home/lpc/models/chinese-bert-wwm-ext/")
config = BertConfig.from_pretrained(
    "/home/lpc/models/chinese-bert-wwm-ext/")
pred = FusionNERPredictor(tokenizer=tokenizer, config=config, from_pretrained='/home/lpc/repos/CNNNER/save_model/CNNNER-weibo_1000_fusion/cnnner_best',
                          label_file=LABEL_FILE, batch_size=4)

# %%
with open(SOURCE_FILE) as f:
    ori_data = f.readlines()
data = [json.loads(i) for i in ori_data]
data_text = [''.join(i['text']) for i in data]
entities_list = []

for entities in pred(data_text):
    entities_list.extend(entities)

# %%
for item, ext_entities in zip(data, entities_list):
    for entity in ext_entities:
        if entity not in item['entities']:
            item['entities'].append(entity)

with open(SAVE_FILE, 'w') as f:
    for i in data:
        f.write(json.dumps(i, ensure_ascii=False) + '\n')

# %%
