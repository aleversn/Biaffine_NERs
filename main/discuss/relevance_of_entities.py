# %%
import json
import json_repair
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from prettytable import PrettyTable

def get_embeddings(model, batch, pooler='cls', gpu=[0]):
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpu).cuda()
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    if pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return pooler_output.cpu()
    elif pooler == 'cls_before_pooler':
        return last_hidden[:, 0].cpu()
    elif pooler == "avg":
        return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(
            1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 *
                         batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    else:
        raise NotImplementedError

def batcher(model, tokenizer, input_list, batch_size=128, gpu=[0]):

    num_batch = len(input_list) // batch_size + 1 if len(input_list) % batch_size != 0 else len(input_list) // batch_size

    embeddings_list = []
    for i in tqdm(range(num_batch)):
        batch = input_list[i*batch_size:(i+1)*batch_size]
        batch = tokenizer(batch, padding=True,
                          truncation=True, return_tensors='pt')
        embeddings = get_embeddings(model, batch, gpu=gpu)
        embeddings_list += embeddings.tolist()
    return embeddings_list

# 计算list1和list2的余弦距离
def get_distance(list1, list2):
    list1 = torch.tensor(list1).cuda()
    list2 = torch.tensor(list2).cuda()
    list1_norm = list1 / list1.norm(dim=1, keepdim=True)
    list2_norm = list2 / list2.norm(dim=1, keepdim=True)

    cosine_distances = 1 - torch.mm(list1_norm, list2_norm.T)
    return cosine_distances

label_dict = {
    'OTHER': ['OTHER', 'other']
}
def update_label_dict(file_name):
    with open(file_name) as f:
        e_dict = json_repair.load(f)
    for key in e_dict:
        cur = e_dict[key]
        if key not in cur:
            cur.append(key)
        if key not in label_dict:
            label_dict[key] = cur
        else:
            label_dict[key] += cur

update_label_dict('/home/lpc/repos/CNNNER/datasets/fusion_knowledge/entity_label.json')
update_label_dict('/home/lpc/repos/CNNNER/datasets/fusion_knowledge/pos_label.json')

for key in label_dict:
    cur = label_dict[key]
    new_cur = []
    for item in cur:
        if item not in new_cur:
            new_cur.append(item)
    label_dict[key] = new_cur

label_format_dict = {}

for key in label_dict:
    for key_item in label_dict[key]:
        label_format_dict[key_item] = key

def label_format(key, distinct_pos=False):
    key = key.strip()
    if distinct_pos and key in ['OTHER']:
        return 'POS'
    if key in label_format_dict:
        format_key = label_format_dict[key]
        return format_key
    return 'WORD'

from_pretrained_path = '/home/lpc/models/simcse-chinese-roberta-wwm-ext/'
tokenizer = AutoTokenizer.from_pretrained(from_pretrained_path)
model = AutoModel.from_pretrained(from_pretrained_path)

ORI_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/resume/train_1000.jsonl'
EXT_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/resume_DA/1000/entity_train_1000.jsonl'

with open(ORI_FILE) as f:
    ori_data = f.readlines()
ori_data = [json.loads(line) for line in ori_data]

ori_entities = {}

for item in tqdm(ori_data):
    entities = item['entities']
    for entity in entities:
        entity_type = entity['entity']
        entity_text = ''.join(entity['text'])
        if entity_type not in ori_entities:
            ori_entities[entity_type] = [entity_text]
        else:
            if entity_text not in ori_entities[entity_type]:
                ori_entities[entity_type].append(entity_text)

emb_entities = {}

for key in tqdm(ori_entities, total=len(ori_entities)):
    texts = ori_entities[key]
    embeddings = batcher(model, tokenizer, texts, batch_size=256, gpu=[0, 1])
    emb_entities[key] = embeddings

# %%
ext_entities = {}
emb_ext_entities = {}

with open(EXT_FILE) as f:
    ext_data = f.readlines()
ext_data = [json_repair.loads(line.split('\t')[1]) for line in ext_data]

for item in ext_data:
    entities = item
    for entity in entities:
        if 'word' in entity:
            if 'pos' not in entity:
                continue
            entity_type = entity['pos']
            entity_text = str(entity['word'])
        else:
            if 'type' not in entity or 'entity' not in entity:
                continue
            entity_type = entity['type']
            entity_text = str(entity['entity'])
        if len(entity_text) <= 1:
            continue
        entity_type = label_format(entity_type)
        if entity_type not in ext_entities:
            ext_entities[entity_type] = [entity_text]
        else:
            if entity_text not in ext_entities[entity_type]:
                ext_entities[entity_type].append(entity_text)

for key in tqdm(ext_entities, total=len(ext_entities)):
    texts = ext_entities[key]
    embeddings = batcher(model, tokenizer, texts, batch_size=256, gpu=[0, 1])
    emb_ext_entities[key] = embeddings

# %%
result = []

for ext_key in tqdm(emb_ext_entities, total=len(emb_ext_entities)):
    cols = []
    for ori_key in emb_entities:
        scores = get_distance(emb_entities[ori_key], emb_ext_entities[ext_key])
        scores = scores.mean().tolist()
        cols.append(scores)
    result.append(cols)

table = PrettyTable()
table.field_names = ['Ext/Ori'] + [key for key in ori_entities]
for idx, key in enumerate(tqdm(emb_ext_entities, total=len(emb_ext_entities))):
    table.add_row([key] + [str(round(score * 100, 2)) for score in result[idx]])
print(table)

# cos_socres = get_distance(emb_sts, emb_wiki)

# %%
