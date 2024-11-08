# %% [markdown]
# 统计合成数据的label types

# %%
# import json_repair
# from tqdm import tqdm

# def collect_label(files_list=[]):
#     labels_dict = {}
#     for file in files_list:
#         with open(file, encoding='utf-8', mode='r') as f:
#             ori_list = f.readlines()
#         try:
#             for idx, item in enumerate(tqdm(ori_list)):
#                 item = item.split('\t')
#                 json_item = json_repair.loads(item[1])
#                 for item in json_item:
#                     if 'type' in item:
#                         if item['type'] not in labels_dict:
#                             labels_dict[item['type']] = 1
#                     if 'pos' in item:
#                         if item['pos'] not in labels_dict:
#                             labels_dict[item['pos']] = 1
#         except:
#             print(item)
#             print(len(item[0]))
#             json_repair.loads(item[1])
#     return labels_dict

# labels_dict = collect_label(files_list=['/home/lpc/repos/CNNNER/datasets/few_shot/cmeee_DA/entity_train_1000.json', '/home/lpc/repos/CNNNER/datasets/few_shot/cmeee_DA/entity_train_1000.json'])

# %% [markdown]
# 读取entity和pos的label_dict并进行校正处理,获得label_format函数

# %%
import json_repair
from tqdm import tqdm

label_dict = {
    'OTHER': ['OTHER', 'other']
}

count_dict = {}
except_dict = {}


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
        if 'POS' not in count_dict:
            count_dict['POS'] = 1
        else:
            count_dict['POS'] += 1
        return 'POS'
    if key in label_format_dict:
        format_key = label_format_dict[key]
        if format_key not in count_dict:
            count_dict[format_key] = 1
        else:
            count_dict[format_key] += 1
        return format_key
    count_dict['OTHER'] += 1
    if key not in except_dict:
        except_dict[key] = 1
    else:
        except_dict[key] += 1
    return 'WORD'

# %%
label_format('OTHER')

# %%
import os
from copy import deepcopy
LABEL_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/cmeee/labels.txt'
ORI_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/cmeee/train_1000.jsonl'
EXT_ENTITY_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/cmeee_DA/1000/entity_train_1000.jsonl'
EXT_POS_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/cmeee_DA/1000/pos_train_1000.jsonl'
DISABLED_ORI_LABELS = False
SAVE_FILE = os.path.join(os.path.dirname(ORI_FILE), os.path.splitext(os.path.basename(ORI_FILE))[0] + '_fusion{}.jsonl'.format('_mask' if DISABLED_ORI_LABELS else ''))
IGNORE_LABELS = []
# IGNORE_LABELS = ['ADVERB', 'NOUN', 'PROPER_NOUN', 'ADJECTIVE', 'QUANTIFIER']

# %%
with open(LABEL_FILE) as f:
    ori_labels = f.readlines()
ori_labels = [i.strip() for i in ori_labels]
with open(ORI_FILE) as f:
    ori_data = f.readlines()
ori_data = [json_repair.loads(i) for i in ori_data]
ori_data_copy = deepcopy(ori_data)
if DISABLED_ORI_LABELS:
    for item in ori_data:
        item['entities'] = []
        item['mask_ori'] = True

dataset_fusion_labels = []

# %%
with open(EXT_ENTITY_FILE, encoding='utf-8', mode='r') as f:
    ori_list = f.readlines()
for idx, item in enumerate(tqdm(ori_list)):
    item = item.split('\t')
    json_item = json_repair.loads(item[1])
    exists_2d = {}
    if not isinstance(json_item, list):
        continue
    for item in json_item:
        if 'entity' not in item or 'type' not in item: continue
        entity, entity_type = str(item['entity']), item['type']
        ent_len = len(entity)
        if ent_len <= 1:
            continue
        entity_type = label_format(entity_type)
        if entity_type in IGNORE_LABELS:
            continue
        if entity_type not in dataset_fusion_labels:
            dataset_fusion_labels.append(entity_type)
        ori_text_list = ori_data[idx]['text']
        for i in range(len(ori_text_list) - ent_len + 1):
            if ''.join(ori_text_list[i:i+ent_len]) == entity:
                if i not in exists_2d:
                    exists_2d[i] = {}
                    if (i + ent_len) not in exists_2d[i]:
                        ori_data[idx]['entities'].append({'start': i, 'end': i+ent_len, 'entity': entity_type, 'text': ori_text_list[i:i+ent_len]})
                        exists_2d[i][i+ent_len] = 1

# %%
with open(EXT_POS_FILE, encoding='utf-8', mode='r') as f:
    ori_list = f.readlines()
for idx, item in enumerate(tqdm(ori_list)):
    item = item.split('\t')
    json_item = json_repair.loads(item[1])
    exists_2d = {}
    if not isinstance(json_item, list):
        continue
    for item in json_item:
        if 'word' not in item or 'pos' not in item: continue
        entity, entity_type = str(item['word']), item['pos']
        ent_len = len(entity)
        if ent_len <= 1:
            continue
        entity_type = label_format(entity_type)
        if entity_type in IGNORE_LABELS:
            continue
        if entity_type not in dataset_fusion_labels:
            dataset_fusion_labels.append(entity_type)
        ori_text_list = ori_data[idx]['text']
        for i in range(len(ori_text_list) - ent_len + 1):
            if ''.join(ori_text_list[i:i+ent_len]) == entity:
                if i not in exists_2d:
                    exists_2d[i] = {}
                    if (i + ent_len) not in exists_2d[i]:
                        ori_data[idx]['entities'].append({'start': i, 'end': i+ent_len, 'entity': entity_type, 'text': ori_text_list[i:i+ent_len]})
                        # ori_data[idx]['entities'].append({'start': i, 'end': i+ent_len, 'entity': 'WORD', 'text': ori_text_list[i:i+ent_len]})
                        exists_2d[i][i+ent_len] = 1

# %%
import json
with open(SAVE_FILE, 'w') as f:
    for item in tqdm(ori_data):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
dataset_fusion_labels = sorted(dataset_fusion_labels)
idx = 0
final_labels = {}
for label in ori_labels:
    final_labels[label] = {
        'idx': idx,
        'count': -1,
        'is_target': True
    }
    idx += 1
for label in dataset_fusion_labels:
    if label in count_dict:
        count = count_dict[label]
    else:
        count = 9999
    final_labels[label] = {
        'idx': idx,
        'count': count,
        'is_target': False
    }
    idx += 1
with open('/home/lpc/repos/CNNNER/datasets/few_shot/cmeee_DA/1000/labels_fusion.json', 'w') as f:
    json.dump(final_labels, f, ensure_ascii=False)

# %%
except_dict

# %%
count_dict

# %%
