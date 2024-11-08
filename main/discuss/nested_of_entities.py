# %%
import json
import json_repair
from tqdm import tqdm

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

ORI_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/youku/train_1000.jsonl'
EXT_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/youku_DA/1000/pos_train_1000.jsonl'

with open(ORI_FILE) as f:
    ori_data = f.readlines()
ori_data = [json.loads(line) for line in ori_data]

ori_entities = []

for item in tqdm(ori_data):
    entities = item['entities']
    ent_list = []
    for entity in entities:
        entity_text = ''.join(entity['text'])
        ent_list.append(entity_text)
    ori_entities.append(ent_list)

# %%
ext_entities = {}

with open(EXT_FILE) as f:
    ext_data = f.readlines()
ext_data = [json_repair.loads(line.split('\t')[1]) for line in ext_data]

for ent_list, item in zip(ori_entities, ext_data):
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
        exists = False
        for ent in ent_list:
            if entity_text in ent and entity_text != ent:
                exists = True
                break
        entity_type = label_format(entity_type)
        if exists:
            if entity_type not in ext_entities:
                ext_entities[entity_type] = 1
            else:
                ext_entities[entity_type] += 1

# %%
for key, value in ext_entities.items():
    print('{}: {}'.format(key, value))

# %%
