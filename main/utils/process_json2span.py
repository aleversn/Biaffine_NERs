# %%
import json
from copy import deepcopy

ORI_JSON = "/home/lpc/repos/CNNNER/datasets/few_shot/resume/train_1350.json"
TARGET_FILE = "/home/lpc/repos/CNNNER/datasets/few_shot/resume/train_1350_split.jsonl"

with open(ORI_JSON, "r") as f:
    ori_data = f.readlines()

data = [json.loads(line) for line in ori_data]

result = []

# 如果在运行过程中报错, 则为数据集本身的Label有问题(不是以B-或S-开头), 需要手动查询line_idx查询修正
for line_idx, item in enumerate(data):
    item['entities'] = []
    entity_item = {}
    is_started = False
    for idx, label in enumerate(item["label"]):
        if label == "O":
            # 异常处理
            if is_started:
                is_started = False
                entity_item['end'] = idx
                entity_item['text'] = item['text'][entity_item['start']: entity_item['end']]
                item['entities'].append(deepcopy(entity_item))
                entity_item.clear()
            continue
        flag, l = label.split("-")
        if flag in ["B", "S"]:
            is_started = True
            entity_item['start'] = idx
            entity_item['entity'] = l
        if flag in ["E", "S"]:
            is_started = False
            entity_item['end'] = idx + 1
            entity_item['text'] = item['text'][entity_item['start']: entity_item['end']]
            item['entities'].append(deepcopy(entity_item))
            entity_item.clear()
    result.append(item)


with open(TARGET_FILE, "w") as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# %%
