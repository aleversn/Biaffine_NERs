# %%
import json
import json_repair
import numpy as np

with open('/home/lpc/repos/CNNNER/datasets/few_shot/ud/labels.txt') as f:
    label_data = f.readlines()
label_data = [label.strip() for label in label_data]
label_dict = {label: i for i, label in enumerate(label_data)}  # Map label to index

with open('/home/lpc/repos/CNNNER/datasets/few_shot/ud/test.jsonl') as f:
    ori_data = f.readlines()
ori_data = [json.loads(i) for i in ori_data]

with open('/home/lpc/repos/CNNNER/data_record/llm_ner/ud_GLM4_LLM_Infer/entity_test.jsonl') as f:
    pred_data = f.readlines()
preds_list = []
for line in pred_data:
    text, pred_json = line.split('\t')
    try:
        pred_json = json_repair.loads(pred_json)
        if type(pred_json) != list:
            pred_json = []
    except:
        pred_json = []
    format_pred_json = []
    for item in pred_json:
        if type(item) != dict or 'text' not in item or 'entity' not in item or item['entity'] not in label_dict:
            continue
        t = item['text']
        start = text.find(t)
        if start == -1:
            continue
        end = start + len(t)
        format_pred_json.append({'start': start, 'end': end, 'text': list(t), 'label': label_dict[item['entity']]})
    
    preds_list.append(format_pred_json)

result = []

matches = []
preds = []
golds = []
for item, pred_item in zip(ori_data, preds_list):
    text = item['text']
    text_str = ''.join(text)
    entities = item["entities"]
    pred_list = []
    gold_list = []
    pred_set = [(item['start'], item['end'], item['label']) for item in pred_item]
    gold_set = [(item['start'], item['end'], label_dict[item['entity']]) for item in entities]
    matches.append(len(set(pred_set)&set(gold_set)))
    preds.append(len(set(pred_set)))
    golds.append(len(set(gold_set)))
    for pred in pred_item:
        t = ''.join(text[pred['start']:pred['end']])
        l = pred['label']
        pred_list.append((t, l))
    for gold in entities:
        t = ''.join(text[gold['start']:gold['end']])
        l = label_dict[gold['entity']]
        gold_list.append((t, l))
    result.append([text_str, pred_list, gold_list])

with open('result.txt', 'w') as f:
    for item in result:
        f.write(f'{item[0]}\t{str(item[1])}\t{str(item[2])}' + '\n')

print(f'matches: {np.sum(matches)}\n')
print(f'preds: {np.sum(preds)}\n')
print(f'golds: {np.sum(golds)}\n')
print(f'P: {np.sum(matches)/np.sum(preds)}\n')
print(f'R: {np.sum(matches)/np.sum(golds)}\n')
print(f'F1: {2*np.sum(matches)/(np.sum(preds)+np.sum(golds))}\n')

# %%
