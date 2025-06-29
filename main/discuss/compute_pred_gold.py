# %%
import json

with open('/home/lpc/repos/CNNNER/datasets/few_shot/youku/test.jsonl') as f:
    ori_data = f.readlines()
ori_data = [json.loads(i) for i in ori_data]

with open('/home/lpc/repos/CNNNER/data_record/CNNNER-youku_1000_fusion_sota/pred_gold_best.jsonl') as f:
    pred_gold = f.readlines()
pred_gold = [json.loads(i) for i in pred_gold]

result = []

matches = 0
preds = 0
golds = 0
for item, pred_gold_item in zip(ori_data, pred_gold):
    text = item['text']
    text_str = ''.join(text)
    pred_list = []
    gold_list = []
    pred_set = [(item['start'], item['end'], item['label']) for item in pred_gold_item['preds']]
    gold_set = [(item['start'], item['end'], item['label']) for item in pred_gold_item['golds']]
    matches += len(set(pred_set)&set(gold_set))
    preds += len(set(pred_set))
    golds += len(set(gold_set))
    for pred in pred_gold_item['preds']:
        t = ''.join(text[pred['start']:pred['end']+1])
        l = pred['label']
        pred_list.append((t, l))
    for gold in pred_gold_item['golds']:
        t = ''.join(text[gold['start']:gold['end']+1])
        l = gold['label']
        gold_list.append((t, l))
    result.append([text_str, pred_list, gold_list])

with open('result.txt', 'w') as f:
    for item in result:
        f.write(f'{item[0]}\t{str(item[1])}\t{str(item[2])}' + '\n')

print(f'matches: {matches}\n')
print(f'preds: {preds}\n')
print(f'golds: {golds}\n')
print(f'P: {matches/preds}\n')
print(f'R: {matches/golds}\n')
print(f'F1: {2*matches/(preds+golds)}\n')

# %%
