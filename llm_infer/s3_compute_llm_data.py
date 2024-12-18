# %%
import os
import json
import json_repair
import numpy as np

DIR = '/home/lpc/repos/CNNNER/datasets/few_shot'
DATA_LIST = ['cmeee', 'msr', 'pku', 'resume', 'taobao', 'ud', 'weibo', 'youku']
few_shot = [250, 500, 1000]
prompt = '''指令: 请识别并抽取"{domain}"领域输入句子的命名实体，并使用JSON格式的数组进行返回，子项包括实体的实体文本(text)和实体类型(entity)：
格式要求: 1. 输出格式为[{{text: '', entity: ''}}]
2. 实体类型只包含{entity_list}这几种
3. 如果不存在任何实体，请输出空数组[]
输入: {text}
输出: '''

result = []
for data in DATA_LIST:
    data_result = []
    with open(os.path.join(DIR, data, 'labels.txt')) as f:
        label_data = f.readlines()
    label_data = [label.strip() for label in label_data]

    with open(os.path.join(DIR, data, 'train_1000.jsonl')) as f:
        ori_data = f.readlines()
    ori_data = [json.loads(i) for i in ori_data]
    
    for item in ori_data:
        text, entities = ''.join(item['text']), item['entities']
        entities = [{'text': i['text'], 'entity': i['entity']} for i in entities]
        input_item = [
            {
                "role": "user",
                "content": prompt.format(domain=data, entity_list=json.dumps(
            label_data, ensure_ascii=False), text=text)
            },
            {
                "role": "assistant",
                "content": json.dumps(entities, ensure_ascii=False)
            }
        ]
        data_result.append(input_item)
    result.append(data_result)

for shot in few_shot:
    all = []
    for result_item in result:
        all.extend(result_item[:shot])
    with open(f'few_shot_zh_ner_{shot}_train.jsonl', 'w', encoding='utf-8') as f:
        for item in all:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
