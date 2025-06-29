# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

def get_average_distance(list1, list2):
    # 将 list1 和 list2 转换为矩阵 [n, 768] 和 [m, 768]
    mat1 = torch.tensor(list1)  # 维度 [n, 768]
    mat2 = torch.tensor(list2)  # 维度 [m, 768]
    
    # 计算每对向量的欧氏距离
    # 扩展维度以实现广播计算
    mat1_expanded = mat1.unsqueeze(1)  # 维度 [n, 1, 768]
    mat2_expanded = mat2.unsqueeze(0)  # 维度 [1, m, 768]
    
    # 计算欧氏距离的平方
    distance_matrix = torch.norm(mat1_expanded - mat2_expanded, dim=2, p=2)  # 维度 [n, m]
    
    # 求平均距离
    average_distance = distance_matrix.mean()
    return average_distance

from_pretrained_path = '/home/lpc/models/simcse-chinese-roberta-wwm-ext/'
tokenizer = AutoTokenizer.from_pretrained(from_pretrained_path)
model = AutoModel.from_pretrained(from_pretrained_path)

# %%
ORI_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/resume/train_1000.jsonl'
EXT_FILE = '/home/lpc/repos/CNNNER/datasets/few_shot/resume_DA/1000/train_1000_split_synthetic.jsonl'

with open(ORI_FILE) as f:
    ori_data = f.readlines()
ori_data = [json.loads(line) for line in ori_data]

ori_text = [''.join(item['text']) for item in ori_data]
ori_embeddings = batcher(model, tokenizer, ori_text, batch_size=256, gpu=[0])

# %%

with open(EXT_FILE) as f:
    ext_data = f.readlines()
ext_data = [json.loads(line) for line in ext_data]

ext_text = [''.join(item['text']) for item in ext_data]
ext_embeddings = batcher(model, tokenizer, ext_text, batch_size=256, gpu=[0])

# %%
scores = get_average_distance(ori_embeddings, ext_embeddings)
print(scores.mean())

# cos_socres = get_distance(emb_sts, emb_wiki)

# %%
