# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np
import torch
import random
import json_repair
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def rgba(r, g, b, a=1.0):
    def norm(x):
        if x < 0:
            x = 0
        elif x > 1:
            x = 1.0
        return x
    return (norm(r/255), norm(g/255), norm(b/255), norm(a))


def show_tSNE(vectors, labels, label_text=None, colors=None, point_size=30, alpha=0.2):
    '''
    vectors: (nums, embedding_size)
    labels: (nums)
    label_text: list of str
    colors: list of (r, g, b, a)
    point_size: int
    alpha: float
    '''
    cls_num = np.unique(labels).shape[0]
    if label_text is None:
        label_text = [str(i) for i in range(cls_num)]
    if colors is None:
        colors = [rgba(173, 38, 45, 0.8) for i in range(cls_num)]
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(vectors)

    # 5. 可视化结果
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        sns.scatterplot(x=tsne_results[idx, 0], y=tsne_results[idx, 1],
                        color=colors[i], label=label_text[i], edgecolors='w', s=point_size, alpha=alpha)
    plt.title("t-SNE visualization of BERT CLS vectors")
    plt.show()


def batcher(model, tokenizer, list, batch_size=128, label_index=0, gpu=[0]):
    num_batch = len(list) // batch_size + 1

    embeddings_list = []
    for i in tqdm(range(num_batch)):
        batch = list[i*batch_size:(i+1)*batch_size]
        batch = tokenizer(batch, padding=True,
                          truncation=True, return_tensors='pt')
        embeddings = get_embeddings(model, batch, gpu=gpu)
        embeddings_list += embeddings.tolist()
    labels = [label_index] * len(embeddings_list)
    return np.array(embeddings_list), np.array(labels)


from_pretrained_path = '/home/lpc/models/simcse-chinese-roberta-wwm-ext/'
tokenizer = BertTokenizer.from_pretrained(from_pretrained_path)
model = BertModel.from_pretrained(from_pretrained_path)

# %%
with open('/home/lpc/repos/CNNNER/datasets/weibo_DA/train_entity_data_glm.jsonl') as f:
    ori_json = f.readlines()

entities_list = []
for item in ori_json:
    item = item.split('\t')
    entities_list.append(json_repair.loads(item[1]))

groups = {}

for item in entities_list:
    for ent_item in item:
        if 'entity' not in ent_item or 'type' not in ent_item:
            continue
        entity, typeof = ent_item['entity'], ent_item['type']
        typeof = typeof.strip().lower()
        if typeof not in groups:
            groups[typeof] = []
        groups[typeof].append(entity + '是' + typeof)

# %%
colors = [rgba(131, 131, 243, 0.8),
          rgba(255, 177, 110, 0.8),
          rgba(202, 150, 198, 0.8),
          rgba(126, 129, 239, 0.78),
          rgba(141, 128, 247, 0.81),
          rgba(100, 135, 232, 0.76),
          rgba(133, 100, 244, 0.84),
          rgba(128, 123, 100, 0.79),
          rgba(90, 136, 245, 0.82),
          rgba(125, 90, 241, 0.8),
          rgba(252, 172, 90, 0.75),
          rgba(80, 182, 103, 0.77),
          rgba(243, 80, 120, 0.79),
          rgba(255, 173, 80, 0.83),
          rgba(70, 178, 118, 0.76),
          rgba(255, 70, 112, 0.8),
          rgba(202, 144, 70, 0.85),
          rgba(210, 152, 202, 0.79),
          rgba(198, 155, 190, 0.78),
          rgba(206, 148, 200, 0.84),
          rgba(200, 149, 193, 0.8),
          rgba(197, 151, 205, 0.76),
          rgba(201, 153, 200, 0.81)]

key_list = []
emb_list = []
label_list = []
color_list = []
l_count = 0
for key in groups:
    # if key == 'other':
    #     continue
    embeddings, labels = batcher(model, tokenizer, groups[key], gpu=[
                                 0], label_index=l_count)
    emb_list.append(embeddings)
    label_list.append(labels)
    key_list.append(key)
    color_list.append(rgba(random.randint(0,255), random.randint(0,255), random.randint(0,255), 1))
    l_count += 1

# %%
embeddings = np.concatenate(emb_list, axis=0)
labels = np.concatenate(label_list, axis=0)

show_tSNE(embeddings, labels, label_text=key_list,
          colors=color_list, alpha=0.5)

# %%
