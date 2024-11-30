# %%
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
    return (r/255, g/255, b/255, a)

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
with open('/home/lpc/repos/CNNNER/datasets/few_shot/resume/train_250.jsonl') as f:
    ori_json = f.readlines()

source_list = [''.join(json.loads(item)['text']) for item in ori_json]

embeddings_1, labels_1 = batcher(model, tokenizer, source_list, gpu=[0])

with open('/home/lpc/repos/CNNNER/datasets/few_shot/resume_DA/250/train_250_synthetic.jsonl') as f:
    ori_json = f.readlines()

target_list = [''.join(json.loads(item)['text']) for item in ori_json]

embeddings_2, labels_2 = batcher(model, tokenizer, target_list, gpu=[0], label_index=1, batch_size=512)

with open('/home/lpc/repos/CNNNER/datasets/few_shot/resume/test.jsonl') as f:
    ori_json = f.readlines()

test_list = [''.join(json.loads(item)['text']) for item in ori_json]

embeddings_3, labels_3 = batcher(model, tokenizer, test_list, gpu=[0], label_index=2)

embeddings = np.concatenate([embeddings_3, embeddings_1, embeddings_2], axis=0)
labels = np.concatenate([labels_3, labels_1, labels_2], axis=0)

show_tSNE(embeddings, labels, label_text=['Test', 'Train', 'Synthetic'], colors=[rgba(131, 131, 243, 0.8), rgba(255, 177, 110, 0.8), rgba(202, 150, 198, 0.8)], alpha=0.5)

# %%
