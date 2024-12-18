# %%
import os
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
import sys
sys.path.append("../")
cmd_args = True
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=0, help='n_gpu')
parser.add_argument('--skip', default=-1, help='skip the first n lines, the skip index is count from the start index of n-th chunks')
parser.add_argument('--file_dir', default='/home/lpc/repos/CNNNER/datasets/few_shot', help='file name')
parser.add_argument('--file_name', default='weibo', help='file name of the dataset, you should make sure it contains `test.jsonl` file')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/chatglm3-6b/', help='model from pretrained')
parser.add_argument('--peft_pretrained', default='/home/lpc/repos/ChatGLM_PEFT/save_model/fewshot_ner_250/ChatGLM_5000', help='model from pretrained')
parser.add_argument('--batch_size', default=20, help='batch size')
parser.add_argument('--eval_mode', default='test.jsonl', help='choose dev or test to eval')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

import json
import random
from tqdm import tqdm

if args.save_type_name == 'GLM3':
    from llm.chatglm_lora import Predictor
else:
    from llm.llm_lora import Predictor

SOURCE_FILE = os.path.join(args.file_dir, args.file_name, args.eval_mode)
LABEL_FILE = os.path.join(args.file_dir, args.file_name, 'labels.txt')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_LLMLoRA_Infer'
basename = os.path.basename(SOURCE_FILE)
pred = Predictor(model_from_pretrained=args.model_from_pretrained, resume_path=args.peft_pretrained)

labels = []
with open(LABEL_FILE) as f:
    ori_data = f.readlines()
for l in ori_data:
    if l.strip() != '' and l.strip() != 'O':
        labels.append(l.strip())

prompt = '''指令: 请识别并抽取"{domain}"领域输入句子的命名实体，并使用JSON格式的数组进行返回，子项包括实体的实体文本(text)和实体类型(entity)：
格式要求: 1. 输出格式为[{{text: '', entity: ''}}]
2. 实体类型只包含{entity_list}这几种
3. 如果不存在任何实体，请输出空数组[]
输入: {text}
输出: '''
with open(SOURCE_FILE) as f:
    ori_data = f.readlines()

ori_data = [json.loads(i) for i in ori_data]
selected_data = ori_data[int(args.skip):] if int(args.skip) > -1 else ori_data
data = []

for item in tqdm(ori_data):
    text = item['text']
    text = ''.join(text)
    user_content = prompt.format(domain=args.file_name, entity_list=json.dumps(labels, ensure_ascii=False), text=text)
    data.append((user_content, text))

num_batches = len(data) // int(args.batch_size) + 1 if len(data) % int(args.batch_size) != 0 else len(data) // int(args.batch_size)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def denoise(ori):
    ori = str(ori)
    ori = ori.replace('```json', '')
    ori = ori.replace('\n', '')
    ori = ori.replace('```', '')
    return ori

for i in tqdm(range(num_batches)):
    max_length = 0
    samples = data[i * int(args.batch_size) : (i + 1) * int(args.batch_size)]
    oris = [item[1] for item in samples]
    inputs = []
    for item in samples:
        inputs.append(item[0])
        if len(item[0]) > max_length:
            max_length = len(item[0])
    outputs = pred(inputs, max_new_tokens=5*max_length, temperature=0.8, build_message=True)
    with open(os.path.join(SAVE_DIR, f'entity_{basename}'), 'a', encoding='utf-8') as f:
        for ori, out in zip(oris, outputs):
            f.write('{}\t{}\n'.format(ori, denoise(out)))

# %%
