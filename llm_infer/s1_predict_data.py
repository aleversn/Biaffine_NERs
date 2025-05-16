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
parser.add_argument('--file_dir', default='/home/lpc/repos/Biaffine_NERs/datasets/few_shot', help='file name')
parser.add_argument('--file_name', default='cmeee', help='file name of the dataset, you should make sure it contains `test.jsonl` file')
parser.add_argument('--llm_name', default='', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--save_type_name', default='Qwen3_32B', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/Qwen3-32B/', help='model from pretrained')
parser.add_argument('--vllm', default='1', help='whether use vllm')
parser.add_argument('--tensor_parallel_size', default=2, help='tensor_parallel_size (TP) for vLLM')
parser.add_argument('--batch_size', default=20, help='batch size')
parser.add_argument('--eval_mode', default='test.jsonl', help='choose dev or test to eval')
parser.add_argument('--skip_thinking', default='0', help='skip deep thinking in RL model with <think>\n\n</think>')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

API_MODELS = ['gpt-4o-mini', 'deepseek-chat', 'deepseek-reasoner']
API_CONFIGS = [('OpenAI', None), ('Deepseek', 'https://api.deepseek.com'), ('Deepseek', 'https://api.deepseek.com')]

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

USE_VLLM = str(args.vllm) == '1'

import json
import random
from tqdm import tqdm

llm_name = args.llm_name if args.llm_name != '' else args.save_type_name
if llm_name == 'GLM3':
    from llm.chatglm import Predictor
elif llm_name in API_MODELS:
    from llm.openai import Predictor
elif USE_VLLM:
    from main.predictor.vllm import Predictor
else:
    from llm.llm import Predictor

SOURCE_FILE = os.path.join(args.file_dir, args.file_name, args.eval_mode)
LABEL_FILE = os.path.join(args.file_dir, args.file_name, 'labels.txt')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_LLM_Infer'
basename = os.path.basename(SOURCE_FILE)

if llm_name not in API_MODELS:
    pred = Predictor(model_from_pretrained=args.model_from_pretrained, tensor_parallel_size=int(args.tensor_parallel_size))
else:
    CONFIG_INDEX = API_MODELS.index(llm_name)
    with open('api_key.txt') as f:
        api_keys = f.readlines()
    for key_item in api_keys:
        key_item = key_item.strip().split(' ')
        if len(key_item) == 1:
            api_key = key_item
            break
        else:
            if key_item[0] == API_CONFIGS[CONFIG_INDEX][0]:
                api_key = key_item[1]
                break
    pred = Predictor(api_key=api_key, base_url=API_CONFIGS[CONFIG_INDEX][1])

labels = []
with open(LABEL_FILE) as f:
    ori_data = f.readlines()
for l in ori_data:
    if l.strip() != '' and l.strip() != 'O':
        labels.append(l.strip())

prompt = '''指令: 请识别并抽取输入句子的命名实体，并使用JSON格式的数组进行返回，子项包括实体的开始位置(start)、结束位置(end)、实体文本(text)和实体类型(entity)：
格式要求: 1. 输出格式为[{{start: '', end: '', text: '', entity: ''}}]
2. 实体类型只包含{entity_list}这几种
3. 如果不存在任何实体，请输出空数组[]
输入: '''
with open(SOURCE_FILE) as f:
    ori_data = f.readlines()

ori_data = [json.loads(i) for i in ori_data]
selected_data = ori_data[int(args.skip):] if int(args.skip) > -1 else ori_data
data = []

for item in tqdm(ori_data):
    text = item['text']
    text = ''.join(text)
    user_content = prompt.format(entity_list=json.dumps(labels, ensure_ascii=False)) + text
    data.append((user_content, text))

def build_chat_custom(content):
    content = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return content

if args.skip_thinking == '1':
    for idx, tp in enumerate(data):
        data[idx] = (build_chat_custom(tp[0]), tp[1])

num_batches = len(data) // int(args.batch_size) + 1 if len(data) % int(args.batch_size) != 0 else len(data) // int(args.batch_size)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def denoise(ori):
    ori = ori.replace('```json', '')
    ori = ori.replace('\n', '')
    ori = ori.replace('```', '')
    return ori

if llm_name not in API_MODELS:
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
else:
    for ask_content, ori in tqdm(data):
        res = pred(ask_content, model=llm_name)
        res = res[0]
        res = res.replace('\n', '')
        res = res.replace(' ', '')
        with open(os.path.join(SAVE_DIR, f'entity_{basename}'), 'a', encoding='utf-8') as f:
            f.write('{}\t{}\n'.format(ori, denoise(res)))

# %%
