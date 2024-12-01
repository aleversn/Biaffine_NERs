## LLM DA for Knowledge Fusion for Rich and Efficient Extraction Model

## 🧭 使用指南

### 🔨 安装依赖

如果同时需要用到`GLM3`, `GLM4`, `QWen2`和`Llama3`等模型, 因此需要安装两个conda环境以兼容新旧版本模型。其中`GLM3`和其它模型对`transformers`库版本的要求不同, 因此 需要分别安装对应版本的`transformers`库。

如果你只需使用到`GLM3`或是除`GLM3`外的其他模型，可以只选择一种conda环境。

1. 创建一个conda环境, 例如`llm`:

``` bash
conda create -n llm python=3.10
conda activate llm
```

2. 安装所需依赖:

- 针对`GLM3`:

```bash
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

- 针对其他模型

```bash
pip install protobuf transformers==4.44 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate tiktoken
```

### 🚀 运行Pipeline

#### 定义

- LLM DA中利用LLM对原始数据进行`entity`和`pos`扩展标注, 最终生成`fusion`数据
- LLM DA还对原始数据进行实体描述合成, 最终生成`syn_fusion`数据

> 如果你想要在jupyter kernel中运行代码, 请将`cmd_args`设为`False`.

1. 标注扩展`entity`和`pos`

```bash
python syn_pipelines/s1_predict_data.py \
    --n_gpu=0 \
    --skip=-1 \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --model_from_pretrained="/home/lpc/models/glm-4-9b-chat/" \
    --batch_size=20 \
    --mode=entity
```

- `n_gpu`: 指定使用的GPU索引 (默认为0)
- `skip`: 跳过已处理文件的数量(默认为-1, 即全部处理)
- `file_dir`: 所有数据集的根目录(默认为"./data/few_shot")
- `file_name`: 数据集名称, 如 "weibo", 你必须确保`./data/few_shot/weibo`底下有`train_1000.jsonl`这个文件
- `save_type_name`: 要保存的数据文件夹前缀 (默认为"GLM4")
- `model_from_pretrained`: 预训练模型的路径
- `batch_size`: 批次大小, 默认为20
- `mode`: 模式(entity/pos), 默认为"entity"

一般来说, 对于不怎么需要修改的参数, 可以直接使用默认值即可。

2. 丰富`entity`解释生成合成数据

```bash
python syn_pipelines/s2_continous_generation.py \
    --n_gpu= 0 \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --model_from_pretrained="/home/lpc/models/glm-4-9b-chat/" \
    --batch_size=40 \
```

3. 标注合成数据的扩展`entity`和`pos`

参数和步骤1一样

```bash
python syn_pipelines/s3_predict_syn_data.py \
    --n_gpu=0 \
    --skip=-1 \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --model_from_pretrained="/home/lpc/models/glm-4-9b-chat/" \
    --batch_size=20 \
    --mode=entity
```

4. 分割数据, 将数据自动按`25%`, `50%`, `100%`分类

```bash
python syn_pipelines/s4_split_data.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4
```

5. 利用步骤`1`和步骤`3`的标注数据合并到对应源数据中, 并分别生成合并的`fusion`标签和`syn_fusion`标签

```bash
python syn_pipelines/s5_merge_data.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --is_syn=0  \
    --save_type_name=GLM4 \
    --label_prefix='' \
    --entity_label='./data/fusion_knowledge/entity_label.json' \
    --pos_label='./data/fusion_knowledge/pos_label.json'
```

- `is_syn`: 是否是合成数据(0表示不是, 1表示是)
- `label_prefix`: 标签前缀(如果扩展标签与原始标签有重名, 建议设置前缀)
- `entity_label`: 实体标签标准化文件路径
- `pos_label`: 词性标签标准化文件路径

6. 合并`fusion`标签和`syn_fusion`标签

```bash
python syn_pipelines/s6_combine_syn_fusion_label.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4
```

7. 利用NER模型预测合成数据的原始目标标签

在执行这一步之前, 你可以先对`fusion`数据集进行训练了, 训练后再利用预训练模型来标注合成数据集的标签

```bash
python syn_pipelines/s7_pred_syn_data.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --from_pretrained='/home/lpc/models/chinese-bert-wwm-ext/' \
    --model_from_pretrained='/home/lpc/repos/Biaffine_NERs/save_model/CNNNER-youku_1000_fusion_sota/cnnner_best' \
    --batch_size=4
```

- `from_pretrained`: 预训练模型路径
- `model_from_pretrained`: 模型保存路径(之所以分为两个路径是因为保存模型可能没包含tokenizer的配置文件)

8. 利用步骤`1`,步骤`3`和步骤`7`的标注数据合并到对应源数据中, 并分别生成合并的`fusion`标签和`syn_fusion`标签, 并最终将`fusion`数据合并到`syn_fusion`数据中

```bash
python syn_pipelines/s8_process_label_and_combine.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --is_syn=0  \
    --save_type_name=GLM4 \
    --label_prefix='' \
    --entity_label='./data/fusion_knowledge/entity_label.json' \
    --pos_label='./data/fusion_knowledge/pos_label.json'
```