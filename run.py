# %%
from main.trainers.fusion_ner_trainer import Trainer
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained(
    "/home/lpc/models/chinese-bert-wwm-ext/")
config = BertConfig.from_pretrained(
    "/home/lpc/models/chinese-bert-wwm-ext/")
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/home/lpc/models/chinese-bert-wwm-ext/',
                  data_name='weibo_fusion', batch_size=4, task_name='CNNNER-weibo')

for i in trainer(num_epochs=60, eval_call_step=lambda x: x % 250 == 0):
    a = i

# %%
