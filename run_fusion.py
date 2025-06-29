# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from main.trainers.fusion_ner_trainer import Trainer
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained(
    "/home/lpc/models/chinese-bert-wwm-ext/")
config = BertConfig.from_pretrained(
    "/home/lpc/models/chinese-bert-wwm-ext/")
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/home/lpc/models/chinese-bert-wwm-ext/',
                  data_name='msr_GLM4_1000_fusion_synthetic',
                  batch_size=8,
                  batch_size_eval=2,
                  task_name='CNNNER-msr_GLM4_1000_fusion_synthetic')

for i in trainer(num_epochs=120, other_lr=1e-3, weight_decay=0.01, remove_clashed=True, nested=False, eval_call_step=lambda x: x / 976 > 15 and x % 976 == 0):
    a = i

# %%
