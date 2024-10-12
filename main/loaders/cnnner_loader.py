# 构造Dataloader
from typing import Any, Dict, List
from torch.utils.data import Dataset
from transformers import BertTokenizer
from main.utils.label_tokenizer import LabelTokenizer
from tqdm import tqdm
import torch
import json
import random


class CNNNERDataset(Dataset):

    def __init__(self, tokenizer: BertTokenizer, labelTokenizer: LabelTokenizer, filename: str, shuffle=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.labelTokenizer = labelTokenizer
        self.filename = filename
        self.num_labels = len(self.labelTokenizer)-1
        self.data = self.load_jsonl()
        self.process_data = self.process_dataset()
        self.shuffle_list = [i for i in range(len(self.process_data))]
        if shuffle:
            random.shuffle(self.shuffle_list)

    def load_jsonl(self):
        data = []
        assert self.filename.endswith(
            ".jsonl"), f"Invalid file format: {self.filename}"
        with open(self.filename, "r", encoding="utf-8") as f:
            from tqdm import tqdm
            bar = tqdm(f, desc=f"Loading {self.filename}")
            for line in bar:
                data.append(json.loads(line.strip()))
        return data

    def process_dataset(self):
        process_data = []
        for sample in tqdm(self.data, desc="converting data..."):
            process_data.append(self.transform(sample))
        return process_data

    def transform(self, sample: Dict[str, any]):
        convert_sample = {
            "input_ids": None,
            "bpe_len": None,
            "labels": None,
            "indexes": None
        }
        text, entities = sample["text"], sample["entities"]
        pieces = list(self.tokenizer.tokenize(word) for word in text)
        pieces = list(self.tokenizer.unk_token if len(
            piece) == 0 else piece for piece in pieces)
        flat_tokens = [i for piece in pieces for i in piece]
        length = len(text)
        bert_length = len(flat_tokens) + 2
        input_ids = torch.tensor([self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(
            flat_tokens) + [self.tokenizer.sep_token_id], dtype=torch.long)
        labels = torch.zeros(
            (length, length, self.num_labels), dtype=torch.long)
        for entity in entities:
            start, end, label = entity["start"], entity["end"] - \
                1, entity["entity"]
            label_id = self.labelTokenizer.convert_tokens_to_ids(label)
            labels[start, end, label_id-1] = 1
            # 原论文中是计算的上下三角形，但是实际的话是只使用上边的三角形效果好
            # labels[end, start, label_id-1] = 1
        indexes = torch.zeros(bert_length, dtype=torch.long)
        offset = 1
        for i, piece in enumerate(pieces):
            indexes[offset: offset+len(piece) + 1] = i + 1
            offset += len(piece)
        convert_sample["input_ids"] = input_ids
        convert_sample["bpe_len"] = torch.tensor(bert_length, dtype=torch.long)
        convert_sample["labels"] = labels
        convert_sample["indexes"] = indexes
        return convert_sample

    def __getitem__(self, index) -> Any:
        index = self.shuffle_list[index]
        return self.process_data[index]

    def __len__(self) -> int:
        return len(self.process_data)


class CNNNERPadCollator:

    def __init__(self):
        pass

    def pad_1d(self, x: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
        max_length = max(i.size(0) for i in x)
        paddings = torch.full((len(x), max_length) +
                              x[0].size()[2:], padding_value, dtype=x[0].dtype)
        for i in range(len(x)):
            paddings[i, :x[i].size(0)] = x[i]
        return paddings

    def pad_2d(self, x: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
        """
        对序列进行二维补全
        """
        max_rows = max(i.size(0) for i in x)
        max_cols = max(i.size(1) for i in x)
        paddings = torch.full((len(x), max_rows, max_cols) +
                              x[0].size()[2:], padding_value, dtype=x[0].dtype)
        # print(paddings.size(),x[0].size())
        for i in range(len(x)):
            paddings[i, :x[i].size(0), :x[i].size(1)] = x[i]
        return paddings

    def __call__(self, samples: List[Any]):
        # for i in samples:
        #     print(i['labels'].size())
        convert_example = {
            "input_ids": self.pad_1d(list(i["input_ids"] for i in samples), 0),
            "bpe_len": torch.stack(list(i["bpe_len"] for i in samples)),
            "labels": self.pad_2d(list(i["labels"] for i in samples), 0),
            "indexes": self.pad_1d(list(i["indexes"] for i in samples), 0)
        }

        return convert_example
