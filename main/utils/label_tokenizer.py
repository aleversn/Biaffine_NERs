from typing import List, Union
from tqdm import tqdm

class LabelTokenizer():
    
    def __init__(self, filename: str=None) -> None:
        self.idx2label = []
        self.label2idx = {}
        if filename is not None:
            with open(filename, 'r') as f:
                for idx, line in tqdm(enumerate(f), desc='Loading labels'):
                    self.idx2label.append(line.strip())
                    self.label2idx[line.strip()] = idx  
                    
    def load(self, labels: List[str], sort=True):
        if sort:
            # 排序确保顺序
            labels = list(labels)
            labels.sort()
        for label in labels:
            self.idx2label.append(label)
        for idx, label in enumerate(self.idx2label):
            self.label2idx[label] = idx
        return self
    
    def remove(self, index: int):
        label = self.idx2label[index]
        self.idx2label.remove(label)
        self.label2idx = {}
        for idx, label in enumerate(self.idx2label):
            self.label2idx[label] = idx
        return self
    
    def add(self, item, index=-1):
        if index!=-1:
            self.idx2label.insert(index, item)
        else:
            self.idx2label.append(item)
        for idx, label in enumerate(self.idx2label):
            self.label2idx[label] = idx
        return self
    
    def convert_tokens_to_ids(self, label: Union[str, List[str]]) -> int:
        if isinstance(label, list):
            return [self.label2idx[l] for l in label]
        return self.label2idx[label]
    
    def convert_ids_to_tokens(self, idx: Union[int, List[int]]) -> int:
        if isinstance(idx, list):
            return [self.idx2label[i] for i in idx]
        return self.idx2label[idx]
    
    def __len__(self):
        return len(self.idx2label)