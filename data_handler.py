import torch
from torch.utils.data import Dataset

class DataHandler(Dataset):
    def __init__(self, texts, max_seq_len, tokenizer):
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        texts = str(self.texts.iloc[idx])
        inputs = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        inputs["labels"] = inputs["input_ids"].clone() 

        return inputs