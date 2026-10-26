import sys
import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
from src.preprocessor import VietnameseTextPreprocessor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

class Vocabulary:
    """Từ điển ánh xạ từ vựng sang chỉ số cho BiLSTM."""
    def __init__(self, min_freq=2, pad_token="<pad>", unk_token="<unk>"):
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.w2i = {pad_token: 0, unk_token: 1}
        self.i2w = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)
        
        for word, count in counter.items():
            if count >= self.min_freq and word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word
        print(f"Vocabulary size: {len(self.w2i)} (min_freq={self.min_freq})")

    def __len__(self):
        return len(self.w2i)

    def encode(self, text, max_len=64):
        tokens = text.split()[:max_len]
        indices = [self.w2i.get(token, self.w2i[self.unk_token]) for token in tokens]
        length = len(indices)
        if len(indices) < max_len:
            indices = indices + [self.w2i[self.pad_token]] * (max_len - len(indices))
        return indices, length, tokens

class BiLSTMDataset(Dataset):
    """Dataset cho BiLSTM + Attention."""
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, preprocessor: VietnameseTextPreprocessor, max_len=64):
        self.max_len = max_len
        self.vocab = vocab
        self.preprocessor = preprocessor
        
        # Lọc bỏ các dòng thiếu dữ liệu
        df_clean = df.dropna(subset=['text', 'label']).copy()
        df_clean = df_clean[df_clean['label'].isin(LABEL2ID.keys())]
        
        self.texts = df_clean['text'].astype(str).tolist()
        self.labels = [LABEL2ID[lbl] for lbl in df_clean['label'].tolist()]
        
        # Tiền xử lý văn bản (chuẩn hóa & tách từ)
        self.processed_texts = [self.preprocessor.preprocess(t) for t in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.processed_texts[idx]
        label = self.labels[idx]
        indices, length, _ = self.vocab.encode(text, self.max_len)
        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": self.texts[idx],
            "processed_text": text
        }

class PhoBERTDataset(Dataset):
    """Dataset cho PhoBERT Fine-tuning."""
    def __init__(self, df: pd.DataFrame, tokenizer, preprocessor: VietnameseTextPreprocessor, max_len=64):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_len = max_len
        
        # Lọc bỏ các dòng thiếu dữ liệu
        df_clean = df.dropna(subset=['text', 'label']).copy()
        df_clean = df_clean[df_clean['label'].isin(LABEL2ID.keys())]
        
        self.texts = df_clean['text'].astype(str).tolist()
        self.labels = [LABEL2ID[lbl] for lbl in df_clean['label'].tolist()]
        self.processed_texts = [self.preprocessor.preprocess(t) for t in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.processed_texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": self.texts[idx],
            "processed_text": text
        }
