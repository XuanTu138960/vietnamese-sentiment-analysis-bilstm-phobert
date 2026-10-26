import sys
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class PhoBERTClassifier(nn.Module):
    """
    Wrapper cho mô hình PhoBERT tiền huấn luyện (vinai/phobert-base-v2)
    sử dụng cho bài toán phân loại cảm xúc 3 nhãn.
    """
    def __init__(self, model_name: str = "vinai/phobert-base-v2", num_classes: int = 3, dropout: float = 0.2):
        super(PhoBERTClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            classifier_dropout=dropout,
            output_attentions=True
        )
        self.phobert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )

    def forward(self, input_ids, attention_mask=None, output_attentions: bool = False):
        """
        Args:
            input_ids: Tensor (batch_size, seq_len)
            attention_mask: Tensor (batch_size, seq_len)
            output_attentions: bool
        Returns:
            logits: Tensor (batch_size, num_classes)
            attentions: tuple of Tensors if output_attentions=True else None
        """
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        logits = outputs.logits
        attentions = outputs.attentions if output_attentions else None
        return logits, attentions

if __name__ == "__main__":
    print("Testing PhoBERTClassifier creation...")
    model = PhoBERTClassifier(num_classes=3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PhoBERT loaded. Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
