import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttention(nn.Module):
    """
    Mô hình Bidirectional LSTM kết hợp cơ chế Attention cho phân loại cảm xúc.
    
    Kiến trúc:
      1. Embedding Layer: Biến đổi token ID thành vector embedding d_e.
      2. Bidirectional LSTM: Trích xuất đặc trưng ngữ cảnh hai chiều (tiến và lùi).
      3. Attention Layer (Additive/Bahdanau-style self-attention):
         - Tính điểm mức độ quan trọng u_t = tanh(W_a * h_t + b_a)
         - Chiếu qua context vector v_a: score_t = v_a^T * u_t
         - Che token <pad> bằng mask (-1e9) rồi áp dụng softmax để ra phân phối trọng số alpha_t.
         - Vector tổng hợp câu (context vector) c = sum(alpha_t * h_t).
      4. Fully-Connected Classification Head: Dropout -> Dense -> ReLU -> Dense -> Logits (3 lớp).
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 2,
        attention_dim: int = 64,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(BiLSTMAttention, self).__init__()
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Attention Mechanism
        lstm_out_dim = hidden_dim * 2  # Bi-directional
        self.attention_proj = nn.Linear(lstm_out_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, lengths=None):
        """
        Args:
            input_ids: Tensor (batch_size, seq_len)
            lengths: Tensor (batch_size,) - độ dài thực tế của câu (tùy chọn)
        Returns:
            logits: Tensor (batch_size, num_classes)
            attention_weights: Tensor (batch_size, seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(self.embedding(input_ids))
        
        # LSTM output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention scoring
        # u = tanh(W_a * h_t) -> (batch_size, seq_len, attention_dim)
        u = torch.tanh(self.attention_proj(lstm_out))
        # scores = v_a^T * u -> (batch_size, seq_len, 1) -> squeeze to (batch_size, seq_len)
        scores = self.context_vector(u).squeeze(-1)
        
        # Masking padding tokens
        if self.pad_idx is not None:
            mask = (input_ids == self.pad_idx)
            scores = scores.masked_fill(mask, -1e9)
            
        # Attention weights alpha_t via Softmax: (batch_size, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Context vector c = sum(alpha_t * h_t)
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, lstm_out_dim) -> (batch_size, 1, lstm_out_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Logits: (batch_size, num_classes)
        logits = self.classifier(context)
        
        return logits, attention_weights

if __name__ == "__main__":
    vocab_size = 500
    model = BiLSTMAttention(vocab_size=vocab_size)
    dummy_input = torch.randint(0, vocab_size, (4, 16))
    dummy_input[0, 10:] = 0  # pad
    logits, attn = model(dummy_input)
    print("BiLSTM+Attention initialized successfully!")
    print(f"Logits shape: {logits.shape}, Attention weights shape: {attn.shape}")
    print(f"Attn sum sample 0: {attn[0].sum().item():.4f}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
