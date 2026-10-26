import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Cài đặt font hiển thị Unicode cho tiếng Việt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

from src.preprocessor import VietnameseTextPreprocessor
from src.dataset import Vocabulary, ID2LABEL
from src.models.bilstm_attention import BiLSTMAttention

def visualize_bilstm_attention(
    model,
    vocab,
    preprocessor,
    sentences,
    output_path="outputs/figures/attention_visualization.png",
    device="cpu"
):
    """
    Trực quan hóa phân phối trọng số Attention của BiLSTM trên các câu mẫu.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(len(sentences), 1, figsize=(12, 3.2 * len(sentences)))
    if len(sentences) == 1:
        axes = [axes]
        
    print("\n=== TRỰC QUAN HÓA TRỌNG SỐ ATTENTION CỦA BiLSTM ===")
    
    for idx, (raw_sent, true_label) in enumerate(sentences):
        # 1. Tiền xử lý
        processed = preprocessor.preprocess(raw_sent)
        tokens = processed.split()
        
        # 2. Mã hóa sang token ID
        indices = [vocab.w2i.get(tok, vocab.w2i["<unk>"]) for tok in tokens]
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        
        # 3. Forward pass qua mô hình
        with torch.no_grad():
            logits, attn_weights = model(input_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            pred_label = ID2LABEL[pred_id]
            attn = attn_weights[0].cpu().numpy()
            
        print(f"\n[Câu {idx+1}] Gốc: \"{raw_sent}\"")
        print(f"       Tách từ: \"{processed}\"")
        print(f"       Dự đoán: {pred_label.upper()} ({probs[pred_id]*100:.2f}%) | Nhãn gốc: {true_label.upper()}")
        print("       Top 3 từ được chú ý nhiều nhất:")
        sorted_indices = np.argsort(attn)[::-1]
        for rank, token_i in enumerate(sorted_indices[:3], 1):
            print(f"         {rank}. '{tokens[token_i]}': {attn[token_i]*100:.2f}%")
            
        # 4. Vẽ Heatmap / Bar chart trọng số attention
        ax = axes[idx]
        
        # Heatmap dạng 1D matrix
        attn_matrix = np.expand_dims(attn, axis=0)
        
        sns.heatmap(
            attn_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar=True,
            xticklabels=tokens,
            yticklabels=["Trọng số $\\alpha$"],
            ax=ax,
            linewidths=1.0,
            linecolor="white",
            annot_kws={"size": 11, "weight": "bold"}
        )
        
        ax.set_title(
            f"Câu {idx+1}: \"{raw_sent}\"\n"
            f"Dự đoán: {pred_label.upper()} (Xác suất: {probs[pred_id]*100:.1f}%) | Nhãn thực tế: {true_label.upper()}",
            fontsize=12,
            fontweight="bold",
            pad=12,
            loc="left"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n--> Biểu đồ trực quan hóa Attention đã được lưu thành công tại: {output_path}")

def run_attention_demo(project_dir=None):
    if project_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    models_dir = os.path.join(project_dir, "outputs", "models")
    vocab_path = os.path.join(models_dir, "vocab.json")
    model_path = os.path.join(models_dir, "bilstm_best.pt")
    
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại {model_path} hoặc từ điển {vocab_path}!")
        return
        
    # Nạp từ điển
    with open(vocab_path, "r", encoding="utf-8") as f:
        w2i = json.load(f)
    vocab = Vocabulary()
    vocab.w2i = w2i
    vocab.i2w = {v: k for k, v in w2i.items()}
    
    # Nạp mô hình
    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_classes=cfg.get("num_classes", 3),
        attention_dim=cfg.get("attention_dim", 64),
        pad_idx=vocab.w2i.get("<pad>", 0)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    preprocessor = VietnameseTextPreprocessor(segment=True)
    
    # Các câu phản hồi sinh viên tiêu biểu để minh họa Attention
    sample_feedbacks = [
        ("thầy dạy rất nhiệt tình , giải đáp thắc mắc cho sinh viên rất chu đáo", "positive"),
        ("slide bài giảng sơ sài khó hiểu , không có ví dụ thực tế gì cả", "negative"),
        ("giảng viên giảng hay nhưng bài tập về nhà quá nhiều và khó", "negative"),
        ("lớp học diễn ra đúng thời gian theo lịch quy định của trường", "neutral")
    ]
    
    out_fig = os.path.join(project_dir, "outputs", "figures", "attention_visualization.png")
    visualize_bilstm_attention(model, vocab, preprocessor, sample_feedbacks, output_path=out_fig)

if __name__ == "__main__":
    run_attention_demo()
