import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

from src.preprocessor import VietnameseTextPreprocessor
from src.dataset import Vocabulary, BiLSTMDataset, PhoBERTDataset, LABEL2ID, ID2LABEL
from src.models.bilstm_attention import BiLSTMAttention
from src.models.phobert_model import PhoBERTClassifier

def measure_inference_speed(model, sample_input, num_iterations=100, device="cpu"):
    """Đo độ trễ suy luận (ms/mẫu) trên thiết bị chỉ định."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if isinstance(sample_input, dict):
                _ = model(**{k: v.to(device) for k, v in sample_input.items()})
            else:
                _ = model(sample_input.to(device))
                
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            if isinstance(sample_input, dict):
                _ = model(**{k: v.to(device) for k, v in sample_input.items()})
            else:
                _ = model(sample_input.to(device))
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / num_iterations) * 1000
    return latency_ms

def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Nhãn dự đoán (Predicted)", fontsize=12)
    plt.ylabel("Nhãn thực tế (True)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_comparison_charts(metrics_df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Accuracy & F1-Macro
    ax1 = axes[0]
    x = np.arange(len(metrics_df))
    width = 0.35
    ax1.bar(x - width/2, metrics_df['Accuracy (%)'], width, label='Accuracy', color='#3498db')
    ax1.bar(x + width/2, metrics_df['Macro F1 (%)'], width, label='Macro F1', color='#2ecc71')
    ax1.set_title('So sánh Độ chính xác & F1-Score', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Mô hình'], fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for i, row in metrics_df.iterrows():
        ax1.text(i - width/2, row['Accuracy (%)'] + 1.5, f"{row['Accuracy (%)']:.1f}%", ha='center', fontweight='bold')
        ax1.text(i + width/2, row['Macro F1 (%)'] + 1.5, f"{row['Macro F1 (%)']:.1f}%", ha='center', fontweight='bold')
    ax1.legend(loc='lower right')
    
    # 2. Parameters & Model Size
    ax2 = axes[1]
    ax2.bar(x, metrics_df['Tham số (Triệu)'], width*1.2, color='#e67e22')
    ax2.set_title('Kích thước mô hình (Số tham số - Triệu)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df['Mô hình'], fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for i, row in metrics_df.iterrows():
        ax2.text(i, row['Tham số (Triệu)'] + 2, f"{row['Tham số (Triệu)']:.2f}M", ha='center', fontweight='bold')
        
    # 3. Inference Latency
    ax3 = axes[2]
    ax3.bar(x, metrics_df['Độ trễ suy luận (ms/mẫu)'], width*1.2, color='#e74c3c')
    ax3.set_title('Độ trễ suy luận CPU (ms/mẫu - càng thấp càng tốt)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_df['Mô hình'], fontsize=11)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for i, row in metrics_df.iterrows():
        ax3.text(i, row['Độ trễ suy luận (ms/mẫu)'] + 0.5, f"{row['Độ trễ suy luận (ms/mẫu)']:.2f} ms", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_evaluation_and_comparison(project_dir=None):
    if project_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    data_dir = os.path.join(project_dir, "data")
    outputs_dir = os.path.join(project_dir, "outputs")
    models_dir = os.path.join(outputs_dir, "models")
    figures_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  ĐÁNH GIÁ VÀ SO SÁNH TOÀN DIỆN BiLSTM VS PhoBERT")
    print(f"=======================================================\n")
    
    # 1. Đọc tập test
    test_csv = os.path.join(data_dir, "UIT-VSFC_test.csv")
    df_test = pd.read_csv(test_csv)
    print(f"Tập kiểm thử Test UIT-VSFC: {len(df_test)} mẫu")
    class_names = ["negative", "neutral", "positive"]
    
    preprocessor = VietnameseTextPreprocessor(segment=True)
    
    # ---------------------------------------------------------
    # 2. Đánh giá BiLSTM + Attention
    # ---------------------------------------------------------
    vocab_path = os.path.join(models_dir, "vocab.json")
    bilstm_path = os.path.join(models_dir, "bilstm_best.pt")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        w2i = json.load(f)
    vocab = Vocabulary()
    vocab.w2i = w2i
    vocab.i2w = {v: k for k, v in w2i.items()}
    
    bilstm_ckpt = torch.load(bilstm_path, map_location=device)
    cfg = bilstm_ckpt.get("config", {})
    bilstm_model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_classes=3,
        attention_dim=cfg.get("attention_dim", 64),
        pad_idx=vocab.w2i.get("<pad>", 0)
    ).to(device)
    bilstm_model.load_state_dict(bilstm_ckpt["model_state_dict"])
    bilstm_model.eval()
    
    bilstm_params = sum(p.numel() for p in bilstm_model.parameters())
    bilstm_size_mb = os.path.getsize(bilstm_path) / (1024 * 1024)
    
    # Predict BiLSTM
    bilstm_dataset = BiLSTMDataset(df_test, vocab, preprocessor, max_len=64)
    bilstm_loader = torch.utils.data.DataLoader(bilstm_dataset, batch_size=64, shuffle=False)
    
    bilstm_preds, true_labels = [], []
    with torch.no_grad():
        for batch in bilstm_loader:
            inp = batch["input_ids"].to(device)
            logits, _ = bilstm_model(inp)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            bilstm_preds.extend(preds)
            true_labels.extend(batch["label"].tolist())
            
    bilstm_acc = accuracy_score(true_labels, bilstm_preds)
    bilstm_f1 = f1_score(true_labels, bilstm_preds, average="macro", zero_division=0)
    bilstm_latency = measure_inference_speed(bilstm_model, torch.randint(0, len(vocab), (1, 64)), device=device)
    
    # ---------------------------------------------------------
    # 3. Đánh giá PhoBERT
    # ---------------------------------------------------------
    phobert_path = os.path.join(models_dir, "phobert_best.pt")
    phobert_ckpt = torch.load(phobert_path, map_location=device)
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = PhoBERTClassifier(model_name="vinai/phobert-base-v2", num_classes=3).to(device)
    phobert_model.load_state_dict(phobert_ckpt["model_state_dict"])
    phobert_model.eval()
    
    phobert_params = sum(p.numel() for p in phobert_model.parameters())
    phobert_size_mb = os.path.getsize(phobert_path) / (1024 * 1024)
    
    # Predict PhoBERT
    phobert_dataset = PhoBERTDataset(df_test, tokenizer, preprocessor, max_len=64)
    phobert_loader = torch.utils.data.DataLoader(phobert_dataset, batch_size=32, shuffle=False)
    
    phobert_preds = []
    with torch.no_grad():
        for batch in phobert_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits, _ = phobert_model(inp, mask)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            phobert_preds.extend(preds)
            
    phobert_acc = accuracy_score(true_labels, phobert_preds)
    phobert_f1 = f1_score(true_labels, phobert_preds, average="macro", zero_division=0)
    
    sample_phobert_inp = {
        "input_ids": torch.randint(0, 1000, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long)
    }
    phobert_latency = measure_inference_speed(phobert_model, sample_phobert_inp, device=device)
    
    # Đọc thời gian train từ file summary
    with open(os.path.join(models_dir, "bilstm_summary.json"), "r", encoding="utf-8") as f:
        bilstm_summary = json.load(f)
    with open(os.path.join(models_dir, "phobert_summary.json"), "r", encoding="utf-8") as f:
        phobert_summary = json.load(f)
        
    bilstm_train_time = bilstm_summary.get("total_training_time_sec", 0.0)
    phobert_train_time = phobert_summary.get("total_training_time_sec", 0.0)
    
    # ---------------------------------------------------------
    # 4. Vẽ Confusion Matrix
    # ---------------------------------------------------------
    cm_bilstm_path = os.path.join(figures_dir, "confusion_matrix_bilstm.png")
    cm_phobert_path = os.path.join(figures_dir, "confusion_matrix_phobert.png")
    
    plot_confusion_matrix(true_labels, bilstm_preds, class_names, "Ma trận nhầm lẫn - BiLSTM + Attention", cm_bilstm_path)
    plot_confusion_matrix(true_labels, phobert_preds, class_names, "Ma trận nhầm lẫn - PhoBERT Fine-tune", cm_phobert_path)
    print(f"Đã lưu biểu đồ ma trận nhầm lẫn tại:\n  - {cm_bilstm_path}\n  - {cm_phobert_path}")
    
    # ---------------------------------------------------------
    # 5. Lập bảng so sánh
    # ---------------------------------------------------------
    comparison_data = {
        "Mô hình": ["BiLSTM + Attention", "PhoBERT-base v2 (Fine-tune)"],
        "Accuracy (%)": [round(bilstm_acc * 100, 2), round(phobert_acc * 100, 2)],
        "Macro F1 (%)": [round(bilstm_f1 * 100, 2), round(phobert_f1 * 100, 2)],
        "Tham số (Triệu)": [round(bilstm_params / 1e6, 3), round(phobert_params / 1e6, 2)],
        "Kích thước file (MB)": [round(bilstm_size_mb, 2), round(phobert_size_mb, 2)],
        "Thời gian huấn luyện (giây)": [round(bilstm_train_time, 1), round(phobert_train_time, 1)],
        "Độ trễ suy luận (ms/mẫu)": [round(bilstm_latency, 2), round(phobert_latency, 2)],
        "Tốc độ suy luận (mẫu/giây)": [round(1000 / bilstm_latency, 1), round(1000 / phobert_latency, 1)]
    }
    
    metrics_df = pd.DataFrame(comparison_data)
    print("\n=== BẢNG SO SÁNH TỔNG HỢP HIỆU NĂNG VÀ TÀI NGUYÊN ===")
    print(metrics_df.to_string(index=False))
    
    # Vẽ biểu đồ so sánh
    compare_chart_path = os.path.join(figures_dir, "model_comparison_chart.png")
    plot_comparison_charts(metrics_df, compare_chart_path)
    print(f"\nĐã lưu biểu đồ so sánh tại: {compare_chart_path}")
    
    # Báo cáo chi tiết từng lớp
    print("\n--- Chi tiết phân loại BiLSTM + Attention ---")
    print(classification_report(true_labels, bilstm_preds, target_names=class_names, digits=4))
    print("\n--- Chi tiết phân loại PhoBERT Fine-tune ---")
    print(classification_report(true_labels, phobert_preds, target_names=class_names, digits=4))
    
    # Lưu file JSON
    results_json_path = os.path.join(outputs_dir, "benchmark_results.json")
    final_results = {
        "comparison_table": metrics_df.to_dict(orient="records"),
        "bilstm_classification_report": classification_report(true_labels, bilstm_preds, target_names=class_names, output_dict=True),
        "phobert_classification_report": classification_report(true_labels, phobert_preds, target_names=class_names, output_dict=True)
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả chi tiết tại: {results_json_path}")
    
    return metrics_df

if __name__ == "__main__":
    run_evaluation_and_comparison()
