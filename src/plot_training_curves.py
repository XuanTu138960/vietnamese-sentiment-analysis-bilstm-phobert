"""
Vẽ đồ thị hàm mất mát (Loss) và độ chính xác (Accuracy/F1) theo từng Epoch
phục vụ báo cáo học thuật và bài tập lớn.
"""
import os
import sys
import json
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

def plot_curves():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, "outputs", "models", "bilstm_summary.json")
    output_fig = os.path.join(base_dir, "outputs", "figures", "training_curves.png")

    if not os.path.exists(summary_path):
        print(f"Không tìm thấy file: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] * 100 for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]
    val_f1 = [h["val_f1"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị 1: Loss
    ax1.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.5, markersize=7, label='Train Loss')
    ax1.plot(epochs, val_loss, 's--', color='#3498db', linewidth=2.5, markersize=7, label='Val Loss')
    ax1.set_title('Đường Cong Hàm Mất Mát (Cross-Entropy Loss)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)

    # Đồ thị 2: Accuracy & F1
    ax2.plot(epochs, train_acc, 'o-', color='#2ecc71', linewidth=2.5, markersize=7, label='Train Accuracy (%)')
    ax2.plot(epochs, val_acc, 's--', color='#f39c12', linewidth=2.5, markersize=7, label='Val Accuracy (%)')
    ax2.plot(epochs, val_f1, '^:', color='#9b59b6', linewidth=2.5, markersize=7, label='Val Macro F1 (%)')
    ax2.set_title('Độ Chính Xác & Macro F1 Theo Epoch', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Tỷ lệ (%)', fontsize=11)
    ax2.set_xticks(epochs)
    ax2.set_ylim(50, 100)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ huấn luyện tại: {output_fig}")

if __name__ == "__main__":
    plot_curves()
