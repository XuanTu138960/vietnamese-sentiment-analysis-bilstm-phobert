import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Đảm bảo đường dẫn project nằm trong sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data.download_data import download_uit_vsfc
from src.train_bilstm import run_train_bilstm
from src.train_phobert import run_train_phobert
from src.visualize_attention import run_attention_demo
from src.evaluate_and_compare import run_evaluation_and_comparison

def main():
    total_start = time.time()
    print("=" * 70)
    print("  DỰ ÁN PHÂN TÍCH CẢM XÚC TIẾNG VIỆT TRÊN PHẢN HỒI SINH VIÊN (UIT-VSFC)")
    print("  SO SÁNH: BiLSTM + ATTENTION vs PhoBERT (vinai/phobert-base-v2)")
    print("=" * 70)
    
    # 1. Tải và kiểm tra dữ liệu
    print("\n[BƯỚC 1/5] Tải và chuẩn bị tập dữ liệu UIT-VSFC...")
    download_uit_vsfc()
    
    # 2. Huấn luyện BiLSTM + Attention
    print("\n[BƯỚC 2/5] Huấn luyện mô hình BiLSTM + Attention từ đầu...")
    data_dir = os.path.join(PROJECT_DIR, "data")
    outputs_dir = os.path.join(PROJECT_DIR, "outputs")
    run_train_bilstm(data_dir=data_dir, output_dir=outputs_dir, epochs=6, batch_size=64)
    
    # 3. Fine-tune PhoBERT
    print("\n[BƯỚC 3/5] Fine-tune mô hình tiền huấn luyện PhoBERT...")
    run_train_phobert(
        data_dir=data_dir,
        output_dir=outputs_dir,
        model_name="vinai/phobert-base-v2",
        epochs=2,
        batch_size=32,
        lr=2e-5,
        max_train_samples=2500
    )
    
    # 4. Trực quan hóa trọng số Attention
    print("\n[BƯỚC 4/5] Trích xuất và trực quan hóa trọng số Attention...")
    run_attention_demo(project_dir=PROJECT_DIR)
    
    # 5. Đánh giá và so sánh toàn diện
    print("\n[BƯỚC 5/5] Đánh giá kiểm thử, vẽ Confusion Matrix và lập bảng so sánh...")
    run_evaluation_and_comparison(project_dir=PROJECT_DIR)
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH TRONG: {total_elapsed:.1f}s ({total_elapsed/60:.2f} phút)")
    print(f"  Kết quả và biểu đồ được lưu tại: {outputs_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
