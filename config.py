"""
Cấu hình tập trung cho toàn bộ dự án Phân tích cảm xúc Tiếng Việt.
Bao gồm đường dẫn dữ liệu, siêu tham số huấn luyện và cấu hình mô hình.
"""
import os
import torch

# Đường dẫn thư mục gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "UIT-VSFC_train.csv")
VALID_PATH = os.path.join(DATA_DIR, "UIT-VSFC_valid.csv")
TEST_PATH = os.path.join(DATA_DIR, "UIT-VSFC_test.csv")

# Đường dẫn đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Ánh xạ nhãn cảm xúc
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES_VI = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực"
}

# Cấu hình thiết bị tính toán
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số cho BiLSTM + Attention
BILSTM_CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "attention_dim": 64,
    "dropout": 0.3,
    "max_len": 64,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 6,
    "weight_decay": 1e-5
}

# Siêu tham số cho PhoBERT
PHOBERT_CONFIG = {
    "model_name": "vinai/phobert-base-v2",
    "max_len": 64,
    "batch_size": 32,
    "lr": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.2
}
