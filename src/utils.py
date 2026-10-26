"""
Các hàm tiện ích bổ trợ cho dự án:
- Cố định seed cho tính tái lập thực nghiệm (Reproducibility)
- Tính toán trọng số lớp (Class Weights) cho bài toán mất cân bằng dữ liệu
- Định dạng hiển thị kết quả và log
"""
import os
import random
import logging
import torch
import numpy as np

def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên cho tất cả các thư viện để đảm bảo tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_class_weights(labels, num_classes=3, device="cpu"):
    """
    Tính trọng số cân bằng lớp (Inverse Class Frequency):
    weight[c] = total_samples / (num_classes * count[c])
    Giúp cải thiện độ nhạy với các lớp thiểu số (như lớp neutral trong UIT-VSFC).
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1).astype(float))
    # Chuẩn hóa về trung bình 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def get_logger(name: str = "SentimentAnalysis"):
    """Cấu hình logger tiêu chuẩn."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
