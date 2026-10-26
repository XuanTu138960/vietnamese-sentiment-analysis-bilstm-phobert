import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.preprocessor import VietnameseTextPreprocessor
from src.dataset import Vocabulary, BiLSTMDataset, LABEL2ID, ID2LABEL
from src.models.bilstm_attention import BiLSTMAttention

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
        
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            logits, _ = model(input_ids)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro, all_preds, all_labels

def run_train_bilstm(
    data_dir: str = "data",
    output_dir: str = "outputs",
    epochs: int = 8,
    batch_size: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    attention_dim: int = 64,
    lr: float = 1e-3,
    max_len: int = 64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== HUẤN LUYỆN BiLSTM + ATTENTION (Device: {device}) ===")
    
    # 1. Tải dữ liệu
    train_path = os.path.join(data_dir, "UIT-VSFC_train.csv")
    valid_path = os.path.join(data_dir, "UIT-VSFC_valid.csv")
    test_path = os.path.join(data_dir, "UIT-VSFC_test.csv")
    
    df_train = pd.read_csv(train_path).dropna(subset=['text', 'label'])
    df_valid = pd.read_csv(valid_path).dropna(subset=['text', 'label'])
    df_test = pd.read_csv(test_path).dropna(subset=['text', 'label'])
    
    # 2. Tiền xử lý & Xây dựng từ điển
    preprocessor = VietnameseTextPreprocessor(segment=True)
    vocab = Vocabulary(min_freq=2)
    
    print("Đang tiền xử lý văn bản tập huấn luyện...")
    train_processed = [preprocessor.preprocess(t) for t in df_train['text'].astype(str)]
    vocab.build_vocab(train_processed)
    
    # Lưu từ điển
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    vocab_path = os.path.join(models_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.w2i, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu từ điển tại: {vocab_path}")
    
    # 3. Tạo DataLoader
    train_dataset = BiLSTMDataset(df_train, vocab, preprocessor, max_len=max_len)
    valid_dataset = BiLSTMDataset(df_valid, vocab, preprocessor, max_len=max_len)
    test_dataset = BiLSTMDataset(df_test, vocab, preprocessor, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Khởi tạo mô hình
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
        num_layers=2,
        attention_dim=attention_dim,
        dropout=0.3,
        pad_idx=vocab.w2i["<pad>"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Số lượng tham số mô hình BiLSTM+Attention: {total_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(models_dir, "bilstm_best.pt")
    history = []
    
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, valid_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.2f}s] "
              f"- Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | F1: {train_f1*100:.2f}% "
              f"- Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': {
                    'vocab_size': len(vocab),
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'attention_dim': attention_dim,
                    'num_classes': 3,
                    'max_len': max_len
                }
            }, best_model_path)
            print(f"  --> Đã lưu checkpoint tốt nhất tại: {best_model_path} (Val F1: {val_f1*100:.2f}%)")
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "epoch_time": epoch_time
        })
        
    total_training_time = time.time() - start_total_time
    print(f"\nHuấn luyện BiLSTM hoàn thành sau: {total_training_time:.2f}s (~{total_training_time/60:.2f} phút)")
    
    # 5. Đánh giá trên tập Test
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ BiLSTM+ATTENTION TRÊN TẬP TEST ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1-macro: {test_f1*100:.2f}%")
    print("\nChi tiết phân loại:")
    print(classification_report(test_labels, test_preds, target_names=["negative", "neutral", "positive"], digits=4))
    
    # Lưu kết quả
    summary = {
        "model": "BiLSTM+Attention",
        "total_params": total_params,
        "total_training_time_sec": total_training_time,
        "avg_epoch_time_sec": total_training_time / epochs,
        "test_acc": test_acc,
        "test_f1_macro": test_f1,
        "history": history
    }
    with open(os.path.join(models_dir, "bilstm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    return summary

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(project_dir, "outputs")
    run_train_bilstm(data_dir=data_dir, output_dir=output_dir, epochs=6)
