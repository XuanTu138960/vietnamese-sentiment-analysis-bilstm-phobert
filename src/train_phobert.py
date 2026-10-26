import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.preprocessor import VietnameseTextPreprocessor
from src.dataset import PhoBERTDataset, LABEL2ID, ID2LABEL
from src.models.phobert_model import PhoBERTClassifier

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
        
        if (step + 1) % 25 == 0 or (step + 1) == len(dataloader):
            current_loss = total_loss / len(all_labels)
            current_acc = accuracy_score(all_labels, all_preds)
            print(f"  Step {step + 1:03d}/{len(dataloader):03d} - Loss: {current_loss:.4f} | Acc: {current_acc*100:.2f}%")
        
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
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro, all_preds, all_labels

def run_train_phobert(
    data_dir: str = "data",
    output_dir: str = "outputs",
    model_name: str = "vinai/phobert-base-v2",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 64,
    max_train_samples: int = 3000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== HUẤN LUYỆN / FINE-TUNE PhoBERT ({model_name}) ===")
    print(f"Device: {device}")
    
    # 1. Tải tokenizer & preprocessor
    print("Đang nạp PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocessor = VietnameseTextPreprocessor(segment=True)
    
    # 2. Đọc dữ liệu
    df_train = pd.read_csv(os.path.join(data_dir, "UIT-VSFC_train.csv")).dropna(subset=['text', 'label'])
    df_valid = pd.read_csv(os.path.join(data_dir, "UIT-VSFC_valid.csv")).dropna(subset=['text', 'label'])
    df_test = pd.read_csv(os.path.join(data_dir, "UIT-VSFC_test.csv")).dropna(subset=['text', 'label'])
    
    if max_train_samples and max_train_samples < len(df_train):
        print(f"Chọn lọc tập huấn luyện đại diện (stratified sample): {max_train_samples}/{len(df_train)} mẫu...")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_train_samples, random_state=42)
        idx_train, _ = next(sss.split(df_train['text'], df_train['label']))
        df_train = df_train.iloc[idx_train].reset_index(drop=True)
        print("Phân bố nhãn tập train sau chọn mẫu:")
        print(df_train['label'].value_counts().to_string())
        
    # Tạo Dataset
    train_dataset = PhoBERTDataset(df_train, tokenizer, preprocessor, max_len=max_len)
    valid_dataset = PhoBERTDataset(df_valid, tokenizer, preprocessor, max_len=max_len)
    test_dataset = PhoBERTDataset(df_test, tokenizer, preprocessor, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Khởi tạo mô hình PhoBERT
    print("Khởi tạo PhoBERT Classifier...")
    model = PhoBERTClassifier(model_name=model_name, num_classes=3, dropout=0.2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PhoBERT Params: Tổng {total_params:,} | Cần huấn luyện: {trainable_params:,}")
    
    # 4. Optimizer & Warmup Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_val_f1 = 0.0
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "phobert_best.pt")
    
    history = []
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Bắt đầu Epoch {epoch}/{epochs} ---")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, valid_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.2f}s] "
              f"- Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | F1: {train_f1*100:.2f}% "
              f"- Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': {
                    'model_name': model_name,
                    'num_classes': 3,
                    'max_len': max_len
                }
            }, best_model_path)
            print(f"  --> Đã lưu checkpoint PhoBERT tốt nhất tại: {best_model_path} (Val F1: {val_f1*100:.2f}%)")
            
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
    print(f"\nHuấn luyện PhoBERT hoàn thành sau: {total_training_time:.2f}s (~{total_training_time/60:.2f} phút)")
    
    # 5. Đánh giá trên tập Test
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ PhoBERT TRÊN TẬP TEST ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1-macro: {test_f1*100:.2f}%")
    print("\nChi tiết phân loại:")
    print(classification_report(test_labels, test_preds, target_names=["negative", "neutral", "positive"], digits=4))
    
    summary = {
        "model": "PhoBERT (vinai/phobert-base-v2)",
        "total_params": total_params,
        "total_training_time_sec": total_training_time,
        "avg_epoch_time_sec": total_training_time / epochs,
        "test_acc": test_acc,
        "test_f1_macro": test_f1,
        "history": history
    }
    with open(os.path.join(models_dir, "phobert_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    return summary

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(project_dir, "outputs")
    run_train_phobert(data_dir=data_dir, output_dir=output_dir, epochs=2, batch_size=32, max_train_samples=2500)
