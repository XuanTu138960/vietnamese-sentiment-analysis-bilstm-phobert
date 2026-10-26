"""
Công cụ dòng lệnh (CLI Inference Tool) dự đoán cảm xúc trên câu bất kỳ.
So sánh trực tiếp kết quả dự đoán giữa BiLSTM + Attention và PhoBERT.

Cách dùng:
  python predict.py --text "Thầy dạy rất nhiệt tình và dễ hiểu"
  python predict.py --interactive
"""
import os
import sys
import json
import argparse
import torch
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer
from config import BASE_DIR, MODELS_DIR, ID2LABEL, LABEL_NAMES_VI, DEVICE
from src.preprocessor import VietnameseTextPreprocessor
from src.dataset import Vocabulary
from src.models.bilstm_attention import BiLSTMAttention
from src.models.phobert_model import PhoBERTClassifier

class SentimentPredictor:
    """Bộ suy luận tích hợp cả BiLSTM+Attention và PhoBERT."""
    def __init__(self, device=DEVICE):
        self.device = device
        self.preprocessor = VietnameseTextPreprocessor(segment=True)
        self._load_bilstm()
        self._load_phobert()

    def _load_bilstm(self):
        vocab_path = os.path.join(MODELS_DIR, "vocab.json")
        model_path = os.path.join(MODELS_DIR, "bilstm_best.pt")
        if not os.path.exists(vocab_path) or not os.path.exists(model_path):
            self.bilstm_model = None
            return

        with open(vocab_path, "r", encoding="utf-8") as f:
            w2i = json.load(f)
        self.vocab = Vocabulary()
        self.vocab.w2i = w2i
        self.vocab.i2w = {v: k for k, v in w2i.items()}

        ckpt = torch.load(model_path, map_location=self.device)
        cfg = ckpt.get("config", {})
        self.bilstm_model = BiLSTMAttention(
            vocab_size=len(self.vocab),
            embed_dim=cfg.get("embed_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_classes=3,
            attention_dim=cfg.get("attention_dim", 64),
            pad_idx=self.vocab.w2i.get("<pad>", 0)
        ).to(self.device)
        self.bilstm_model.load_state_dict(ckpt["model_state_dict"])
        self.bilstm_model.eval()

    def _load_phobert(self):
        phobert_path = os.path.join(MODELS_DIR, "phobert_best.pt")
        if not os.path.exists(phobert_path):
            self.phobert_model = None
            return

        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.phobert_model = PhoBERTClassifier(model_name="vinai/phobert-base-v2", num_classes=3).to(self.device)
        ckpt = torch.load(phobert_path, map_location=self.device)
        self.phobert_model.load_state_dict(ckpt["model_state_dict"])
        self.phobert_model.eval()

    def predict_bilstm(self, text: str):
        if self.bilstm_model is None:
            return None
        processed = self.preprocessor.preprocess(text)
        tokens = processed.split()
        indices = [self.vocab.w2i.get(t, self.vocab.w2i["<unk>"]) for t in tokens]
        if not indices:
            indices = [self.vocab.w2i["<unk>"]]
            tokens = ["<unk>"]

        inp = torch.tensor([indices], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, attn = self.bilstm_model(inp)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            attn_weights = attn[0].cpu().numpy()

        # Ghép từ với trọng số attention
        token_attn = list(zip(tokens, [float(w) for w in attn_weights[:len(tokens)]]))
        token_attn_sorted = sorted(token_attn, key=lambda x: x[1], reverse=True)

        return {
            "processed_text": processed,
            "tokens": tokens,
            "prediction": ID2LABEL[pred_id],
            "prediction_vi": LABEL_NAMES_VI[ID2LABEL[pred_id]],
            "confidence": float(probs[pred_id]),
            "probabilities": {ID2LABEL[i]: float(probs[i]) for i in range(3)},
            "token_attention": token_attn,
            "top_attended_words": token_attn_sorted[:3]
        }

    def predict_phobert(self, text: str):
        if self.phobert_model is None:
            return None
        processed = self.preprocessor.preprocess(text)
        encoding = self.tokenizer(processed, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        inp = encoding["input_ids"].to(self.device)
        mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, _ = self.phobert_model(inp, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))

        return {
            "processed_text": processed,
            "prediction": ID2LABEL[pred_id],
            "prediction_vi": LABEL_NAMES_VI[ID2LABEL[pred_id]],
            "confidence": float(probs[pred_id]),
            "probabilities": {ID2LABEL[i]: float(probs[i]) for i in range(3)}
        }

    def predict_both(self, text: str):
        return {
            "raw_text": text,
            "bilstm": self.predict_bilstm(text),
            "phobert": self.predict_phobert(text)
        }

def print_result(res):
    print("\n" + "="*65)
    print(f"Câu gốc: \"{res['raw_text']}\"")
    print("="*65)

    if res.get("bilstm"):
        b = res["bilstm"]
        print(f"\n[1] MÔ HÌNH BiLSTM + ATTENTION:")
        print(f"  - Tách từ       : {b['processed_text']}")
        print(f"  - Kết quả       : {b['prediction_vi'].upper()} (Độ tin cậy: {b['confidence']*100:.2f}%)")
        print(f"  - Xác suất chi tiết:")
        for lbl, prob in b["probabilities"].items():
            bar = "#" * int(prob * 20)
            print(f"      {LABEL_NAMES_VI[lbl]:<12}: {prob*100:5.1f}% |{bar:<20}|")
        print(f"  - Từ ngữ được chú ý nhiều nhất (Top Attention):")
        for word, weight in b["top_attended_words"]:
            print(f"      * '{word}': {weight*100:.1f}%")

    if res.get("phobert"):
        p = res["phobert"]
        print(f"\n[2] MÔ HÌNH PhoBERT (vinai/phobert-base-v2):")
        print(f"  - Kết quả       : {p['prediction_vi'].upper()} (Độ tin cậy: {p['confidence']*100:.2f}%)")
        print(f"  - Xác suất chi tiết:")
        for lbl, prob in p["probabilities"].items():
            bar = "#" * int(prob * 20)
            print(f"      {LABEL_NAMES_VI[lbl]:<12}: {prob*100:5.1f}% |{bar:<20}|")

    print("="*65)

def main():
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc văn bản phản hồi sinh viên (BiLSTM & PhoBERT)")
    parser.add_argument("--text", type=str, help="Câu phản hồi cần phân tích cảm xúc")
    parser.add_argument("--interactive", action="store_true", help="Chế độ nhập tương tác liên tục từ bàn phím")
    args = parser.parse_args()

    predictor = SentimentPredictor()

    if args.interactive:
        print("\n=== CHẾ ĐỘ PHÂN TÍCH CẢM XÚC TƯƠNG TÁC (Gõ 'exit' hoặc 'quit' để thoát) ===")
        while True:
            try:
                user_input = input("\nNhập phản hồi sinh viên > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Tạm biệt!")
                    break
                res = predictor.predict_both(user_input)
                print_result(res)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.text:
        res = predictor.predict_both(args.text)
        print_result(res)
    else:
        # Ví dụ mẫu mặc định
        demo_sentences = [
            "thầy dạy rất nhiệt tình , giải đáp thắc mắc cho sinh viên rất chu đáo",
            "slide bài giảng khó hiểu quá , không có ví dụ thực tế gì cả",
            "giảng viên dạy hay nhưng bài tập về nhà quá nhiều và khó",
            "lớp học diễn ra đúng thời gian theo lịch của nhà trường"
        ]
        print("\n--- CHẠY THỬ NGHIỆM TRÊN CÁC CÂU MẪU ĐIỂN HÌNH ---")
        for s in demo_sentences:
            res = predictor.predict_both(s)
            print_result(res)

if __name__ == "__main__":
    main()
