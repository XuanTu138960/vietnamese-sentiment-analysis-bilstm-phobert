"""
Ứng dụng Web tương tác (Flask Web App) Phân tích Cảm xúc Tiếng Việt.
Cho phép người dùng nhập trực tiếp nhận xét sinh viên, hiển thị:
  - Dự đoán cảm xúc từ cả BiLSTM và PhoBERT
  - Trọng số và Heatmap Attention tương tác theo thời gian thực (hover xem %)
  - Phân bổ xác suất và độ tin cậy

Khởi chạy:
  python app.py
Truy cập:
  http://localhost:5000
"""
import os
import sys
from flask import Flask, render_template, request, jsonify

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from predict import SentimentPredictor

app = Flask(__name__)

# Khởi tạo mô hình một lần khi server start
print("Đang khởi tạo các mô hình suy luận...")
predictor = SentimentPredictor()
print("Mô hình đã sẵn sàng phục vụ Web App!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Vui lòng cung cấp văn bản cần phân tích'}), 400

    result = predictor.predict_both(text)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" Web Demo đang chạy tại: http://localhost:{port}")
    print(f" Mở trình duyệt và trải nghiệm giao diện phân tích!")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
