# Hướng Dẫn Đóng Góp (Contributing Guidelines)

Cảm ơn bạn đã quan tâm đến dự án **Phân Tích Cảm Xúc Tiếng Việt: BiLSTM + Attention vs PhoBERT**!

## 1. Quy Trình Đóng Góp (Workflow)

1. **Fork** repository này về tài khoản của bạn.
2. Tạo một nhánh mới (branch) cho tính năng hoặc sửa lỗi của bạn:
   ```bash
   git checkout -b feature/ten-tinh-nang-moi
   ```
3. Thực hiện thay đổi và kiểm tra cẩn thận:
   - Đảm bảo mã nguồn tuân thủ PEP 8.
   - Thêm chú thích bằng tiếng Việt hoặc tiếng Anh rõ ràng, dễ hiểu.
4. Commit thay đổi:
   ```bash
   git commit -m "feat: thêm tính năng X cho mô hình Y"
   ```
5. Push nhánh lên GitHub và tạo **Pull Request (PR)**:
   ```bash
   git push origin feature/ten-tinh-nang-moi
   ```

## 2. Hướng Mở Rộng Tiềm Năng

- Thử nghiệm các mô hình ngôn ngữ mới: **ViSoBERT (EMNLP 2023)**, **ViDeBERTa (EACL 2023)**.
- Tích hợp thêm hàm mất mát **Focal Loss** để cải thiện độ nhận diện lớp thiểu số (`neutral`).
- Tối ưu hóa mô hình với **ONNX Runtime** hoặc **TorchScript** để tăng tốc độ suy luận.
