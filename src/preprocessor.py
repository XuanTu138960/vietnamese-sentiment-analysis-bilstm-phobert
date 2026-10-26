import os
import sys
import re
import unicodedata
from pyvi import ViTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Bảng chuẩn hóa dấu tiếng Việt (chuẩn hóa về kiểu mới - chuẩn Bộ GD&ĐT)
# Ví dụ: hoà -> hòa, thuỷ -> thủy, oà -> òa
BANG_NGUYEN_AM = [
    ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
    ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
    ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
    ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
    ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
    ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
    ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
    ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
    ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
    ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
    ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
    ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
]

# Từ điển viết tắt / teencode thông dụng trong phản hồi sinh viên
TEENCODE_DICT = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "khg": "không",
    "kg": "không",
    "hok": "không",
    "hong": "không",
    "hông": "không",
    "dc": "được",
    "đc": "được",
    "dk": "được",
    "đk": "được",
    "sv": "sinh viên",
    "gv": "giảng viên",
    "gvcn": "giảng viên chủ nhiệm",
    "thầy": "thầy",
    "cô": "cô",
    "bt": "bài tập",
    "btvn": "bài tập",
    "th": "thực hành",
    "lt": "lý thuyết",
    "vs": "với",
    "wa": "quá",
    "qua": "quá",
    "rat": "rất",
    "nhieu": "nhiều",
    "nhìu": "nhiều",
    "hieu": "hiểu",
    "kho": "khó",
    "de": "dễ",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "thx": "cảm ơn",
    "ok": "tốt",
    "oke": "tốt",
    "okie": "tốt",
    "cx": "cũng",
    "j": "gì",
    "thik": "thích",
    "thix": "thích",
    "tl": "trả lời",
    "rep": "trả lời",
    "fb": "phản hồi",
    "sl": "số lượng",
    "pt": "phương pháp",
    "pp": "phương pháp",
    "mon": "môn",
    "hoc": "học",
    "thi": "thi",
    "tot": "tốt",
    "te": "tệ",
    "chan": "chán"
}

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa bảng mã Unicode về dạng NFC."""
    return unicodedata.normalize('NFC', text)

def remove_repeated_characters(text: str) -> str:
    """
    Chuẩn hóa các ký tự lặp kéo dài (ví dụ: 'quáaaa' -> 'quá', 'hayyy' -> 'hay').
    Giữ lại tối đa 1 ký tự liên tiếp lặp.
    """
    return re.sub(r'([a-zA-ZÀ-ỹ])\1{2,}', r'\1', text)

def replace_teencode(text: str) -> str:
    """Thay thế các từ viết tắt / teencode bằng từ ngữ chuẩn."""
    words = text.split()
    normalized_words = [TEENCODE_DICT.get(w.lower(), w) for w in words]
    return " ".join(normalized_words)

def clean_text(text: str) -> str:
    """Làm sạch ký tự đặc biệt, dấu câu thừa, chuẩn hóa khoảng trắng."""
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode NFC
    text = normalize_unicode(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Thay thế emoji/ký tự đặc biệt thành dấu cách
    text = re.sub(r'[^\w\s\.,!?_-]', ' ', text)
    
    # 4. Chuẩn hóa dấu câu (cách dấu chấm, phẩy)
    text = re.sub(r'([.,!?])', r' \1 ', text)
    
    # 5. Ký tự kéo dài
    text = remove_repeated_characters(text)
    
    # 6. Teencode
    text = replace_teencode(text)
    
    # 7. Khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_words(text: str) -> str:
    """
    Tách từ tiếng Việt sử dụng PyVi.
    Các từ ghép sẽ được nối bằng dấu gạch dưới '_'
    Ví dụ: 'sinh viên rất hài lòng' -> 'sinh_viên rất hài_lòng'
    """
    cleaned = clean_text(text)
    segmented = ViTokenizer.tokenize(cleaned)
    return segmented

class VietnameseTextPreprocessor:
    """Lớp tiền xử lý toàn diện cho tiếng Việt phục vụ BiLSTM và PhoBERT."""
    def __init__(self, segment: bool = True):
        self.segment = segment

    def preprocess(self, text: str) -> str:
        if self.segment:
            return segment_words(text)
        return clean_text(text)

    def preprocess_batch(self, texts):
        return [self.preprocess(t) for t in texts]

if __name__ == "__main__":
    test_samples = [
        "thầy dạy rất nhiệttt tìnhhhhh, e cảm ơn thầy nhiềuuuu <3",
        "slide bài giảng khó hiểu wa, ko có ví dụ thực tế j cả!!",
        "môn này rất ok, gv giải đáp thắc mắc cho sv rất nhanh.",
        "chưa áp dụng công nghệ thông tin và các thiết bị hỗ trợ"
    ]
    preprocessor = VietnameseTextPreprocessor(segment=True)
    print("--- KIỂM TRA TIỀN XỬ LÝ TIẾNG VIỆT ---")
    for s in test_samples:
        print(f"Gốc : {s}")
        print(f"Xử lý: {preprocessor.preprocess(s)}\n")
