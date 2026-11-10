import json
import re
from pathlib import Path

import joblib


# 1. Cấu hình
BASE_DIR = Path(__file__).resolve().parent

MIN_CONFIDENCE = 0.30

DEFAULT_FALLBACK = (
    "Xin lỗi, tôi chưa xác định chắc chắn yêu cầu của bạn. "
    "Bạn vui lòng diễn đạt rõ hơn về dịch vụ ngân hàng cần hỗ trợ. "
    "Nếu liên quan đến mất thẻ, giao dịch bất thường hoặc lộ thông tin, "
    "hãy liên hệ ngay ngân hàng qua ứng dụng hoặc tổng đài chính thức. "
    "Không cung cấp mật khẩu, mã PIN hay OTP trong cuộc trò chuyện."
)


def load_json(relative_path):
    with (BASE_DIR / relative_path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


responses = load_json("data/responses.json")
rules = load_json("data/rules.json")

if not isinstance(responses, dict) or not isinstance(rules, list):
    raise ValueError("Kiểm tra cấu trúc responses.json (dict) và rules.json (list).")

MODEL_CANDIDATES = [
    BASE_DIR / "models" / "best_model.joblib",
    BASE_DIR / "Models" / "best_model.joblib",
    BASE_DIR / "best_model.joblib",
]
model_path = next((path for path in MODEL_CANDIDATES if path.is_file()), None)
if model_path is None:
    raise FileNotFoundError(
        "Không tìm thấy best_model.joblib. Hãy chạy train.py và kiểm tra "
        "thư mục models/, Models/ hoặc thư mục gốc dự án."
    )

model = joblib.load(model_path)

if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
    raise TypeError(
        "Mô hình cần hỗ trợ predict_proba() và classes_. "
        "Hãy lưu Pipeline đã huấn luyện từ train.py."
    )


# 2. Chuẩn hóa văn bản

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def contains_phrase(question, phrase):
    phrase = clean_text(phrase)
    if not phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, question) is not None


# 3. Hệ chuyên gia IF–THEN

def find_rule_intent(question):
    question = clean_text(question)
    if not question:
        return None

    sorted_rules = sorted(
        rules,
        key=lambda rule: rule.get("priority", 0),
        reverse=True,
    )

    for rule in sorted_rules:
        if any(contains_phrase(question, word) for word in rule.get("exclude", [])):
            continue

        if any(contains_phrase(question, word) for word in rule.get("keywords", [])):
            return rule["intent"]

    return None


# 4. Dự đoán: ưu tiên IF–THEN, sau đó dùng AI + fallback

def predict_intent(question):
    question = clean_text(question)
    if not question:
        return "fallback", "Không có"

    rule_intent = find_rule_intent(question)
    if rule_intent is not None:
        return rule_intent, "IF-THEN"

    probabilities = model.predict_proba([question])[0]
    best_index = int(probabilities.argmax())
    intent = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    if confidence < MIN_CONFIDENCE or intent not in responses:
        return "fallback", "Machine Learning (fallback)"

    return intent, "Machine Learning"


# 5. Chọn câu trả lời và trả kết quả cho giao diện

def get_response(intent):
    fallback = responses.get("fallback") or DEFAULT_FALLBACK
    answer = responses.get(intent) or fallback
    return answer if isinstance(answer, str) else str(fallback)


def chatbot(question):
    question = clean_text(question)
    if not question:
        return get_response("fallback"), "fallback", "Không có"

    intent, method = predict_intent(question)
    answer = get_response(intent)
    return answer, intent, method


# 6. Chạy trên Terminal (không chạy khi import từ app.py/notebook)
if __name__ == "__main__":
    print("=== CHATBOT HỖ TRỢ KHÁCH HÀNG NGÂN HÀNG ===")
    print("Đang dùng mô hình:", model_path)
    print(f"Ngưỡng thử nghiệm fallback: {MIN_CONFIDENCE:.2f}")
    print("Gõ 'thoat' để kết thúc. Không nhập OTP/PIN/mật khẩu thật.")

    while True:
        try:
            question = input("\nBạn: ")
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Tạm biệt!")
            break

        if clean_text(question) in {"thoat", "thoát"}:
            print("Chatbot: Tạm biệt!")
            break

        answer, intent, method = chatbot(question)
        print("Chatbot:", answer)
        print("Intent:", intent)
        print("Phương pháp:", method)
