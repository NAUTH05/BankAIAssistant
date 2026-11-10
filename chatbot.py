import json
import joblib

with open("data/responses.json", "r", encoding="utf-8-sig") as file:
    responses = json.load(file)

with open("data/rules.json", "r", encoding="utf-8-sig") as file:
    rules = json.load(file)

model = joblib.load("models/best_model.joblib")

def clean_text(text):
    # Thực hiện các bước tiền xử lý văn bản như loại bỏ ký tự đặc biệt, chuyển về chữ thường, v.v.
    text = text.lower()
    # Thêm các bước tiền xử lý khác nếu cần
    text = " ".join(text.split())
    return text


def find_rule_intent(question):

    # Sắp xếp các luật theo độ ưu tiên
    sorted_rules = sorted(
        rules,
        key=lambda rule: rule.get("priority", 0),
        reverse=True
    )

    for rule in sorted_rules:

        # Kiểm tra điều kiện loại trừ
        excluded = False

        for word in rule.get("exclude", []):
            if clean_text(word) in question:
                excluded = True
                break

        if excluded:
            continue

        # Kiểm tra từ khóa của luật
        for keyword in rule["keywords"]:
            if clean_text(keyword) in question:
                return rule["intent"]

    return None



def predict_intent(question):

    question = clean_text(question)

    # Ưu tiên kiểm tra hệ luật
    intent = find_rule_intent(question)

    if intent is not None:
        return intent, "IF-THEN"

    # Nếu không có luật, dùng Machine Learning
    intent = model.predict([question])[0]

    return intent, "Machine Learning"

def get_response(intent):
    return responses.get(
        intent,
        responses.get("fallback")
    )


def chatbot(question):

    question = clean_text(question)

    # Không xử lý câu hỏi rỗng
    if not question:
        return responses["fallback"], "fallback", "Không có"

    # Dự đoán intent
    intent, method = predict_intent(question)

    # Lấy câu trả lời
    answer = get_response(intent)

    return answer, intent, method


if __name__ == "__main__":

    print("=== CHATBOT NGÂN HÀNG ===")
    print("Nhập 'thoat' để kết thúc.")

    while True:
        question = input("\nBạn: ")

        if question.lower().strip() == "thoat":
            print("Chatbot: Tạm biệt!")
            break

        answer, intent, method = chatbot(question)

        print("Chatbot:", answer)
        print("Intent:", intent)
        print("Phương pháp:", method)

