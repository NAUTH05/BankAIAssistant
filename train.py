
# DT51 - Huấn luyện và so sánh mô hình phân loại câu hỏi ngân hàng

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Tham số cá nhân
SEED = 45116
DATA_RATIO = 0.95    # Sử dụng 95% dữ liệu gốc
TEST_SIZE = 0.30     # Dành 30% dữ liệu đã lấy mẫu làm Test
CV_FOLDS = 7         # Số fold dùng để đánh giá


# 2. Đọc và làm sạch dữ liệu
df = pd.read_csv("data/questions.csv", encoding="utf-8-sig")

print("Số câu hỏi ban đầu:", len(df))
print("Dữ liệu thiếu:")
print(df[["question", "intent"]].isnull().sum())

# Xóa dòng thiếu thông tin
df = df.dropna(subset=["question", "intent"]).copy()

# Chuẩn hóa câu hỏi về chữ thường và khoảng trắng
df["question"] = df["question"].str.lower().str.strip()
df["question"] = df["question"].str.replace(r"\s+", " ", regex=True)
df["intent"] = df["intent"].str.strip()

# Loại bỏ câu hỏi hoặc intent rỗng
df = df[(df["question"] != "") & (df["intent"] != "")].copy()

# Kiểm tra câu hỏi giống nhau nhưng có intent khác nhau
conflict = df.groupby("question")["intent"].nunique()

if (conflict > 1).any():
    raise ValueError("Có câu hỏi trùng nhưng khác intent. Hãy kiểm tra questions.csv.")

# Xóa câu hỏi trùng sau khi kiểm tra nhãn
print("Số câu hỏi trùng:", df["question"].duplicated().sum())
df = df.drop_duplicates(subset=["question"])

print("Số câu sau làm sạch:", len(df))
print("Số nhóm intent:", df["intent"].nunique())


# 3. Lấy mẫu và chia dữ liệu Train/Test
df_sample = df.sample(
    frac=DATA_RATIO,
    random_state=SEED
)

train_df, test_df = train_test_split(
    df_sample,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df_sample["intent"]
)

print("Số câu được lấy mẫu:", len(df_sample))
print("Số câu Train:", len(train_df))
print("Số câu Test:", len(test_df))

# Mỗi intent cần ít nhất 7 mẫu trong tập Train để chia 7 fold
if train_df["intent"].value_counts().min() < CV_FOLDS:
    raise ValueError("Có nhóm intent có ít hơn 7 câu trong Train. Hãy bổ sung dữ liệu.")


# 4. Khai báo các mô hình học máy
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=SEED
    )
}


# 5. Đánh giá từng mô hình bằng Cross-validation
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=SEED
)

results = []

for name, algorithm in models.items():

    # Kết hợp TF-IDF và mô hình phân loại
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", algorithm)
    ])

    # Đánh giá bằng 7-fold Cross-validation
    scores = cross_val_score(
        pipe,
        train_df["question"],
        train_df["intent"],
        cv=cv,
        scoring="f1_macro"
    )

    # Lưu kết quả của mô hình vào danh sách
    results.append({
        "model": name,
        "f1_mean": scores.mean(),
        "f1_std": scores.std()
    })

    print(f"{name}: CV Macro F1 = {scores.mean():.4f} +/- {scores.std():.4f}")


# 6. Tạo bảng so sánh và chọn mô hình tốt nhất
bang_ket_qua = pd.DataFrame(results).sort_values(
    by="f1_mean",
    ascending=False
).reset_index(drop=True)

print("\nBẢNG SO SÁNH:")
print(bang_ket_qua.to_string(index=False))

# Lưu bảng để sử dụng trong báo cáo
bang_ket_qua.to_csv(
    "CSV/model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# Lấy tên mô hình có Macro F1 trung bình cao nhất
best_name = bang_ket_qua.loc[0, "model"]

print("\nMô hình được chọn:", best_name)


# 7. Huấn luyện lại mô hình được chọn trên toàn bộ Train
best_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", models[best_name])
])

best_pipeline.fit(
    train_df["question"],
    train_df["intent"]
)


# 8. Đánh giá mô hình tốt nhất trên tập Test
y_pred = best_pipeline.predict(test_df["question"])

accuracy = accuracy_score(test_df["intent"], y_pred)

f1 = f1_score(
    test_df["intent"],
    y_pred,
    average="macro"
)

print("\nTEST Accuracy:", round(accuracy, 4))
print("TEST Macro F1:", round(f1, 4))

# In kết quả Precision, Recall, F1 của từng intent
print("\nBáo cáo từng intent:")
print(classification_report(
    test_df["intent"],
    y_pred,
    zero_division=0
))

# Lưu các câu dự đoán sai để phân tích
errors = test_df[["question", "intent"]].copy()
errors["predicted"] = y_pred
errors = errors[errors["intent"] != errors["predicted"]]

errors.to_csv(
    "error_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Số câu dự đoán sai:", len(errors))


# 9. Lưu mô hình để sử dụng cho chatbot
joblib.dump(best_pipeline, "models/best_model.joblib")
print("Đã lưu: best_model.joblib")


# # 10. Thử dự đoán câu hỏi mới
# question = input("\nNhập câu hỏi để thử (Enter để bỏ qua): ").strip()

# if question:
#     question = " ".join(question.lower().split())
#     intent = best_pipeline.predict([question])[0]
#     print("Intent dự đoán:", intent)
