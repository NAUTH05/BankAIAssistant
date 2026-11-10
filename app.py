
import gradio as gr
from chatbot import chatbot


# Hàm xử lý tin nhắn từ giao diện
def reply(message, history):

    answer, intent, method = chatbot(message)

    return answer


# Tạo giao diện chatbot
demo = gr.ChatInterface(
    fn=reply,
    title="BankAI - Chatbot hỗ trợ ngân hàng",
    description="Chatbot hỗ trợ giải đáp các câu hỏi ngân hàng bằng tiếng Việt. Đây là dự án mô phỏng phục vụ học tập."
)

demo = gr.ChatInterface(
    fn=reply,
    title="BankAI - Chatbot hỗ trợ ngân hàng",
    description="Chatbot ngân hàng tiếng Việt - Dự án DT51.",
    examples=[
        "Tôi muốn mở tài khoản ngân hàng",
        "Tôi làm mất thẻ ATM",
        "Tôi quên mật khẩu ứng dụng",
        "Chuyển tiền không thành công",
        "Làm sao để gửi tiết kiệm?"
    ]
)


# Chạy giao diện trên máy cá nhân
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
