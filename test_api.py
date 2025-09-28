# test_gemini.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Tải file .env để lấy API key và cấu hình SSL
load_dotenv()

# Cấu hình SSL (quan trọng trên Windows)
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file:
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ssl_cert_file

try:
    print("Đang cấu hình Google API...")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model_name_to_test = "gemini-2.5-flash" 

    print(f"Đang khởi tạo model: {model_name_to_test}...")
    model = genai.GenerativeModel(model_name_to_test)

    print("Đang gửi một yêu cầu đơn giản...")
    response = model.generate_content("Chào bạn, bạn có khỏe không?")

    print("\n--- KẾT QUẢ THÀNH CÔNG ---")
    print(response.text)
    print("--------------------------")
    print("\n>>> Kết nối đến Google Gemini thành công với model này! Vấn đề có thể nằm ở việc gọi song song trong crewAI.")

except Exception as e:
    print("\n--- ĐÃ XẢY RA LỖI ---")
    print(f"Lỗi: {e}")
    print("----------------------")
    print("\n>>> Nếu bạn vẫn thấy lỗi 503 ở đây, vấn đề nằm ở API key hoặc tài khoản Google của bạn, không phải do crewAI.")

    "C:/Users/nguye/Documents/vn_stock_rag/bctcfpt.pdf"