# 📊 vn_stock_RAG  

## 🔎 Giới thiệu  
**vn_stock_RAG** là một hệ thống **Retrieval-Augmented Generation (RAG)**, được xây dựng bằng `crewAI`.  
Hệ thống cho phép phân tích cổ phiếu bằng cách:  
- **Truy xuất (Retrieval)** dữ liệu từ nhiều nguồn (web, API tài chính, báo cáo tài chính, file JSON,...).  
- **Bổ sung (Augmentation)** ngữ cảnh giữa các agent trong pipeline.  
- **Sinh (Generation)** báo cáo phân tích và chiến lược đầu tư bằng LLM (Gemini API).  

Mục tiêu: tạo một **trợ lý đầu tư thông minh**, có khả năng phân tích cổ phiếu dựa trên thông tin **thực tế và cập nhật** thay vì chỉ dựa vào kiến thức huấn luyện cũ.  

---

## 🏗️ Kiến trúc hệ thống  

### Các bước chính:
1. **News Researcher** → thu thập tin tức từ web.  
2. **Financial Data Analyst** → lấy số liệu tài chính từ API (P/E, EPS, ROE, ...).  
3. **Technical Analyst** → phân tích dữ liệu kỹ thuật (RSI, MACD, ...).  
4. **Fundamental Analyst** → xử lý báo cáo tài chính hoặc các tài liệu khác.  
5. **Investment Strategist** → tổng hợp toàn bộ báo cáo, đưa ra quyết định.  

### Công nghệ sử dụng:
- [crewAI](https://github.com/joaomdmoura/crewAI) – Framework điều phối agent.  
- **Gemini API (Google Generative AI)** – LLM backend (`gemini-2.5-pro`, `gemini-2.5-flash`).  
- **vnstock API** – Dữ liệu tài chính Việt Nam.  
- **Serper API** – Tìm kiếm web.  
- **MistralOCRTool** – OCR file PDF.  

---

## ⚙️ Yêu cầu cài đặt  

### 1. Cài đặt môi trường  
```bash
# clone repo
git clone https://github.com/meth04/vn_stock_rag.git
cd vn_stock_rag

python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# cài dependencies
pip install -r requirements.txt
```

### 2. Cấu hình API Keys
Tạo file `.env` ở thư mục gốc:
# Google Generative AI (Gemini)
GOOGLE_API_KEY=your_google_api_key

# Serper API
SERPER_API_KEY=your_serper_api_key

#Mistral API
MISTRAL_API_KEY=your_mistral_api_key

### 3. Cách chạy
```bash
python main.py
```

Hãy hỏi Agent về công ty mà bạn muốn. Báo cáo cuối sẽ được lưu tại thư mục reports.