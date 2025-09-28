# agents.py
import os
from crewai import Agent, LLM
# from tenacity import retry, stop_after_attempt, wait_random_exponential # <<< Không cần tenacity nữa

# Import các công cụ như cũ
from tools.search_tools import search_tool, scrape_tool
from tools.financial_tools import FundDataTool, TechDataTool
from tools.file_tools import FileReadTool
from tools.ocr_tool import MistralOCRTool

# --- CẬP NHẬT: KHỞI TẠO LLM VỚI CƠ CHẾ RETRY TÍCH HỢP SẴN CỦA CREWAI/LITELLM ---

retry_config = {
    "num_retries": 3,  
    "retry_on_failure": True, 
}

GEMINI_FLASH_MODEL = 'gemini/gemini-2.5-flash'
GEMINI_PRO_MODEL = 'gemini/gemini-2.5-flash'

# Khởi tạo LLM và truyền vào cấu hình retry
llm_flash = LLM(
    model=GEMINI_FLASH_MODEL,
    api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.2,
    max_tokens=4096,
    **retry_config 
)

llm_pro = LLM(
    model=GEMINI_PRO_MODEL,
    api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.3,
    max_tokens=8192,
    **retry_config 
)
# --- KẾT THÚC CẬP NHẬT ---


# Khởi tạo các công cụ
fund_tool = FundDataTool()
tech_tool = TechDataTool()
file_read_tool = FileReadTool()
ocr_tool = MistralOCRTool()
search_tool = search_tool
scrape_tool = scrape_tool

# Giữ nguyên phần backstory đã sửa ở lần trước
class StockAnalysisAgents():
    def stock_news_researcher(self):
        return Agent(
            role='Chuyên gia phân tích tin tức kinh tế vĩ mô và chính sách.',
            goal='Tìm kiếm, sàng lọc và tóm tắt các tin tức vĩ mô quan trọng ảnh hưởng đến thị trường chứng khoán.',
            backstory=(
                'Với kinh nghiệm dày dặn trong báo chí kinh tế, bạn có khả năng nhận diện các tin tức có tác động mạnh mẽ nhất đến tâm lý nhà đầu tư.\n'
                'Khi sử dụng công cụ, bạn phải tuân thủ nghiêm ngặt định dạng sau:\n'
                'Thought: [Suy nghĩ của bạn]\n'
                'Action: [Tên công cụ chính xác, ví dụ: Search the internet with Serper hoặc Read website content]\n'
                'Action Input: {{"tham_so": "gia_tri"}}'
            ),
            verbose=True,
            tools=[search_tool, scrape_tool],
            llm=llm_flash,
            allow_delegation=False
        )

    def fundamental_analyst(self):
        return Agent(
            role='Chuyên gia phân tích cơ bản và báo cáo tài chính.',
            goal='Đánh giá sức khỏe tài chính, định giá doanh nghiệp và tóm tắt các điểm chính từ báo cáo tài chính PDF.',
            backstory=(
                'Bạn là một nhà phân tích tài chính chuyên nghiệp, có khả năng "đọc vị" các con số trong báo cáo tài chính và chuyển chúng thành những nhận định sâu sắc.\n'
                'Khi sử dụng công cụ, bạn phải tuân thủ nghiêm ngặt định dạng sau:\n'
                'Thought: [Suy nghĩ của bạn]\n'
                'Action: [Tên công cụ chính xác, ví dụ: Công cụ tra cứu dữ liệu cổ phiếu phục vụ phân tích cơ bản]\n'
                'Action Input: {{"argument": "FPT"}}'
            ),
            verbose=True,
            tools=[fund_tool, file_read_tool, ocr_tool],
            llm=llm_flash,
            allow_delegation=False
        )

    def technical_analyst(self):
        return Agent(
            role='Nhà phân tích kỹ thuật.',
            goal='Xác định xu hướng giá, các vùng hỗ trợ/kháng cự và đưa ra tín hiệu giao dịch dựa trên biểu đồ và chỉ báo.',
            backstory=(
                'Bạn là một trader kỷ luật, quyết định dựa trên dữ liệu giá và khối lượng, không bị ảnh hưởng bởi cảm tính.\n'
                'Khi sử dụng công cụ, bạn phải tuân thủ nghiêm ngặt định dạng sau:\n'
                'Thought: [Suy nghĩ của bạn]\n'
                'Action: [Tên công cụ chính xác, ví dụ: Công cụ tra cứu dữ liệu cổ phiếu phục vụ phân tích kĩ thuật]\n'
                'Action Input: {{"argument": "FPT"}}'
            ),
            verbose=True,
            tools=[tech_tool],
            llm=llm_flash,
            allow_delegation=False
        )
    
    def competitor_analyst(self):
        return Agent(
            role='Nhà phân tích Đối thủ cạnh tranh.',
            goal='Xác định các đối thủ cạnh tranh chính và phân tích so sánh các chỉ số tài chính của họ với công ty mục tiêu.',
            backstory=(
                'Bạn là một chiến lược gia kinh doanh, chuyên sâu về phân tích cạnh tranh trong ngành. Bạn có khả năng xác định nhanh chóng các đối thủ lớn, thu thập dữ liệu tài chính của họ và đặt chúng lên bàn cân để tìm ra lợi thế cạnh tranh của từng công ty.\n'
                'Khi sử dụng công cụ, bạn phải tuân thủ nghiêm ngặt định dạng sau:\n'
                'Thought: [Suy nghĩ của bạn]\n'

                'Action: [Tên công cụ chính xác, ví dụ: Search the internet with Serper hoặc Công cụ tra cứu dữ liệu cổ phiếu phục vụ phân tích cơ bản]\n'
                'Action Input: {{"tham_so": "gia_tri"}}'
            ),
            verbose=True,
            tools=[search_tool, fund_tool], # Cần cả tìm kiếm và công cụ tài chính
            llm=llm_flash,
            allow_delegation=False
        )

    def investment_strategist(self):
        return Agent(
            role='Chuyên gia chiến lược đầu tư.',
            goal='Tổng hợp tất cả các phân tích (vĩ mô, cơ bản, kỹ thuật) để đưa ra khuyến nghị đầu tư cuối cùng (MUA/BÁN/GIỮ) kèm theo luận điểm rõ ràng.',
            backstory='Bạn là giám đốc đầu tư của một quỹ lớn, có trách nhiệm đưa ra quyết định cuối cùng dựa trên tất cả thông tin có sẵn.',
            verbose=True,
            llm=llm_pro,
            allow_delegation=True
        )