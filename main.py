# main.py

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from crewai import Crew, Process
from agents import StockAnalysisAgents
from tasks import StockAnalysisTasks
from datetime import datetime

# --- NÂNG CẤP HÀM PHÂN TÍCH YÊU CẦU ---
def get_user_intent_with_mistral(user_query: str) -> dict:
    try:
        print("Đang dùng Mistral AI để phân tích yêu cầu...")
        API_KEY = os.environ.get("MISTRAL_API_KEY")
        url = "https://api.mistral.ai/v1/chat/completions"
        
        prompt = f"""
        Phân tích yêu cầu của người dùng và trả về một đối tượng JSON.
        Yêu cầu: "{user_query}"

        Các nhiệm vụ có thể có:
        - "analyze_stock": Nếu người dùng chỉ muốn phân tích một mã cổ phiếu.
        - "analyze_pdf": Nếu người dùng chỉ muốn phân tích một file PDF.
        - "comprehensive_analysis": Nếu người dùng muốn phân tích một mã cổ phiếu VÀ có cung cấp một file PDF báo cáo tài chính đi kèm.
        - "unknown": Nếu không xác định được.

        Đối tượng JSON phải có 3 trường: "task", "ticker", "file_path".
        - "task": điền một trong các giá trị trên.
        - "ticker": trích xuất mã cổ phiếu. Nếu không có, điền null.
        - "file_path": trích xuất đường dẫn file PDF. Nếu không có, điền null.

        Chỉ trả về đối tượng JSON, không giải thích gì thêm.

        Ví dụ 1 (Chỉ cổ phiếu): "phân tích fpt" -> {{"task": "analyze_stock", "ticker": "FPT", "file_path": null}}
        Ví dụ 2 (Chỉ PDF): "tóm tắt file C:/bctc.pdf" -> {{"task": "analyze_pdf", "ticker": null, "file_path": "C:/bctc.pdf"}}
        Ví dụ 3 (Toàn diện): "phân tích cổ phiếu FPT dựa vào báo cáo này D:\\data\\FPT_Q2.pdf" -> {{"task": "comprehensive_analysis", "ticker": "FPT", "file_path": "D:\\\\data\\\\FPT_Q2.pdf"}}
        """
        payload = { "model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"} }
        headers = { "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "Accept": "application/json" }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        import json
        intent_data = json.loads(response.json()["choices"][0]["message"]["content"])
        
        print(f"Mistral đã phân tích yêu cầu: {intent_data}")
        return intent_data
    except Exception as e:
        print(f"Lỗi khi phân tích yêu cầu: {e}")
        return {"task": "unknown", "ticker": None, "file_path": None}

class FinancialCrew:
    def __init__(self, symbol=None, file_path=None):
        self.symbol = symbol
        self.file_path = file_path
        self.agents = StockAnalysisAgents()
        self.tasks = StockAnalysisTasks()

    def run_stock_analysis(self):
        # Tạo các agent
        news_researcher = self.agents.stock_news_researcher()
        fundamental_analyst = self.agents.fundamental_analyst()
        technical_analyst = self.agents.technical_analyst()
        competitor_analyst = self.agents.competitor_analyst() # Agent mới
        investment_strategist = self.agents.investment_strategist()

        # Tạo các task
        news_task = self.tasks.news_collecting(news_researcher, self.symbol)
        fundamental_task = self.tasks.fundamental_analysis(fundamental_analyst, self.symbol)
        technical_task = self.tasks.technical_analysis(technical_analyst, self.symbol)
        competitor_task = self.tasks.competitor_analysis(competitor_analyst, self.symbol) # Task mới
        
        investment_task = self.tasks.investment_decision(
            investment_strategist, self.symbol,
            # Thêm competitor_task vào context
            [news_task, fundamental_task, technical_task, competitor_task]
        )
        
        crew = Crew(
            agents=[news_researcher, fundamental_analyst, technical_analyst, competitor_analyst, investment_strategist],
            tasks=[news_task, fundamental_task, technical_task, competitor_task, investment_task],
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff()

    def run_pdf_analysis(self):
        pdf_task = self.tasks.analyze_financial_report(
            agent=self.agents.fundamental_analyst(),
            file_path=self.file_path,
            company_ticker=self.symbol
        )
        crew = Crew(agents=[self.agents.fundamental_analyst()], tasks=[pdf_task], verbose=True)
        return crew.kickoff()

    def run_comprehensive_analysis(self):
        # Tạo các agent
        news_researcher = self.agents.stock_news_researcher()
        technical_analyst = self.agents.technical_analyst()
        fundamental_analyst = self.agents.fundamental_analyst()
        competitor_analyst = self.agents.competitor_analyst() # Agent mới
        investment_strategist = self.agents.investment_strategist()

        # Tạo các task
        news_task = self.tasks.news_collecting(news_researcher, self.symbol)
        tech_task = self.tasks.technical_analysis(technical_analyst, self.symbol)
        pdf_analysis_task = self.tasks.analyze_financial_report(fundamental_analyst, self.file_path, self.symbol)
        competitor_task = self.tasks.competitor_analysis(competitor_analyst, self.symbol) # Task mới

        comprehensive_task = self.tasks.comprehensive_stock_analysis(
            agent=investment_strategist,
            symbol=self.symbol,
            # Thêm competitor_task vào context
            context=[news_task, tech_task, pdf_analysis_task, competitor_task]
        )
        
        crew = Crew(
            agents=[news_researcher, technical_analyst, fundamental_analyst, competitor_analyst, investment_strategist],
            tasks=[news_task, tech_task, pdf_analysis_task, competitor_task, comprehensive_task],
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff()

if __name__ == "__main__":
    print("## Chào mừng bạn đến với Crew Phân tích Cổ phiếu ##")
    print('--------------------------------------------------')
    user_input = input("Nhập yêu cầu của bạn: ")
    
    intent = get_user_intent_with_mistral(user_input)
    result = None
    report_filename = ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if intent['task'] == 'comprehensive_analysis' and intent['ticker'] and intent['file_path']:
        print(f"Bắt đầu quy trình phân tích TOÀN DIỆN cho {intent['ticker']} sử dụng file {intent['file_path']}")
        financial_crew = FinancialCrew(symbol=intent['ticker'], file_path=intent['file_path'])
        result = financial_crew.run_comprehensive_analysis()
        report_filename = f"reports/{intent['ticker'].lower()}_comprehensive_analysis_{timestamp}.md"

    elif intent['task'] == 'analyze_stock' and intent['ticker']:
        print(f"Bắt đầu quy trình phân tích cổ phiếu: {intent['ticker']}")
        financial_crew = FinancialCrew(symbol=intent['ticker'])
        result = financial_crew.run_stock_analysis()
        report_filename = f"reports/{intent['ticker'].lower()}_stock_analysis_{timestamp}.md"

    elif intent['task'] == 'analyze_pdf' and intent['file_path']:
        print(f"Bắt đầu quy trình phân tích file PDF: {intent['file_path']}")
        financial_crew = FinancialCrew(file_path=intent['file_path'], symbol=intent.get('ticker'))
        result = financial_crew.run_pdf_analysis()
        base_name = os.path.basename(intent['file_path']).split('.')[0]
        report_filename = f"reports/{base_name}_pdf_analysis_{timestamp}.md"
        
    else:
        print("Không thể xác định yêu cầu của bạn. Vui lòng thử lại với yêu cầu rõ ràng hơn.")

    if result and report_filename:
        os.makedirs('reports', exist_ok=True)
        with open(report_filename, "w", encoding='utf-8') as f:
            f.write(result.raw)
        print('--------------------------------------------------')
        print(f"Báo cáo phân tích đã được lưu tại: {report_filename}")
