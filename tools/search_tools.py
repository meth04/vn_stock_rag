# tools/search_tools.py
import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Khởi tạo công cụ tìm kiếm với API key từ file .env
search_tool = SerperDevTool(api_key=os.environ.get("SERPER_API_KEY"))

# Khởi tạo công cụ cào dữ liệu web
scrape_tool = ScrapeWebsiteTool()