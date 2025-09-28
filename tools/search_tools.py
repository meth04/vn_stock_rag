# tools/search_tools.py
import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

search_tool = SerperDevTool(api_key=os.environ.get("SERPER_API_KEY"))

scrape_tool = ScrapeWebsiteTool()