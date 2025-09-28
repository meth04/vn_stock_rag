# tools/ocr_tool.py

import os
import sys
import json
from pathlib import Path
from typing import Any, Type, List, Dict
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from mistralai import Mistral

def serialize(obj: Any) -> Any:
    import dataclasses
    if obj is None or isinstance(obj, (str, bool, int, float)): return obj
    if isinstance(obj, dict): return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [serialize(i) for i in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try: return serialize(obj.to_dict())
        except Exception: pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try: return serialize(obj.dict())
        except Exception: pass
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try: return serialize(obj.model_dump())
        except Exception: pass
    if hasattr(obj, "json") and callable(getattr(obj, "json")):
        try:
            j = obj.json()
            return json.loads(j) if isinstance(j, str) else serialize(j)
        except Exception: pass
    try:
        if dataclasses.is_dataclass(obj): return serialize(dataclasses.asdict(obj))
    except Exception: pass
    if hasattr(obj, "__dict__"):
        try: return serialize(vars(obj))
        except Exception: pass
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try: return [serialize(i) for i in obj]
        except Exception: pass
    return str(obj)

class OCRToolInput(BaseModel):
    pdf_path: str = Field(..., description="Đường dẫn đầy đủ và chính xác đến file PDF cần trích xuất văn bản.")

class MistralOCRTool(BaseTool):
    name: str = "Công cụ trích xuất văn bản từ file PDF"
    description: str = (
        "Rất hữu ích khi bạn cần đọc và trích xuất toàn bộ nội dung văn bản từ một file PDF. "
        "Đầu vào là đường dẫn đến file PDF. "
        "Đầu ra là một chuỗi thông báo thành công và đường dẫn đến file .txt chứa nội dung đã được trích xuất."
    )
    args_schema: Type[BaseModel] = OCRToolInput

    def _run(self, pdf_path: str) -> str:
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                return f"Lỗi: Không tìm thấy file tại đường dẫn '{pdf_path}'. Vui lòng kiểm tra lại đường dẫn."
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                return "Lỗi: Biến môi trường MISTRAL_API_KEY chưa được thiết lập."

            output_prefix = pdf_file.with_suffix("")
            client = Mistral(api_key=api_key)

            print(f"\n[OCR Tool] Đang tải lên file: {pdf_file.name}...")
            with open(pdf_file, "rb") as f:
                uploaded = client.files.upload(file={"file_name": pdf_file.name, "content": f}, purpose="ocr")
            file_id = getattr(uploaded, "id", None)
            if not file_id:
                return f"Lỗi: Không thể lấy ID của file sau khi tải lên. Phản hồi từ API: {serialize(uploaded)}"
            
            print(f"[OCR Tool] Tải lên thành công. File ID: {file_id}")
            signed = client.files.get_signed_url(file_id=file_id)
            url = getattr(signed, "url", None)
            if not url:
                return f"Lỗi: Không thể lấy URL đã ký cho file ID: {file_id}"

            print("[OCR Tool] Đang gọi API của Mistral OCR...")
            ocr_resp = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": url}
            )

            raw_response = serialize(ocr_resp)

            print("[OCR Tool] Đang xử lý phản hồi từ API...")
            
            print("[OCR Tool] Snippet phản hồi thô từ API:")
            print(json.dumps(raw_response, indent=2, ensure_ascii=False)[:1000]) # In 1000 ký tự đầu

            pages_data = []
            if isinstance(raw_response, dict):
                possible_page_keys = ["pages", "page", "pages_data"]
                for key in possible_page_keys:
                    if key in raw_response and isinstance(raw_response[key], list):
                        pages_data = raw_response[key]
                        break
            elif isinstance(raw_response, list):
                pages_data = raw_response

            extracted_pages: List[Dict] = []
            if pages_data:
                for i, p_data in enumerate(pages_data, start=1):
                    text = ""
                    page_number = i
                    if isinstance(p_data, dict):
                        possible_text_keys = ["text", "plain_text", "markdown", "content"]
                        for key in possible_text_keys:
                            if key in p_data and p_data[key]:
                                text = p_data[key]
                                break
                        possible_page_num_keys = ["page_number", "page", "index"]
                        for key in possible_page_num_keys:
                            if key in p_data:
                                page_number = p_data[key]
                                break
                    elif isinstance(p_data, str):
                        text = p_data
                    
                    extracted_pages.append({"page_number": page_number, "text": text})
            
            if not extracted_pages and isinstance(raw_response, dict):
                top_level_text = raw_response.get("text")
                if top_level_text:
                    extracted_pages.append({"page_number": 1, "text": top_level_text})

            if not extracted_pages:
                return f"Lỗi: Không thể trích xuất được nội dung văn bản từ phản hồi của API. Vui lòng kiểm tra snippet log ở trên."

            all_text = "\n\n".join(
                [f"--- TRANG {p['page_number']} ---\n{p['text']}" for p in extracted_pages]
            )

            txt_path = Path(str(output_prefix) + ".ocr_text.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(all_text)
            
            print(f"[OCR Tool] Trích xuất thành công!")
            
            return f"Trích xuất OCR từ file '{pdf_file.name}' thành công. Toàn bộ nội dung đã được lưu tại file: '{txt_path}'"

        except Exception as e:
            error_message = f"Đã xảy ra lỗi không mong muốn trong quá trình OCR: {e}"
            print(f"[OCR Tool] {error_message}", file=sys.stderr)
            return error_message