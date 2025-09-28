# tools/file_tools.py
from typing import Type, Optional, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class FileReadToolSchema(BaseModel):
    """Input for FileReadTool."""

    file_path: str = Field(..., description="Mandatory file full path to read the file")
    start_line: Optional[int] = Field(1, description="Line number to start reading from (1-indexed)")
    line_count: Optional[int] = Field(None, description="Number of lines to read. If None, reads the entire file")


class FileReadTool(BaseTool):
    name: str = "Read a file's content"
    description: str = "A tool that reads the content of a file. To use this tool, provide a 'file_path' parameter with the path to the file you want to read. Optionally, provide 'start_line' to start reading from a specific line and 'line_count' to limit the number of lines read."
    args_schema: Type[BaseModel] = FileReadToolSchema
    file_path: Optional[str] = None

    def __init__(self, file_path: Optional[str] = None, **kwargs: Any) -> None:
        if file_path is not None:
            kwargs["description"] = (
                f"A tool that reads file content. The default file is {file_path}, but you can provide a different 'file_path' parameter to read another file. You can also specify 'start_line' and 'line_count' to read specific parts of the file."
            )
        super().__init__(**kwargs)
        self.file_path = file_path

    def _run(self, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path", self.file_path)
        start_line = kwargs.get("start_line", 1)
        line_count = kwargs.get("line_count", None)

        if file_path is None:
            return "Error: No file path provided. Please provide a file path either in the constructor or as an argument."
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                if start_line == 1 and line_count is None:
                    return file.read()
                start_idx = max(start_line - 1, 0)
                selected_lines = [
                    line
                    for i, line in enumerate(file)
                    if i >= start_idx and (line_count is None or i < start_idx + line_count)
                ]
                if not selected_lines and start_idx > 0:
                    return f"Error: Start line {start_line} exceeds the number of lines in the file."
                return "".join(selected_lines)
        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except PermissionError:
            return f"Error: Permission denied when trying to read file: {file_path}"
        except Exception as e:
            return f"Error: Failed to read file {file_path}. {str(e)}"