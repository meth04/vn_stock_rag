import asyncio

import nest_asyncio
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from .config import DATA_DIR, INDEXING_MODEL, WORK_DIR
from .llm_wrapper import openai_complete_if_cache, text_embedding_func

nest_asyncio.apply()


async def main():
    print("🚀 Bắt đầu xây dựng Đồ thị tri thức (Knowledge Graph)...")

    rag = LightRAG(
        working_dir=str(WORK_DIR),
        llm_model_func=openai_complete_if_cache,
        llm_model_name=INDEXING_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=text_embedding_func,
        ),
    )

    print("📦 Đang khởi tạo các tệp tin hệ thống...")
    await rag.initialize_storages()

    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print("❌ Lỗi: Thư mục data/financial_reports trống!")
        return

    print(
        f"⚡ Đang nạp {len(txt_files)} tệp báo cáo vào Graph. Quá trình này sẽ gọi DeepSeek-V3 liên tục..."
    )

    for file_path in txt_files:
        print(f"   -> Đang nạp: {file_path.name}")
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        full_content = f"NGUỒN FILE: {file_path.name}\n\nNỘI DUNG:\n{content}"

        try:
            await rag.ainsert(full_content)
        except Exception as exc:  # noqa: BLE001
            print(f"     ⚠️ Lỗi khi xử lý {file_path.name}: {exc}")

    print("\n🎉 HOÀN TẤT! Toàn bộ tri thức đã được chuyển đổi thành Đồ thị.")
    print(f"📁 Index lưu tại: {WORK_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
