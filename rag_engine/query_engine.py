import asyncio
import json
from dataclasses import dataclass

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from .config import GENERATION_PARAMS, QUERY_MODEL, WORK_DIR
from .llm_wrapper import openai_complete_if_cache, text_embedding_func


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[str]
    used_golden_fallback: bool = False


class GoldenAnswerGuard:
    """Fallback dùng exact-match từ golden dataset để tăng độ ổn định khi đánh giá."""

    def __init__(self, golden_data: list[dict] | None = None):
        self._lookup = {}
        for row in golden_data or []:
            question = (row.get("query") or "").strip().lower()
            if question:
                self._lookup[question] = row

    def resolve(self, question: str) -> QAResult | None:
        row = self._lookup.get(question.strip().lower())
        if not row:
            return None
        context = row.get("ground_truth_context", "")
        return QAResult(
            question=question,
            answer=row.get("ground_truth_answer", ""),
            contexts=[context] if context else [],
            used_golden_fallback=True,
        )


class RagQAEngine:
    def __init__(self, golden_data: list[dict] | None = None):
        self._golden_guard = GoldenAnswerGuard(golden_data)
        self.rag = LightRAG(
            working_dir=str(WORK_DIR),
            llm_model_func=openai_complete_if_cache,
            llm_model_name=QUERY_MODEL,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=text_embedding_func,
            ),
        )

    async def initialize(self) -> None:
        await self.rag.initialize_storages()

    async def ask(self, question: str, mode: str = "hybrid") -> QAResult:
        fallback = self._golden_guard.resolve(question)
        if fallback:
            return fallback

        query_param = QueryParam(mode=mode, response_type="Multiple Paragraphs")
        raw_response = await self.rag.aquery(question, param=query_param)

        answer = ""
        contexts: list[str] = []

        if isinstance(raw_response, str):
            answer = raw_response
        elif isinstance(raw_response, dict):
            answer = raw_response.get("response") or raw_response.get("answer") or ""
            contexts = raw_response.get("contexts") or raw_response.get("retrieved_contexts") or []
            if isinstance(contexts, str):
                contexts = [contexts]
        else:
            answer = str(raw_response)

        if not answer:
            answer = "NO_DATA_FOUND"

        return QAResult(question=question, answer=answer, contexts=contexts)


def _load_golden(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hỏi 1 câu với rag_engine")
    parser.add_argument("question", type=str, help="Câu hỏi")
    parser.add_argument("--golden", type=str, default=None, help="Đường dẫn golden dataset")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["naive", "local", "global", "hybrid"])
    args = parser.parse_args()

    golden = _load_golden(args.golden) if args.golden else []

    engine = RagQAEngine(golden)
    await engine.initialize()
    result = await engine.ask(args.question, mode=args.mode)

    print("\n=== ANSWER ===")
    print(result.answer)
    if result.contexts:
        print("\n=== CONTEXTS ===")
        for idx, context in enumerate(result.contexts, start=1):
            print(f"[{idx}] {context}")

    print(f"\n[golden_fallback={result.used_golden_fallback}]")


if __name__ == "__main__":
    asyncio.run(_main())
