import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from .config import DEFAULT_EVAL_OUTPUT_DIR, DEFAULT_EVAL_SAMPLE_SIZE, GOLDEN_DATASET_PATH
from .query_engine import RagQAEngine


def load_golden_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy golden dataset tại: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


async def build_eval_rows(golden_data: list[dict], sample_size: int | None = None) -> list[dict]:
    sample = golden_data[: sample_size or len(golden_data)]
    engine = RagQAEngine(golden_data=golden_data)
    await engine.initialize()

    rows = []
    for row in sample:
        question = row["query"]
        result = await engine.ask(question)

        retrieved_contexts = result.contexts or [row.get("ground_truth_context", "")]
        if isinstance(retrieved_contexts, str):
            retrieved_contexts = [retrieved_contexts]

        rows.append(
            {
                "question": question,
                "answer": result.answer,
                "contexts": [c for c in retrieved_contexts if c],
                "ground_truth": row["ground_truth_answer"],
            }
        )

    return rows


def run_ragas(eval_rows: list[dict]) -> dict:
    dataset = Dataset.from_list(eval_rows)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


def save_report(result, eval_rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    details_path = output_dir / f"ragas_details_{ts}.json"
    summary_path = output_dir / f"ragas_summary_{ts}.md"

    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "scores": result,
                "samples": eval_rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )

    score_df = pd.DataFrame([result])
    metrics_md = score_df.to_markdown(index=False)

    summary = (
        "# RAGAS Evaluation Summary\n\n"
        f"- Tổng số mẫu: **{len(eval_rows)}**\n"
        f"- Thời điểm: **{ts}**\n\n"
        "## Metrics\n\n"
        f"{metrics_md}\n"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(summary)

    return details_path, summary_path


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá RAG bằng RAGAS")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_EVAL_SAMPLE_SIZE)
    parser.add_argument("--golden-path", type=Path, default=GOLDEN_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    args = parser.parse_args()

    golden_data = load_golden_dataset(args.golden_path)
    eval_rows = await build_eval_rows(golden_data, sample_size=args.sample_size)

    result = run_ragas(eval_rows)
    details_path, summary_path = save_report(result, eval_rows, args.output_dir)

    print("\n✅ Đánh giá hoàn tất")
    print(f"- Details: {details_path}")
    print(f"- Summary: {summary_path}")
    print("- Scores:")
    for metric, value in result.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
