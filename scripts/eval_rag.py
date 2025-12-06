from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------- Monitoring stubs (Prometheus/OpenTelemetry) ---------
class PrometheusMonitor:
    """Stub monitor that mimics Prometheus metrics collection."""

    def __init__(self) -> None:
        self.counters: dict[str, float] = {}
        self.observations: dict[str, list[float]] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + value
        logger.debug("Prometheus counter %s incremented to %.2f", name, self.counters[name])

    def observe(self, name: str, value: float) -> None:
        self.observations.setdefault(name, []).append(value)
        logger.debug("Prometheus observation %s recorded value %.4f", name, value)


class OpenTelemetryTracer:
    """Simple tracer stub to time operations."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, float]] = []

    def start_span(self, name: str) -> float:
        start = time.perf_counter()
        self.spans.append((name, start))
        logger.debug("Span %s started", name)
        return start

    def end_span(self, name: str, start_time: float) -> None:
        duration = time.perf_counter() - start_time
        logger.debug("Span %s ended after %.4fs", name, duration)


# -------------------- Evaluation utilities --------------------
@dataclass
class EvaluationQuestion:
    question: str
    expected_keywords: list[str]
    ground_truth: str
    reference_context: str


@dataclass
class RetrievalResult:
    text: str
    score: float


@dataclass
class QuestionResult:
    question: str
    answer: str
    precision_at_k: float
    faithfulness: float
    retrieved: list[RetrievalResult] = field(default_factory=list)


class SimpleRetriever:
    """Lightweight keyword-based retriever for demo evaluation."""

    def __init__(self, data_path: Path | None, fallback_contexts: Sequence[str]):
        self.data_path = data_path if data_path and data_path.exists() else None
        self.fallback_contexts = fallback_contexts
        self.dataset: list[str] = []
        if self.data_path is not None:
            frame = pd.read_parquet(self.data_path)
            self.dataset = frame["text"].astype(str).tolist()
            logger.info("Loaded %d chunks from %s", len(self.dataset), self.data_path)
        else:
            self.dataset = list(self.fallback_contexts)
            logger.info("Using %d fallback contexts (no parquet dataset found)", len(self.dataset))

    @staticmethod
    def _score(text: str, keywords: Iterable[str]) -> float:
        text_lower = text.lower()
        return sum(text_lower.count(keyword.lower()) for keyword in keywords)

    def retrieve(self, query: EvaluationQuestion, k: int) -> list[RetrievalResult]:
        scored = [
            RetrievalResult(text=chunk, score=self._score(chunk, query.expected_keywords))
            for chunk in self.dataset
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        top_k = scored[:k]
        logger.debug("Top-%d scores: %s", k, [result.score for result in top_k])
        return top_k


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = {token for token in text_a.lower().split() if token}
    tokens_b = {token for token in text_b.lower().split() if token}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


class RAGEvaluator:
    def __init__(
        self,
        questions: Sequence[EvaluationQuestion],
        retriever: SimpleRetriever,
        monitor: PrometheusMonitor,
        tracer: OpenTelemetryTracer,
    ) -> None:
        self.questions = list(questions)
        self.retriever = retriever
        self.monitor = monitor
        self.tracer = tracer

    @staticmethod
    def _is_relevant(text: str, keywords: Iterable[str]) -> bool:
        lower = text.lower()
        return any(keyword.lower() in lower for keyword in keywords)

    @staticmethod
    def _compose_answer(retrieved: list[RetrievalResult], ground_truth: str) -> str:
        if retrieved:
            return " ".join(result.text for result in retrieved[:1])
        return ground_truth

    def evaluate_question(self, question: EvaluationQuestion, k: int) -> QuestionResult:
        span = self.tracer.start_span("retrieve")
        retrieved = self.retriever.retrieve(question, k)
        self.tracer.end_span("retrieve", span)

        precision = 0.0
        if retrieved:
            relevant = sum(
                1 for item in retrieved if self._is_relevant(item.text, question.expected_keywords)
            )
            precision = relevant / max(k, 1)
        self.monitor.observe("precision", precision)

        answer = self._compose_answer(retrieved, question.ground_truth)
        contexts_text = " ".join(item.text for item in retrieved)
        faithfulness = jaccard_similarity(answer, contexts_text or question.reference_context)
        self.monitor.observe("faithfulness", faithfulness)

        logger.info(
            "Question: %s | precision@%d=%.3f | faithfulness=%.3f",
            question.question,
            k,
            precision,
            faithfulness,
        )

        return QuestionResult(
            question=question.question,
            answer=answer,
            precision_at_k=precision,
            faithfulness=faithfulness,
            retrieved=retrieved,
        )

    def run(self, k: int) -> dict:
        results = [self.evaluate_question(question, k) for question in self.questions]
        aggregate_precision = sum(item.precision_at_k for item in results) / len(results)
        aggregate_faithfulness = sum(item.faithfulness for item in results) / len(results)
        self.monitor.inc("questions_evaluated", len(results))
        return {
            "aggregate": {
                "precision_at_k": aggregate_precision,
                "faithfulness": aggregate_faithfulness,
            },
            "results": results,
        }


# ------------------------- Plotting ---------------------------
def plot_metrics(results: list[QuestionResult], output_path: Path) -> None:
    labels = [f"Q{idx + 1}" for idx in range(len(results))]
    precision_values = [item.precision_at_k for item in results]
    faithfulness_values = [item.faithfulness for item in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = range(len(labels))
    ax.bar([i - width / 2 for i in x], precision_values, width, label="Precision@k")
    ax.bar([i + width / 2 for i in x], faithfulness_values, width, label="Faithfulness")

    ax.set_xlabel("Câu hỏi")
    ax.set_ylabel("Điểm số")
    ax.set_ylim(0, 1.05)
    ax.set_title("RAG Evaluation Metrics")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved metric plot to %s", output_path)


# --------------------------- CLI ------------------------------
def load_default_questions() -> list[EvaluationQuestion]:
    return [
        EvaluationQuestion(
            question="Các triệu chứng điển hình của lao phổi là gì?",
            expected_keywords=["ho", "sốt", "gầy sút", "lao phổi"],
            ground_truth="Ho kéo dài, sốt nhẹ, đổ mồ hôi đêm và gầy sút cân là các triệu chứng điển hình.",
            reference_context=(
                "Bệnh nhân lao phổi thường có ho kéo dài, ho khan hoặc ho đờm, sốt nhẹ về chiều, "
                "ra mồ hôi đêm, chán ăn và gầy sút cân."
            ),
        ),
        EvaluationQuestion(
            question="X-quang phổi giúp xác định dấu hiệu gì trong viêm phổi?",
            expected_keywords=["thâm nhiễm", "x-quang", "viêm phổi"],
            ground_truth="Hình ảnh thâm nhiễm hoặc đông đặc nhu mô là dấu hiệu gợi ý viêm phổi trên X-quang.",
            reference_context=(
                "Trên phim X-quang ngực, viêm phổi thường thể hiện vùng thâm nhiễm phế nang hoặc đông đặc "
                "kèm theo mờ đồng nhất."
            ),
        ),
        EvaluationQuestion(
            question="Yếu tố nguy cơ nào liên quan đến tràn dịch màng phổi do lao?",
            expected_keywords=["lao", "màng phổi", "nguy cơ", "HIV"],
            ground_truth="Nhiễm HIV, đái tháo đường hoặc suy giảm miễn dịch là các yếu tố nguy cơ liên quan.",
            reference_context=(
                "Tràn dịch màng phổi do lao thường gặp ở người nhiễm HIV, bệnh nhân đái tháo đường hoặc "
                "những trường hợp suy giảm miễn dịch kéo dài."
            ),
        ),
    ]


def save_json(result: dict, output_path: Path) -> None:
    def encode(obj):
        if isinstance(obj, QuestionResult):
            return {
                "question": obj.question,
                "answer": obj.answer,
                "precision_at_k": obj.precision_at_k,
                "faithfulness": obj.faithfulness,
                "retrieved": [vars(item) for item in obj.retrieved],
            }
        if isinstance(obj, RetrievalResult):
            return vars(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2, default=encode)
    logger.info("Saved evaluation report to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with sample questions")
    parser.add_argument("--data", type=Path, default=Path("data/processed/chunks.parquet"))
    parser.add_argument("--k", type=int, default=3, help="Number of contexts to retrieve")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/eval"), help="Directory to store evaluation reports"
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir: Path = args.output_dir
    json_path = output_dir / f"{timestamp}_eval.json"
    plot_path = output_dir / f"{timestamp}_metrics.png"

    monitor = PrometheusMonitor()
    tracer = OpenTelemetryTracer()
    questions = load_default_questions()
    fallback_contexts: list[str] = [q.reference_context for q in questions]
    retriever = SimpleRetriever(args.data, fallback_contexts)
    evaluator = RAGEvaluator(questions, retriever, monitor, tracer)
    raw_result = evaluator.run(k=args.k)

    payload = {
        "metadata": {
            "generated_at": timestamp,
            "k": args.k,
            "data_path": str(args.data),
            "dataset_available": retriever.data_path is not None,
        },
        "aggregate": raw_result["aggregate"],
        "results": raw_result["results"],
    }

    save_json(payload, json_path)
    plot_metrics(raw_result["results"], plot_path)


if __name__ == "__main__":
    main()
