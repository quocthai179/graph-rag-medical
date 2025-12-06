"""Prompt templates for combining textual context and graph data."""
from collections.abc import Mapping
from typing import Iterable, Sequence


def format_graph_triples(graph_triples: Iterable[object]) -> str:
    """Format a collection of graph triples into a readable bullet list.

    Each triple can be a mapping with ``subject``/``predicate``/``object`` keys
    (or their common aliases) or a sequence of three values. Unknown structures
    are converted to strings so the caller can still inspect the content.
    """

    bullets: list[str] = []

    for triple in graph_triples or []:
        subject: str | None
        predicate: str | None
        obj: str | None

        if isinstance(triple, Mapping):
            subject = (
                triple.get("subject")
                or triple.get("head")
                or triple.get("source")
                or triple.get("s")
            )
            predicate = (
                triple.get("predicate")
                or triple.get("relation")
                or triple.get("edge")
                or triple.get("p")
            )
            obj = triple.get("object") or triple.get("tail") or triple.get("target") or triple.get("o")
        elif isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)) and len(triple) >= 3:
            subject, predicate, obj = triple[0], triple[1], triple[2]
        else:
            subject, predicate, obj = str(triple), "", ""

        subject = subject if subject is not None else "(không rõ chủ thể)"
        predicate = predicate if predicate is not None else "liên hệ"
        obj = obj if obj is not None else "(không rõ đối tượng)"
        bullets.append(f"- **{subject}** --{predicate}--> {obj}")

    if not bullets:
        return "(không có dữ liệu quan hệ)"

    return "\n".join(bullets)


def build_context_graph_prompt(
    *,
    question: str,
    context: str,
    graph_triples: Iterable[object] | None = None,
    guardrails: bool = True,
) -> str:
    """Create a prompt that combines unstructured context and graph triples.

    The template asks the LLM to answer in Vietnamese, cite sources, and
    optionally apply guardrails to remind the model to fix missing citations or
    missing sources before responding.
    """

    graph_section = format_graph_triples(graph_triples)

    guardrail_section = ""
    if guardrails:
        guardrail_section = (
            "\nKiểm tra cuối cùng:\n"
            "- Đảm bảo mọi ý quan trọng đều có trích dẫn nguồn từ phần bối cảnh hoặc quan hệ đồ thị.\n"
            "- Nếu phát hiện thiếu nguồn hoặc thiếu cite, hãy tự sửa và bổ sung trích dẫn trước khi gửi.\n"
            "- Nếu thực sự không đủ nguồn để trả lời, hãy nêu rõ điều đó thay vì suy đoán.\n"
        )

    prompt = (
        "Bạn là trợ lý y khoa, chỉ sử dụng thông tin được cung cấp.\n"
        "Trả lời ngắn gọn bằng tiếng Việt và ưu tiên tính chính xác.\n\n"
        f"Câu hỏi: {question}\n\n"
        "=== Bối cảnh (dùng để trích dẫn) ===\n"
        f"{context.strip() or '(không có bối cảnh)'}\n\n"
        "=== Quan hệ trong đồ thị (dùng để trích dẫn) ===\n"
        f"{graph_section}\n\n"
        "Yêu cầu trả lời:\n"
        "- Chỉ dùng thông tin từ bối cảnh và quan hệ đồ thị; không tự suy diễn.\n"
        "- Mỗi câu hoặc ý chính phải kèm trích dẫn nguồn dạng [source].\n"
        "- Nếu thông tin không đủ, hãy nói rõ.\n"
        f"{guardrail_section}"
    )

    return prompt
