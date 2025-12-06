import textwrap

import pytest

from src.generation.prompt_templates import build_context_graph_prompt, format_graph_triples


def test_format_graph_triples_handles_dicts_and_sequences():
    triples = [
        {"subject": "Thuốc A", "predicate": "điều trị", "object": "Bệnh X"},
        {"head": "Bác sĩ", "relation": "kê đơn", "tail": "Thuốc A"},
        ("Bệnh X", "liên quan", "Triệu chứng Y"),
    ]

    formatted = format_graph_triples(triples)

    expected = textwrap.dedent(
        """
        - **Thuốc A** --điều trị--> Bệnh X
        - **Bác sĩ** --kê đơn--> Thuốc A
        - **Bệnh X** --liên quan--> Triệu chứng Y
        """
    ).strip()

    assert formatted == expected


def test_build_context_graph_prompt_snapshot():
    prompt = build_context_graph_prompt(
        question="Thuốc A có dùng cho bệnh X không?",
        context="Nguồn 1: Thuốc A thường được chỉ định cho bệnh X.",
        graph_triples=[
            {"subject": "Thuốc A", "predicate": "điều trị", "object": "Bệnh X"},
            ("Thuốc A", "tác dụng phụ", "Buồn ngủ"),
        ],
    )

    expected = textwrap.dedent(
        """
        Bạn là trợ lý y khoa, chỉ sử dụng thông tin được cung cấp.
        Trả lời ngắn gọn bằng tiếng Việt và ưu tiên tính chính xác.

        Câu hỏi: Thuốc A có dùng cho bệnh X không?

        === Bối cảnh (dùng để trích dẫn) ===
        Nguồn 1: Thuốc A thường được chỉ định cho bệnh X.

        === Quan hệ trong đồ thị (dùng để trích dẫn) ===
        - **Thuốc A** --điều trị--> Bệnh X
        - **Thuốc A** --tác dụng phụ--> Buồn ngủ

        Yêu cầu trả lời:
        - Chỉ dùng thông tin từ bối cảnh và quan hệ đồ thị; không tự suy diễn.
        - Mỗi câu hoặc ý chính phải kèm trích dẫn nguồn dạng [source].
        - Nếu thông tin không đủ, hãy nói rõ.

        Kiểm tra cuối cùng:
        - Đảm bảo mọi ý quan trọng đều có trích dẫn nguồn từ phần bối cảnh hoặc quan hệ đồ thị.
        - Nếu phát hiện thiếu nguồn hoặc thiếu cite, hãy tự sửa và bổ sung trích dẫn trước khi gửi.
        - Nếu thực sự không đủ nguồn để trả lời, hãy nêu rõ điều đó thay vì suy đoán.
        """
    ).strip()

    assert prompt.strip() == expected


def test_build_context_graph_prompt_can_disable_guardrails():
    prompt = build_context_graph_prompt(
        question="Tôi cần gì?",
        context="",
        graph_triples=[],
        guardrails=False,
    )

    assert "Kiểm tra cuối cùng" not in prompt
    assert "(không có bối cảnh)" in prompt
    assert "(không có dữ liệu quan hệ)" in prompt
