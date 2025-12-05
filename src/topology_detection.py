"""Topology detection using Gemini for all JSONL files in ``data/pre_json``.

The script batches text chunks, asks Gemini to label ontology nodes and
relationships, and exports unified CSV files for Neo4j consumption.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# ========= CONFIG =========
BASE_INPUT_DIR = Path("data/pre_json")
OUTPUT_DIR = Path("data/topology")
GEMINI_MODEL = "gemini-2.0-flash"
BATCH_SIZE = 10
RATE_LIMIT_SLEEP = 1.0

ALLOWED_CLASSES = [
    "Disease",
    "AnatomicalSite",
    "ImagingFinding",
    "ClinicalSign",
    "LabTest",
    "Pathogen",
    "RiskFactor",
    "Procedure",
    "Treatment",
    "Guideline",
    "ImagingModality",
    "Measurement",
    "Drug",
    "Severity",
]

ALLOWED_REL_TYPES = [
    "hasSite",
    "hasImagingFinding",
    "hasClinicalSign",
    "causedBy",
    "associatedWithRisk",
    "confirmedBy",
    "managedBy",
    "medicinalSuggests",
    "procedureSuggests",
    "findingSuggests",
]

ontology_schema = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "class": {"type": "string", "enum": ALLOWED_CLASSES},
                    "codeSystem": {"type": "string"},
                    "hasCode": {"type": "string"},
                },
                "required": ["label", "class"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_label": {"type": "string"},
                    "start_class": {"type": "string", "enum": ALLOWED_CLASSES},
                    "end_label": {"type": "string"},
                    "end_class": {"type": "string", "enum": ALLOWED_CLASSES},
                    "type": {"type": "string", "enum": ALLOWED_REL_TYPES},
                    "note": {"type": "string"},
                },
                "required": [
                    "start_label",
                    "start_class",
                    "end_label",
                    "end_class",
                    "type",
                ],
            },
        },
    },
    "required": ["nodes", "relationships"],
}

BASE_SYSTEM_PROMPT = """
Bạn là hệ thống trích xuất tri thức có cấu trúc (information extraction) cho một ontology y khoa
về bệnh lý vùng lồng ngực (ngực, phổi, màng phổi, trung thất…).

Nhiệm vụ của bạn:
- Đọc một đoạn văn bản (tiếng Việt, có thể trộn tiếng Anh y khoa).
- Nhận diện các thực thể (entities) thuộc các lớp cho phép.
- Suy luận các quan hệ (relationships) giữa các thực thể đó, nhưng CHỈ trong phạm vi ontology cho trước.
- Trả về KẾT QUẢ DUY NHẤT ở dạng JSON, đúng theo schema đã được cung cấp.

### 1. Lớp (Classes) được phép

Chỉ sử dụng đúng các class sau (tên tiếng Anh chuẩn như dưới):

- Disease          : Bệnh (vd: "Lao phổi", "Viêm phổi", "Tràn khí màng phổi")
- AnatomicalSite   : Vị trí giải phẫu (vd: "Phổi phải", "Thùy trên", "Đỉnh phổi", "Màng phổi")
- ImagingFinding   : Dấu hiệu hình ảnh (vd: "thâm nhiễm", "nốt", "hang", "tràn dịch", "tổn thương mô kẽ")
- ClinicalSign     : Triệu chứng, dấu hiệu lâm sàng (vd: "ho", "sốt", "gầy sút cân", "đau ngực")
- LabTest          : Xét nghiệm (vd: "AFB đờm", "Xpert MTB/RIF", "nuôi cấy MGIT")
- Pathogen         : Tác nhân gây bệnh (vd: "Mycobacterium tuberculosis", "Streptococcus pneumoniae")
- RiskFactor       : Yếu tố nguy cơ (vd: "hút thuốc lá", "HIV", "đái tháo đường", "suy giảm miễn dịch")
- Procedure        : Thủ thuật/Can thiệp (vd: "dẫn lưu màng phổi", "sinh thiết")
- Treatment        : Điều trị (vd: "phác đồ lao 6 tháng", "kháng sinh", "corticoid")
- Guideline        : Hướng dẫn, phác đồ, khuyến cáo (vd: "Phác đồ Lao 2020", "Hướng dẫn chẩn đoán lao")
- ImagingModality  : Kỹ thuật chẩn đoán hình ảnh (vd: "X-quang phổi", "CT ngực", "MRI ngực")
- Measurement      : Đo đạc định lượng (vd: kích thước nốt, giá trị HU, mức độ tràn dịch)
- Drug             : Thuốc chữa trị (vd: Paracetamol)
- Severity         : Mức độ nặng, thang điểm (vd: "nhẹ", "vừa", "nặng", "CURB-65 = 3")

Không tạo class mới ngoài danh sách trên.

### 2. Kiểu quan hệ (Object properties) được phép

Chỉ sử dụng các quan hệ sau, với đúng hướng và kiểu:

- hasSite           : Disease → AnatomicalSite
- hasImagingFinding : Disease → ImagingFinding
- hasClinicalSign   : Disease → ClinicalSign
- causedBy          : Disease → Pathogen
- associatedWithRisk: Disease → RiskFactor
- confirmedBy       : Disease → (LabTest | ImagingModality | Measurement)
- managedBy         : Disease → (Treatment | Guideline)
- medicinalSuggests : Disease -> Drug
- procedureSuggests : Disease -> Procedure
- findingSuggests   : ImagingFinding → Disease

KHÔNG sử dụng tên quan hệ khác, KHÔNG tráo chiều domain-range.

### 3. Yêu cầu trích xuất (rất quan trọng)

1. CHỈ trích xuất những thực thể và quan hệ:
   - xuất hiện rõ ràng trong đoạn văn bản
   - được suy ra trực tiếp, hiển nhiên với ngữ cảnh y khoa trong đoạn đó.
2. KHÔNG được:
   - Bịa thêm bệnh, xét nghiệm, guideline, điều trị không có trong đoạn.
   - Đoán mã ICD-10/SNOMED/RadLex nếu văn bản không ghi rõ.
3. Với mỗi node:
   - "label": dùng tên ngắn, nhất quán. Có thể giữ tiếng Việt cho thuật ngữ chính thức ("Lao phổi"),
     hoặc thêm tiếng Anh nếu quen dùng ("Pulmonary tuberculosis") nhưng gói trong cùng một label rõ nghĩa.
   - "class": chọn đúng một trong các class ở trên.
   - "codeSystem" và "hasCode": chỉ điền nếu trong đoạn có thông tin mã rõ ràng; nếu không, để chuỗi rỗng.
4. Với mỗi relationship:
   - start_label/start_class phải khớp một node hợp lệ.
   - end_label/end_class phải khớp một node hợp lệ.
   - "type" phải là một trong các quan hệ cho phép.
   - "note": có thể ghi ngắn gọn lý do hoặc câu trích tóm tắt; nếu không cần, có thể để chuỗi rỗng.

5. Nếu đoạn văn không chứa tri thức phù hợp với ontology trên:
   - Trả về: "nodes": [], "relationships": [].

### 4. Định dạng trả về

- CHỈ trả về JSON theo đúng schema đã cho:
  {
    "nodes": [ ... ],
    "relationships": [ ... ]
  }
- KHÔNG thêm giải thích, KHÔNG thêm text bên ngoài JSON.
- KHÔNG đổi tên trường, KHÔNG thêm trường mới.

Hãy đọc kỹ đoạn văn bản sau và trích xuất entities + relationships theo đúng yêu cầu.
"""


# ===================== UTILS =====================

def slugify(label: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", label.strip().lower())
    sanitized = re.sub(r"^-+|-+$", "", sanitized)
    return sanitized or "unk"


def class_prefix(cls: str) -> str:
    mapping = {
        "Disease": "D",
        "AnatomicalSite": "AS",
        "ImagingFinding": "IF",
        "ClinicalSign": "CS",
        "LabTest": "LT",
        "Pathogen": "PTH",
        "RiskFactor": "RF",
        "Procedure": "PR",
        "Treatment": "TR",
        "Guideline": "G",
        "ImagingModality": "IM",
        "Measurement": "M",
        "Drug": "DR",
        "Severity": "SV",
    }
    return mapping.get(cls, "X")


def make_node_id(label: str, cls: str) -> str:
    return f"{class_prefix(cls)}:{slugify(label)}"


def batched_iter(source: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    """Yield successive batches from an iterable."""
    iterator = iter(source)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


# ===================== DATA CLASSES =====================

@dataclass
class TopologyAccumulator:
    nodes: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    provenances: Dict[str, dict] = field(default_factory=dict)
    rels: List[dict] = field(default_factory=list)

    def register_provenance(self, entry: dict) -> str:
        doc_id = entry.get("doc_id", "UNKNOWN_DOC")
        page = entry.get("page", 0)
        block_id = entry.get("block_id", 0)
        chunk_id = entry.get("chunk_id", 0)
        key = f"{doc_id}_p{page}_b{block_id}_c{chunk_id}"
        prov_id = f"P:{key}"

        if prov_id not in self.provenances:
            self.provenances[prov_id] = {
                ":ID": prov_id,
                ":LABEL": "Provenance",
                "docId": doc_id,
                "page": page,
                "blockId": block_id,
                "chunkId": chunk_id,
                "md5": entry.get("md5", ""),
                "bbox": json.dumps(entry.get("bbox", [])),
                "charCount": entry.get("char_count", 0),
                "tokenCount": entry.get("token_count_approx", 0),
            }
        return prov_id

    def get_or_create_node(
        self,
        label: str,
        cls: str,
        *,
        code_system: str = "",
        has_code: str = "",
    ) -> str:
        key = (label.strip(), cls)
        if key not in self.nodes:
            node_id = make_node_id(label, cls)
            self.nodes[key] = {
                ":ID": node_id,
                ":LABEL": cls,
                "label": label.strip(),
                "codeSystem": code_system,
                "hasCode": has_code,
            }
        else:
            stored = self.nodes[key]
            if code_system and not stored.get("codeSystem"):
                stored["codeSystem"] = code_system
            if has_code and not stored.get("hasCode"):
                stored["hasCode"] = has_code
        return self.nodes[key][":ID"]

    def append_relationship(
        self,
        start_id: str,
        end_id: str,
        rel_type: str,
        note: str | None = None,
    ) -> None:
        self.rels.append(
            {
                ":START_ID": start_id,
                ":END_ID": end_id,
                ":TYPE": rel_type,
                "note": note or "",
            }
        )

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        nodes_df = pd.DataFrame(list(self.nodes.values()))[[
            ":ID",
            ":LABEL",
            "label",
            "codeSystem",
            "hasCode",
        ]]

        provenance_df = pd.DataFrame(list(self.provenances.values()))[[
            ":ID",
            ":LABEL",
            "docId",
            "page",
            "blockId",
            "chunkId",
            "md5",
            "bbox",
            "charCount",
            "tokenCount",
        ]]

        rels_df = pd.DataFrame(self.rels)
        if rels_df.empty:
            rels_df = pd.DataFrame(columns=[":START_ID", ":END_ID", ":TYPE", "note"])
        else:
            if "note" not in rels_df.columns:
                rels_df["note"] = ""
            rels_df = rels_df[[":START_ID", ":END_ID", ":TYPE", "note"]]

        return nodes_df, provenance_df, rels_df

    def write_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes_df, provenance_df, rels_df = self.to_dataframes()
        nodes_df.to_csv(output_dir / "nodes.csv", index=False)
        provenance_df.to_csv(output_dir / "provenance.csv", index=False)
        rels_df.to_csv(output_dir / "rels.csv", index=False)


# ===================== GEMINI CALLS =====================

def load_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment/.env")
    return genai.Client(api_key=api_key)


def call_gemini_extract_batch(
    client: genai.Client,
    items: List[dict],
) -> List[dict]:
    payload = {
        "task": "batch_extraction",
        "instructions": (
            "For each item, extract ontology-conformant nodes and relationships. "
            "Return a JSON array with the SAME ORDER as input; each element must conform to the provided schema."
        ),
        "items": [{"index": item["index"], "text": item["text"]} for item in items],
    }

    prompt = f"{BASE_SYSTEM_PROMPT}\n\nDỮ LIỆU BATCH (JSON):\n" + json.dumps(payload, ensure_ascii=False)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={"type": "array", "items": ontology_schema},
        ),
    )

    try:
        data = json.loads(resp.text)
        if not isinstance(data, list) or len(data) != len(items):
            logging.warning(
                "Gemini response length mismatch; using empty results for this batch.",
            )
            return [{"nodes": [], "relationships": []} for _ in items]
        return data
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to parse Gemini batch response: %s", exc)
        return [{"nodes": [], "relationships": []} for _ in items]


# ===================== PIPELINE =====================

def process_batch(
    accumulator: TopologyAccumulator,
    batch_lines: List[str],
    client: genai.Client,
) -> None:
    entries: List[dict] = []
    prov_ids: List[str] = []
    items: List[dict] = []

    for line in batch_lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping malformed JSON line.")
            continue

        text = (entry.get("text") or "").strip()
        if not text:
            continue

        prov_id = accumulator.register_provenance(entry)
        prov_ids.append(prov_id)
        entries.append(entry)
        items.append({"index": len(items), "text": text})

    if not entries:
        return

    if RATE_LIMIT_SLEEP > 0:
        time.sleep(RATE_LIMIT_SLEEP)

    batch_results = call_gemini_extract_batch(client, items)
    for idx, result in enumerate(batch_results):
        prov_id = prov_ids[idx]
        chunk_nodes = result.get("nodes", []) or []
        chunk_rels = result.get("relationships", []) or []

        for node in chunk_nodes:
            label = (node.get("label") or "").strip()
            cls = (node.get("class") or "").strip()
            if not label or cls not in ALLOWED_CLASSES:
                continue

            node_id = accumulator.get_or_create_node(
                label=label,
                cls=cls,
                code_system=node.get("codeSystem") or "",
                has_code=node.get("hasCode") or "",
            )
            accumulator.append_relationship(node_id, prov_id, "derivedFrom")

        for rel in chunk_rels:
            st_label = (rel.get("start_label") or "").strip()
            st_class = (rel.get("start_class") or "").strip()
            en_label = (rel.get("end_label") or "").strip()
            en_class = (rel.get("end_class") or "").strip()
            rel_type = (rel.get("type") or "").strip()
            note = rel.get("note", "")

            if not (st_label and st_class and en_label and en_class and rel_type):
                continue
            if rel_type not in ALLOWED_REL_TYPES:
                continue

            start_id = accumulator.get_or_create_node(st_label, st_class)
            end_id = accumulator.get_or_create_node(en_label, en_class)
            accumulator.append_relationship(start_id, end_id, rel_type, note)


def process_file(path: Path, accumulator: TopologyAccumulator, client: genai.Client) -> None:
    logging.info("Processing file: %s", path)
    with path.open("r", encoding="utf-8") as file_obj:
        for batch_lines in tqdm(
            batched_iter(file_obj, BATCH_SIZE),
            desc=f"{path.name} (batch={BATCH_SIZE})",
        ):
            process_batch(accumulator, batch_lines, client)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = load_client()
    accumulator = TopologyAccumulator()

    input_files = sorted(BASE_INPUT_DIR.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No .jsonl files found in {BASE_INPUT_DIR}")

    for input_path in input_files:
        process_file(input_path, accumulator, client)

    accumulator.write_outputs(OUTPUT_DIR)
    logging.info("Wrote topology outputs to %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
