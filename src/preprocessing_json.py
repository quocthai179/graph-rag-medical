import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List

import fitz

MAX_CHARS = 1000
MIN_CHARS = 200
OVERLAP_CHARS = 120


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+(?=[A-ZÀ-Ỹ0-9“\"(\[])", flags=re.UNICODE)


def simple_sentence_split(paragraph: str) -> List[str]:
    paragraph = clean_text(paragraph)
    if not paragraph:
        return []

    parts: List[str] = []
    for segment in paragraph.splitlines():
        segment = segment.strip()
        if not segment:
            continue
        parts.extend(_SENT_SPLIT.split(segment))
    return [sentence.strip() for sentence in parts if sentence.strip()]


def sentences_to_passages(
    sentences: Iterable[str],
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[str]:
    passages: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    def flush_buffer() -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if text:
            passages.append(text)

    index = 0
    sentences_list = list(sentences)
    while index < len(sentences_list):
        sentence = sentences_list[index]
        if buffer_len + len(sentence) + 1 <= max_chars or buffer_len < min_chars:
            buffer.append(sentence)
            buffer_len += len(sentence) + 1
            index += 1
            continue

        flush_buffer()
        if overlap_chars > 0 and passages:
            tail = passages[-1][-overlap_chars:]
            tail = re.sub(r"^\S*\s", "", tail)
            if tail:
                buffer = [tail]
                buffer_len = len(tail)
            else:
                buffer = []
                buffer_len = 0
        else:
            buffer = []
            buffer_len = 0

    flush_buffer()
    return passages


def approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def md5_of(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_blocks(page) -> List[dict]:
    blocks = []
    try:
        for block_id, block in enumerate(page.get_text("blocks")):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, block_text = block[:5]
            text = clean_text(block_text)
            if not text:
                continue
            blocks.append(
                {
                    "block_id": block_id,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": text,
                }
            )
    except Exception:
        text = clean_text(page.get_text())
        if text:
            blocks.append({"block_id": 0, "bbox": None, "text": text})

    filtered_blocks = []
    for block in blocks:
        text = block["text"]
        if len(text) <= 3 and re.fullmatch(r"[\divx]+", text.lower()):
            continue
        filtered_blocks.append(block)
    return filtered_blocks


def write_jsonl(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_pdf(pdf_path: Path) -> List[dict]:
    document = fitz.open(pdf_path)
    doc_id = pdf_path.stem

    all_records: List[dict] = []
    for page_number in range(len(document)):
        page = document[page_number]
        blocks = extract_blocks(page)

        for block in blocks:
            sentences = simple_sentence_split(block["text"])
            if not sentences:
                continue
            passages = sentences_to_passages(sentences)

            for chunk_id, passage in enumerate(passages):
                all_records.append(
                    {
                        "doc_id": doc_id,
                        "page": page_number + 1,
                        "block_id": block["block_id"],
                        "chunk_id": chunk_id,
                        "text": passage,
                        "bbox": block["bbox"],
                        "char_count": len(passage),
                        "token_count_approx": approx_token_count(passage),
                        "md5": md5_of(
                            f"{doc_id}|{page_number + 1}|{block['block_id']}|{chunk_id}|{passage[:50]}"
                        ),
                    }
                )

    return all_records


def process_folder(input_folder: Path, output_file: Path) -> None:
    records: List[dict] = []
    for pdf_file in sorted(input_folder.glob("*.pdf")):
        records.extend(process_pdf(pdf_file))

    if records:
        write_jsonl(records, output_file)
        print(f"Saved {len(records)} passages to {output_file}")
    else:
        print(f"No PDF files found in {input_folder}, skipping...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF documents to passages JSONL")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw data folders")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Directory to store output JSONL files")
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")

    for folder in sorted(args.raw_dir.iterdir()):
        if not folder.is_dir():
            continue
        output_file = args.output_dir / f"{folder.name}.passages.jsonl"
        process_folder(folder, output_file)


if __name__ == "__main__":
    main()
