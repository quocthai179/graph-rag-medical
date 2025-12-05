from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Sequence

import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm


@dataclass
class RawSection:
    source: str
    filepath: Path
    section: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\r\f]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+(?=[A-ZÀ-Ỹ0-9“\"(\[])", flags=re.UNICODE)


def split_sentences(paragraph: str) -> List[str]:
    paragraph = clean_text(paragraph)
    if not paragraph:
        return []

    sentences: List[str] = []
    for line in paragraph.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences.extend(_SENT_SPLIT.split(line))

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def semantic_chunk(
    text: str,
    max_chars: int = 900,
    min_chars: int = 200,
    overlap_chars: int = 100,
) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    def flush() -> None:
        if not buffer:
            return
        chunk_text = " ".join(buffer).strip()
        if chunk_text:
            chunks.append(chunk_text)

    idx = 0
    while idx < len(sentences):
        sentence = sentences[idx]
        if buffer_len + len(sentence) + 1 <= max_chars or buffer_len < min_chars:
            buffer.append(sentence)
            buffer_len += len(sentence) + 1
            idx += 1
            continue

        flush()
        if overlap_chars > 0 and chunks:
            tail = chunks[-1][-overlap_chars:]
            tail = re.sub(r"^\S*\s", "", tail)
            buffer = [tail] if tail else []
            buffer_len = len(tail) if tail else 0
        else:
            buffer = []
            buffer_len = 0

    flush()
    return chunks


def parse_pdf(path: Path) -> Iterator[RawSection]:
    reader = PdfReader(path)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        yield RawSection(source=path.stem, filepath=path, section=idx + 1, text=text)


Parser = Callable[[Path], Iterator[RawSection]]


def load_raw_documents(raw_dir: Path) -> List[RawSection]:
    parsers: dict[str, Parser] = {
        ".pdf": parse_pdf,
    }

    sections: List[RawSection] = []
    for path in tqdm(sorted(raw_dir.glob("**/*")), desc="Scanning files"):
        if path.is_dir():
            continue
        parser = parsers.get(path.suffix.lower())
        if parser is None:
            continue
        sections.extend(list(parser(path)))

    return sections


def build_chunks(sections: Sequence[RawSection]) -> List[dict]:
    chunk_rows: List[dict] = []
    for section in sections:
        cleaned = clean_text(section.text)
        if not cleaned:
            continue

        chunks = semantic_chunk(cleaned)
        for chunk_id, chunk in enumerate(chunks):
            chunk_rows.append(
                {
                    "source": section.source,
                    "filepath": str(section.filepath),
                    "page_or_section": section.section,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    return chunk_rows


def save_chunks(chunks: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(chunks)
    frame.to_parquet(output_path, index=False)


def preprocess(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/chunks.parquet"),
) -> None:
    sections = load_raw_documents(raw_dir)
    chunk_rows = build_chunks(sections)

    if chunk_rows:
        save_chunks(chunk_rows, output_path)
        print(f"Processed {len(set(row['filepath'] for row in chunk_rows))} files")
        print(f"Total chunks: {len(chunk_rows)}")
        print(f"Saved to {output_path}")
    else:
        print("No chunks generated. Check input data and parsers.")


if __name__ == "__main__":
    preprocess()
