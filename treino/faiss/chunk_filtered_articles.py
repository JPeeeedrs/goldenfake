#!/usr/bin/env python3
"""Generate hierarchical overlapping chunks from filtered Wikipedia articles."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, TextIO


@dataclass
class ChunkState:
    processed: int = 0
    written: int = 0


PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
TOKEN_PATTERN = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply hierarchical chunking with overlap to filtered Wikipedia articles"
    )
    parser.add_argument(
        "articles_path",
        type=Path,
        help="Path to the JSON or JSONL file containing filtered articles",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=Path("chunked_articles.jsonl"),
        help="Destination file for chunked articles in JSON Lines format",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Optional progress file used for incremental saving",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved progress, if available",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Persist progress every N processed articles",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per chunk (recommended range 128-512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Token overlap between consecutive chunks (recommended 50-100)",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=64,
        help="Minimum token threshold before emitting a chunk (tail chunks always kept)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_progress(progress_path: Path) -> Optional[dict]:
    if not progress_path.is_file():
        return None
    with progress_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = progress_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(progress_path)


def detect_json_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "json"
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(2048)
    first = next((char for char in prefix if not char.isspace()), "")
    if first == "{":
        return "jsonl"
    return "json"


def iter_articles(path: Path, fmt: str, skip: int) -> Iterator[dict]:
    processed = 0
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if processed < skip:
                    processed += 1
                    continue
                yield json.loads(line)
                processed += 1
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError("JSON input must be a list when not using JSON Lines format")
        for idx, article in enumerate(payload):
            if idx < skip:
                continue
            yield article


def prepare_output_handle(path: Path, append: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    return path.open(mode, encoding="utf-8")


def split_paragraphs(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT.split(normalized) if part.strip()]
    if not paragraphs and normalized.strip():
        paragraphs = [normalized.strip()]
    return paragraphs


def chunk_paragraph(paragraph: str, max_tokens: int, overlap: int, min_tokens: int) -> List[str]:
    matches = list(TOKEN_PATTERN.finditer(paragraph))
    total_tokens = len(matches)
    if total_tokens == 0:
        return []

    stride = max_tokens - overlap
    if stride <= 0:
        stride = 1

    chunks: List[str] = []
    start_index = 0
    while start_index < total_tokens:
        end_index = min(start_index + max_tokens, total_tokens)
        start_char = matches[start_index].start()
        end_char = matches[end_index - 1].end()
        chunk_text = paragraph[start_char:end_char].strip()
        token_count = end_index - start_index
        if chunk_text:
            if token_count >= min_tokens or end_index >= total_tokens or not chunks:
                chunks.append(chunk_text)
        if end_index >= total_tokens:
            break
        start_index += stride
    return chunks


def chunk_article_text(text: str, max_tokens: int, overlap: int, min_tokens: int) -> List[str]:
    chunks: List[str] = []
    for paragraph in split_paragraphs(text):
        chunks.extend(chunk_paragraph(paragraph, max_tokens, overlap, min_tokens))
    return chunks


def merge_state_with_progress(
    state: ChunkState,
    progress: Optional[dict],
    args: argparse.Namespace,
) -> tuple[ChunkState, int]:
    if not progress:
        return state, 0

    expected_articles = str(args.articles_path.resolve())
    expected_output = str(args.output_path.resolve())

    if progress.get("articles_path") != expected_articles:
        logging.warning("Progress file references a different articles file. Ignoring saved state.")
        return state, 0
    if progress.get("output_path") != expected_output:
        logging.warning("Progress file references a different output path. Ignoring saved state.")
        return state, 0

    if progress.get("max_tokens") != args.max_tokens or progress.get("overlap") != args.overlap:
        logging.warning("Chunk configuration changed. Saved state will be ignored.")
        return state, 0

    state.processed = progress.get("processed", 0)
    state.written = progress.get("written", 0)
    logging.info(
        "Resuming from article index %d with %d chunked results already written.",
        state.processed,
        state.written,
    )
    return state, state.processed


def persist_progress(
    progress_path: Path,
    args: argparse.Namespace,
    state: ChunkState,
    finished: bool = False,
) -> None:
    payload = {
        "articles_path": str(args.articles_path.resolve()),
        "output_path": str(args.output_path.resolve()),
        "processed": state.processed,
        "written": state.written,
        "max_tokens": args.max_tokens,
        "overlap": args.overlap,
        "min_chunk_tokens": args.min_chunk_tokens,
        "finished": finished,
    }
    save_progress(progress_path, payload)


def process_articles(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    if args.save_interval <= 0:
        raise ValueError("save-interval must be greater than zero")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be greater than zero")
    if args.overlap < 0:
        raise ValueError("overlap must be zero or positive")
    if args.min_chunk_tokens < 0:
        raise ValueError("min-chunk-tokens must be zero or positive")
    if args.overlap >= args.max_tokens:
        raise ValueError("overlap must be smaller than max-tokens")

    fmt = detect_json_format(args.articles_path)

    progress_path = args.progress_file or args.output_path.with_suffix(".progress.json")
    if progress_path.exists() and not args.resume:
        progress_path.unlink()

    progress_payload = load_progress(progress_path) if args.resume else None

    state = ChunkState()
    state, skip_count = merge_state_with_progress(state, progress_payload, args)

    append_output = state.written > 0
    output_handle = prepare_output_handle(args.output_path, append=append_output)

    articles_since_save = 0
    try:
        for article in iter_articles(args.articles_path, fmt, skip_count):
            article_id = article.get("id") or article.get("page_id")
            title = article.get("title")
            text = article.get("text")
            if not text:
                state.processed += 1
                articles_since_save += 1
                continue

            chunks = chunk_article_text(text, args.max_tokens, args.overlap, args.min_chunk_tokens)
            if chunks:
                payload = {
                    "id": article_id,
                    "title": title,
                    "chunks": chunks,
                }
                output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                state.written += 1
            state.processed += 1
            articles_since_save += 1

            if articles_since_save >= args.save_interval:
                persist_progress(progress_path, args, state)
                logging.info("Progress saved: %d processed, %d chunked.", state.processed, state.written)
                articles_since_save = 0

        logging.info("Chunking completed: %d processed, %d written.", state.processed, state.written)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Saving progress before exiting.")
        persist_progress(progress_path, args, state)
        raise

    finally:
        output_handle.flush()
        output_handle.close()
        persist_progress(progress_path, args, state, finished=True)


def main() -> None:
    args = parse_args()
    process_articles(args)


if __name__ == "__main__":
    main()
