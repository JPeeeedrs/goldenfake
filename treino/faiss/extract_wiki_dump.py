#!/usr/bin/env python3
"""Utility for extracting cleaned Wikipedia articles from XML dumps."""

from __future__ import annotations

import argparse
import bz2
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, TextIO
from xml.etree.ElementTree import iterparse

from wikiextractor.extract import Extractor


@dataclass
class Article:
    """Represents the minimal information we keep per article."""

    page_id: str
    revid: str
    title: str
    text: str


@dataclass
class ExtractionState:
    """Tracks counters used for chunking and resume support."""

    processed_articles: int = 0
    output_file_index: int = 0
    articles_in_current_file: int = 0
    last_page_id: Optional[str] = None

    @property
    def current_filename(self) -> str:
        return f"wiki_articles_{self.output_file_index:05d}"


# Configure WikiExtractor defaults once so every extractor instance behaves the same.
Extractor.keepLinks = False
Extractor.keepSections = False
Extractor.HtmlFormatting = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cleaned Wikipedia articles from a compressed XML dump using WikiExtractor"
    )
    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to the compressed Wikipedia XML dump (.bz2)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("wiki_extraction_output"),
        help="Directory where the cleaned article files will be written",
    )
    parser.add_argument(
        "-f",
        "--output-format",
        choices=("jsonl", "txt"),
        default="jsonl",
        help="Output format for the extracted articles",
    )
    parser.add_argument(
        "-n",
        "--articles-per-file",
        type=int,
        default=1000,
        help="Number of articles to store in each output file",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Optional path to persist progress. Defaults to <output-dir>/extraction_progress.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved progress if available",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Persist progress every N articles processed",
    )
    parser.add_argument(
        "--url-base",
        default="https://pt.wikipedia.org/wiki/",
        help="Base URL used when constructing article URLs in JSON output",
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


def extract_namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.find("}")]
    return ""


def iter_dump_articles(dump_path: Path) -> Iterator[tuple[str, str, str, str]]:
    if not dump_path.is_file():
        raise FileNotFoundError(f"Dump file not found: {dump_path}")

    with bz2.open(dump_path, "rb") as handle:
        context = iterparse(handle, events=("start", "end"))
        try:
            event, root = next(context)
        except StopIteration as exc:  # pragma: no cover - empty dump guard
            raise RuntimeError("Dump appears to be empty") from exc

        if event != "start":
            raise RuntimeError("Unexpected XML stream state: root element not found")

        namespace = extract_namespace(root.tag)
        ns = f"{{{namespace}}}" if namespace else ""
        page_tag = f"{ns}page"
        title_tag = f"{ns}title"
        ns_tag = f"{ns}ns"
        id_tag = f"{ns}id"
        revision_tag = f"{ns}revision"
        redirect_tag = f"{ns}redirect"
        text_tag = f"{ns}text"

        for event, elem in context:
            if event != "end" or elem.tag != page_tag:
                continue

            title_el = elem.find(title_tag)
            namespace_el = elem.find(ns_tag)
            redirect_el = elem.find(redirect_tag)
            page_id_el = elem.find(id_tag)
            revision_el = elem.find(revision_tag)

            if (
                title_el is None
                or namespace_el is None
                or page_id_el is None
                or revision_el is None
                or namespace_el.text != "0"
                or redirect_el is not None
            ):
                elem.clear()
                continue

            text_el = revision_el.find(text_tag)
            revid_el = revision_el.find(id_tag)

            if text_el is None or text_el.text is None:
                elem.clear()
                continue

            page_id = page_id_el.text or ""
            revid = revid_el.text if revid_el is not None and revid_el.text else ""
            title = title_el.text or ""
            raw_text = text_el.text

            yield page_id, revid, title, raw_text
            elem.clear()
            root.clear()


def clean_article_text(page_id: str, revid: str, title: str, raw_text: str, url_base: str) -> Article:
    page_lines = raw_text.splitlines(keepends=True)
    extractor = Extractor(page_id, revid, url_base, title, page_lines)

    cleaned_lines = extractor.clean_text(raw_text, html_safe=False)
    cleaned_text = "\n".join(cleaned_lines).strip()

    return Article(
        page_id=page_id,
        revid=revid,
        title=title,
        text=cleaned_text,
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
    try:
        temp_path.replace(progress_path)
    except PermissionError:
        if progress_path.exists():
            progress_path.unlink()
        temp_path.replace(progress_path)


def open_output_file(output_dir: Path, state: ExtractionState, output_format: str, append: bool) -> tuple[Path, TextIO]:
    extension = "jsonl" if output_format == "jsonl" else "txt"
    file_path = output_dir / f"{state.current_filename}.{extension}"
    mode = "a" if append else "w"
    handle = file_path.open(mode, encoding="utf-8")
    return file_path, handle


def write_article(handle, article: Article, output_format: str, url_base: str) -> None:
    if output_format == "jsonl":
        payload = {
            "id": article.page_id,
            "revid": article.revid,
            "title": article.title,
            "url": f"{url_base}?curid={article.page_id}",
            "text": article.text,
        }
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        handle.write(f"ID: {article.page_id}\n")
        handle.write(f"Revision: {article.revid}\n")
        handle.write(f"Title: {article.title}\n")
        handle.write("Text:\n")
        handle.write(article.text + "\n")
        handle.write("-" * 40 + "\n")


def merge_state_with_progress(state: ExtractionState, progress: Optional[dict], args: argparse.Namespace) -> tuple[ExtractionState, int]:
    if not progress:
        return state, 0

    expected_dump = str(args.dump_path.resolve())
    if progress.get("dump_path") != expected_dump:
        logging.warning("Progress file references a different dump. Ignoring saved state.")
        return state, 0

    if progress.get("articles_per_file") != args.articles_per_file:
        logging.warning("Articles-per-file mismatch. Saved state will be ignored.")
        return state, 0

    state.processed_articles = progress.get("processed_articles", 0)
    state.output_file_index = progress.get("output_file_index", 0)
    state.articles_in_current_file = progress.get("articles_in_current_file", 0)
    state.last_page_id = progress.get("last_page_id")

    skip_count = state.processed_articles

    logging.info(
        "Resuming from article %s (file index %d, %d articles already written in current file).",
        state.last_page_id,
        state.output_file_index,
        state.articles_in_current_file,
    )

    return state, skip_count


def process_dump(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    if args.articles_per_file <= 0:
        raise ValueError("articles-per-file must be greater than zero")
    if args.save_interval <= 0:
        raise ValueError("save-interval must be greater than zero")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = args.progress_file or args.output_dir / "extraction_progress.json"

    if progress_path.exists() and not args.resume:
        progress_path.unlink()

    progress_data = load_progress(progress_path) if args.resume else None

    state = ExtractionState()
    state, skip_remaining = merge_state_with_progress(state, progress_data, args)

    append_mode = state.articles_in_current_file > 0
    output_path, output_handle = open_output_file(args.output_dir, state, args.output_format, append_mode)
    logging.info("Writing output to %s", output_path)

    articles_since_save = 0

    try:
        for page_id, revid, title, raw_text in iter_dump_articles(args.dump_path):
            if skip_remaining > 0:
                skip_remaining -= 1
                continue

            article = clean_article_text(page_id, revid, title, raw_text, args.url_base)
            if not article.text:
                continue

            if state.articles_in_current_file >= args.articles_per_file:
                output_handle.close()
                state.output_file_index += 1
                state.articles_in_current_file = 0
                output_path, output_handle = open_output_file(
                    args.output_dir,
                    state,
                    args.output_format,
                    append=False,
                )
                logging.info("Created new output file: %s", output_path)

            write_article(output_handle, article, args.output_format, args.url_base)
            state.articles_in_current_file += 1
            state.processed_articles += 1
            state.last_page_id = article.page_id
            articles_since_save += 1

            if articles_since_save >= args.save_interval:
                payload = {
                    "dump_path": str(args.dump_path.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "output_format": args.output_format,
                    "articles_per_file": args.articles_per_file,
                    "processed_articles": state.processed_articles,
                    "output_file_index": state.output_file_index,
                    "articles_in_current_file": state.articles_in_current_file,
                    "last_page_id": state.last_page_id,
                }
                save_progress(progress_path, payload)
                articles_since_save = 0
                logging.info(
                    "Progress saved: %d articles processed (file %d).",
                    state.processed_articles,
                    state.output_file_index,
                )

        logging.info("Extraction completed: %d articles processed in total.", state.processed_articles)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Saving progress before exiting.")
        payload = {
            "dump_path": str(args.dump_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "output_format": args.output_format,
            "articles_per_file": args.articles_per_file,
            "processed_articles": state.processed_articles,
            "output_file_index": state.output_file_index,
            "articles_in_current_file": state.articles_in_current_file,
            "last_page_id": state.last_page_id,
        }
        save_progress(progress_path, payload)
        raise

    finally:
        output_handle.flush()
        output_handle.close()

        payload = {
            "dump_path": str(args.dump_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "output_format": args.output_format,
            "articles_per_file": args.articles_per_file,
            "processed_articles": state.processed_articles,
            "output_file_index": state.output_file_index,
            "articles_in_current_file": state.articles_in_current_file,
            "last_page_id": state.last_page_id,
            "is_finished": True,
        }
        save_progress(progress_path, payload)


def main() -> None:
    args = parse_args()
    process_dump(args)


if __name__ == "__main__":
    main()
