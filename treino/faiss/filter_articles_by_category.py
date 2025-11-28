#!/usr/bin/env python3
"""Filter extracted Wikipedia articles by category membership using the MediaWiki API."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, TextIO, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class FilterState:
    processed: int = 0
    written: int = 0


DEFAULT_API_ENDPOINT = "https://pt.wikipedia.org/w/api.php"
CATEGORY_NAMESPACES = {"category", "categoria"}
USER_AGENT = "GoldenFakeFilter/1.0 (+https://github.com/JPeeeedrs/goldenfake)"


def create_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter extracted Wikipedia articles using a list of relevant categories"
    )
    parser.add_argument(
        "articles_path",
        type=Path,
        help="Path to the JSON/JSONL file containing extracted articles",
    )
    parser.add_argument(
        "categories_path",
        type=Path,
        help="Path to the JSON file listing the relevant categories",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=Path("filtered_articles.jsonl"),
        help="Destination file for filtered articles in JSON Lines format",
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
        "--batch-size",
        type=int,
        default=20,
        help="Number of article titles to query per API request (max 50)",
    )
    parser.add_argument(
        "--api-endpoint",
        default=DEFAULT_API_ENDPOINT,
        help="MediaWiki API endpoint to use for category lookups",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden categories when evaluating matches",
    )
    parser.add_argument(
        "--extra-category",
        action="append",
        default=None,
        help="Additional category to include without editing the JSON list",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API requests to avoid rate limiting",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries when the API responds with errors",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_relevant_categories(path: Path, extras: Optional[Sequence[str]]) -> Set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Categories JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    def extract_strings(payload: object) -> List[str]:
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, list):
            items: List[str] = []
            for element in payload:
                items.extend(extract_strings(element))
            return items
        return []

    if isinstance(data, dict):
        if "categories" in data:
            items = extract_strings(data["categories"])
        else:
            items = []
            for value in data.values():
                items.extend(extract_strings(value))
    elif isinstance(data, list):
        items = extract_strings(data)
    else:
        raise TypeError(
            "Categories JSON must be a list or contain a 'categories' list")

    categories = {normalize_category_name(item)
                  for item in items if isinstance(item, str)}
    if extras:
        categories.update(normalize_category_name(item) for item in extras)
    if not categories:
        raise ValueError(
            "No categories provided after processing the input JSON")
    return categories


def normalize_category_name(name: str) -> str:
    cleaned = name.strip().replace("_", " ")
    if ":" in cleaned:
        prefix, remainder = cleaned.split(":", 1)
        if prefix.lower() in CATEGORY_NAMESPACES:
            cleaned = remainder
    return cleaned.strip().casefold()


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


def collect_article_sources(path: Path) -> List[Tuple[Path, str]]:
    if path.is_file():
        return [(path, detect_json_format(path))]
    if path.is_dir():
        files: List[Tuple[Path, str]] = []
        for child in sorted(path.iterdir()):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if suffix not in {".json", ".jsonl", ".jsonlines", ".ndjson"}:
                continue
            lower_name = child.name.lower()
            if lower_name.endswith(".progress.json") or "progress" in lower_name:
                continue
            try:
                fmt = detect_json_format(child)
            except Exception:
                continue
            files.append((child, fmt))
        if not files:
            raise FileNotFoundError(
                f"No JSON or JSONL article files found inside directory: {path}"
            )
        return files
    raise FileNotFoundError(f"Articles path not found: {path}")


def iter_articles(sources: Sequence[Tuple[Path, str]], skip: int) -> Iterator[dict]:
    processed = 0
    for path, fmt in sources:
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
                raise TypeError(
                    f"JSON input must be a list when not using JSON Lines format (file: {path})"
                )
            for article in payload:
                if processed < skip:
                    processed += 1
                    continue
                yield article
                processed += 1


def prepare_output_handle(path: Path, append: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    return path.open(mode, encoding="utf-8")


def fetch_categories(
    session: requests.Session,
    endpoint: str,
    titles: Sequence[str],
    include_hidden: bool,
    request_delay: float,
    retries: int = 3,
) -> Dict[str, List[str]]:
    if not titles:
        return {}

    params = {
        "action": "query",
        "format": "json",
        "prop": "categories",
        "titles": "|".join(titles),
        "cllimit": "max",
    }
    if not include_hidden:
        params["clshow"] = "!hidden"

    results: Dict[str, List[str]] = {title: [] for title in titles}
    normalization: Dict[str, str] = {}
    delay = max(0.0, request_delay)

    request_params = params.copy()
    while True:
        payload: Optional[dict] = None
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if delay:
                    time.sleep(delay)
                response = session.get(
                    endpoint, params=request_params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                break
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status in {403, 429} and attempt < retries - 1:
                    backoff = max(delay, 1.0) * (attempt + 1)
                    logging.warning(
                        "API returned status %s. Backing off %.1fs before retrying.",
                        status,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Failed to query categories for batch: {exc}") from exc
                logging.warning(
                    "Retrying category fetch after HTTP error: %s", exc)
            except requests.RequestException as exc:
                last_error = exc
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Failed to query categories for batch: {exc}") from exc
                logging.warning("Retrying category fetch after error: %s", exc)
        else:
            assert last_error is not None
            raise RuntimeError(
                f"Failed to query categories for batch: {last_error}") from last_error

        assert payload is not None

        query = payload.get("query", {})
        for item in query.get("normalized", []):
            normalization[item.get("to", "")] = item.get("from", "")
        for item in query.get("redirects", []):
            normalization[item.get("to", "")] = item.get("from", "")

        for page in query.get("pages", {}).values():
            page_title = page.get("title", "")
            original_title = normalization.get(page_title, page_title)
            buckets = results.setdefault(original_title, [])
            for cat in page.get("categories", []) or []:
                title = cat.get("title", "")
                if title:
                    buckets.append(title)

        cont = payload.get("continue")
        if not cont:
            break
        request_params.update(cont)

    return results


def should_keep_article(categories: Sequence[str], relevant: Set[str]) -> List[str]:
    matched: List[str] = []
    for category in categories:
        norm = normalize_category_name(category)
        if norm in relevant:
            matched.append(category)
    return matched


def merge_state_with_progress(
    state: FilterState,
    progress: Optional[dict],
    args: argparse.Namespace,
    relevant_categories: Set[str],
) -> tuple[FilterState, int]:
    if not progress:
        return state, 0

    expected_articles = str(args.articles_path.resolve())
    expected_categories = str(args.categories_path.resolve())
    expected_output = str(args.output_path.resolve())

    if progress.get("articles_path") != expected_articles:
        logging.warning(
            "Progress file references a different articles file. Ignoring saved state.")
        return state, 0
    if progress.get("categories_path") != expected_categories:
        logging.warning(
            "Progress file references a different categories file. Ignoring saved state.")
        return state, 0
    if progress.get("output_path") != expected_output:
        logging.warning(
            "Progress file references a different output path. Ignoring saved state.")
        return state, 0

    saved_categories = {normalize_category_name(
        cat) for cat in progress.get("relevant_categories", [])}
    if saved_categories and saved_categories != relevant_categories:
        logging.warning(
            "Relevant categories have changed. Saved state will be ignored.")
        return state, 0

    state.processed = progress.get("processed", 0)
    state.written = progress.get("written", 0)
    logging.info(
        "Resuming from article index %d with %d filtered results already written.",
        state.processed,
        state.written,
    )
    return state, state.processed


def process_articles(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    if args.save_interval <= 0:
        raise ValueError("save-interval must be greater than zero")
    if args.batch_size <= 0 or args.batch_size > 50:
        raise ValueError("batch-size must be between 1 and 50")
    if args.request_delay < 0:
        raise ValueError("request-delay must be zero or positive")
    if args.max_retries < 1:
        raise ValueError("max-retries must be at least one")

    relevant_categories = load_relevant_categories(
        args.categories_path, args.extra_category)

    progress_path = args.progress_file or args.output_path.with_suffix(
        ".progress.json")
    if progress_path.exists() and not args.resume:
        progress_path.unlink()

    progress_payload = load_progress(progress_path) if args.resume else None

    state = FilterState()
    state, skip_count = merge_state_with_progress(
        state, progress_payload, args, relevant_categories)

    sources = collect_article_sources(args.articles_path)
    append_output = state.written > 0
    output_handle = prepare_output_handle(
        args.output_path, append=append_output)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    articles_since_save = 0
    try:
        batch_titles: List[str] = []
        batch_articles: List[dict] = []

        for article in iter_articles(sources, skip_count):
            title = str(article.get("title") or "").strip()
            if not title:
                state.processed += 1
                articles_since_save += 1
                continue

            batch_titles.append(title)
            batch_articles.append(article)

            if len(batch_titles) >= args.batch_size:
                state, articles_since_save = handle_batch(
                    session,
                    args,
                    state,
                    relevant_categories,
                    batch_titles,
                    batch_articles,
                    output_handle,
                    articles_since_save,
                )
                batch_titles.clear()
                batch_articles.clear()

        if batch_titles:
            state, articles_since_save = handle_batch(
                session,
                args,
                state,
                relevant_categories,
                batch_titles,
                batch_articles,
                output_handle,
                articles_since_save,
            )

        logging.info("Filtering completed: %d processed, %d kept.",
                     state.processed, state.written)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Saving progress before exiting.")
        persist_progress(progress_path, args, state, relevant_categories)
        raise

    finally:
        output_handle.flush()
        output_handle.close()
        persist_progress(progress_path, args, state,
                         relevant_categories, finished=True)


def handle_batch(
    session: requests.Session,
    args: argparse.Namespace,
    state: FilterState,
    relevant_categories: Set[str],
    titles: Sequence[str],
    articles: Sequence[dict],
    output_handle: TextIO,
    articles_since_save: int,
) -> tuple[FilterState, int]:
    lookup = fetch_categories(
        session,
        args.api_endpoint,
        titles,
        args.include_hidden,
        args.request_delay,
        args.max_retries,
    )
    for title, article in zip(titles, articles):
        categories = lookup.get(title, [])
        matched = should_keep_article(categories, relevant_categories)
        state.processed += 1
        articles_since_save += 1
        if matched:
            payload = dict(article)
            payload["matched_categories"] = matched
            payload["all_categories"] = categories
            output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            state.written += 1
    output_handle.flush()

    if articles_since_save >= args.save_interval:
        progress_path = args.progress_file or args.output_path.with_suffix(
            ".progress.json")
        persist_progress(progress_path, args, state, relevant_categories)
        logging.info("Progress saved: %d processed, %d kept.",
                     state.processed, state.written)
        articles_since_save = 0
    return state, articles_since_save


def persist_progress(
    progress_path: Path,
    args: argparse.Namespace,
    state: FilterState,
    relevant_categories: Set[str],
    finished: bool = False,
) -> None:
    payload = {
        "articles_path": str(args.articles_path.resolve()),
        "categories_path": str(args.categories_path.resolve()),
        "output_path": str(args.output_path.resolve()),
        "processed": state.processed,
        "written": state.written,
        "relevant_categories": sorted(relevant_categories),
        "finished": finished,
    }
    save_progress(progress_path, payload)


def main() -> None:
    args = parse_args()
    process_articles(args)


if __name__ == "__main__":
    main()
