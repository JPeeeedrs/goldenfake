#!/usr/bin/env python3
"""Expand Wikipedia category lists by fetching subcategories from the Portuguese MediaWiki API."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_API_ENDPOINT = "https://pt.wikipedia.org/w/api.php"
USER_AGENT = "GoldenFakeCategoryExpander/1.0 (+https://github.com/JPeeeedrs/goldenfake)"


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
        description="Expand Wikipedia categories with their subcategories")
    parser.add_argument("input_json", type=Path,
                        help="Path to the JSON file describing category themes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temas_expandidos.json"),
        help="Destination JSON file to store the expanded categories",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API requests to avoid rate limiting",
    )
    parser.add_argument(
        "--api-endpoint",
        default=DEFAULT_API_ENDPOINT,
        help="MediaWiki API endpoint (for testing or alternative mirrors)",
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


def load_category_map(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise TypeError(
            "Input JSON must be an object mapping theme names to category lists")

    categories: Dict[str, List[str]] = {}
    for theme, values in payload.items():
        if isinstance(values, str):
            items = [values]
        elif isinstance(values, Sequence):
            items = [str(item)
                     for item in values if isinstance(item, (str, bytes))]
        else:
            raise TypeError(
                f"Expected list of strings for theme '{theme}', got {type(values)!r}")
        categories[str(theme)] = items
    return categories


def normalize_category_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return ""
    if cleaned.lower().startswith("categoria:"):
        cleaned = cleaned.split(":", 1)[1]
    cleaned = cleaned.replace("_", " ")
    return " ".join(cleaned.split())


def build_category_title(name: str) -> str:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Category name cannot be empty")
    if trimmed.lower().startswith("categoria:"):
        return trimmed
    return f"Categoria:{trimmed.replace(' ', '_')}"


def fetch_subcategories(
    session: requests.Session,
    category: str,
    delay: float,
    endpoint: str,
    retries: int = 3,
) -> List[str]:
    title = build_category_title(category)
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": title,
        "cmtype": "subcat",
        "cmlimit": "max",
    }

    subcategories: List[str] = []
    cmcontinue: str | None = None

    while True:
        request_params = params.copy()
        if cmcontinue:
            request_params["cmcontinue"] = cmcontinue

        attempt = 0
        while True:
            attempt += 1
            try:
                if delay > 0:
                    time.sleep(delay)
                response = session.get(
                    endpoint, params=request_params, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Failed to fetch subcategories for {title}: {exc}") from exc
                logging.warning(
                    "Erro ao buscar subcategorias de %s (tentativa %d/%d): %s",
                    category,
                    attempt,
                    retries,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)
            except requests.RequestException as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Failed to fetch subcategories for {title}: {exc}") from exc
                logging.warning(
                    "Erro inesperado ao buscar subcategorias de %s (tentativa %d/%d): %s",
                    category,
                    attempt,
                    retries,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

        query = data.get("query", {})
        members = query.get("categorymembers", [])
        for item in members:
            title_value = item.get("title")
            if isinstance(title_value, str):
                subcategories.append(title_value)

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue") if isinstance(
            cont, Mapping) else None
        if not cmcontinue:
            break

    return subcategories


def expand_categories(
    session: requests.Session,
    input_map: Mapping[str, Iterable[str]],
    delay: float,
    endpoint: str,
) -> Dict[str, List[str]]:
    expanded: Dict[str, List[str]] = {}
    for theme, categories in input_map.items():
        unique: Set[str] = set()
        processed: Set[str] = set()
        for category in categories:
            normalized = normalize_category_name(str(category))
            if not normalized:
                continue
            unique.add(normalized)

            if normalized in processed:
                continue
            processed.add(normalized)

            try:
                fetched = fetch_subcategories(
                    session, category, delay, endpoint)
            except RuntimeError as exc:
                logging.error(
                    "Falha ao buscar subcategorias de %s: %s", normalized, exc)
                continue

            for subcat in fetched:
                normalized_sub = normalize_category_name(subcat)
                if normalized_sub:
                    unique.add(normalized_sub)
            logging.info(
                "Buscando subcategorias de %s... %d encontradas",
                normalized,
                len(fetched),
            )

        expanded[theme] = sorted(unique)
    return expanded


def persist_output(path: Path, payload: Mapping[str, Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def calculate_totals(payload: Mapping[str, Sequence[str]]) -> int:
    return sum(len(values) for values in payload.values())


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    input_map = load_category_map(args.input_json)
    original_total = calculate_totals(input_map)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        expanded = expand_categories(
            session, input_map, args.delay, args.api_endpoint)
    expanded_total = calculate_totals(expanded)

    persist_output(args.output, expanded)

    logging.info(
        "%d categorias originais expandidas para %d categorias totais.",
        original_total,
        expanded_total,
    )


if __name__ == "__main__":
    main()
