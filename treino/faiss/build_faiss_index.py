#!/usr/bin/env python3
"""Build and query a FAISS cosine-similarity index from precomputed embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class BuildState:
    processed_files: List[str]
    total_vectors: int


DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SAVE_INTERVAL = 20
PROGRESS_FILENAME = "faiss_build.progress.json"
CONFIG_FILENAME = "faiss_config.json"
METADATA_FILENAME = "faiss_metadata.json"
INDEX_FILENAME = "faiss_index.bin"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utilities for building and querying a FAISS index over Wikipedia embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Construct or extend a FAISS index from embedding files")
    build_parser.add_argument(
        "embeddings_dir",
        type=Path,
        help="Directory containing .npy or .pt embedding files (one per article)",
    )
    build_parser.add_argument(
        "chunks_path",
        type=Path,
        help="Path to the chunked articles JSON/JSONL used to provide metadata previews",
    )
    build_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("vector_store"),
        help="Directory where the FAISS index and metadata will be stored",
    )
    build_parser.add_argument(
        "--progress-file",
        type=Path,
        help="Optional path for incremental progress tracking",
    )
    build_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted indexing run using existing progress and index files",
    )
    build_parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_SAVE_INTERVAL,
        help="Number of files to process before persisting progress and the partial index",
    )
    build_parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    build_npz = subparsers.add_parser(
        "build-from-npz",
        help="Build a FAISS index directly from a consolidated NPZ (per-id arrays) and a metadata JSON",
    )
    build_npz.add_argument("npz_path", type=Path, help="Path to the consolidated embeddings .npz (keys are ids or includes an 'embeddings' array)")
    build_npz.add_argument("metadata_json", type=Path, help="Path to metadata_final.json mapping ids -> info (must include 'chunks' or chunk_count)")
    build_npz.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("vector_store"),
        help="Directory where the FAISS index and metadata will be stored",
    )
    build_npz.add_argument(
        "--progress-file",
        type=Path,
        help="Optional path for incremental progress tracking",
    )
    build_npz.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run using existing progress and index files",
    )
    build_npz.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_SAVE_INTERVAL,
        help="Number of ids to process before persisting progress and the partial index",
    )
    build_npz.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    query_parser = subparsers.add_parser("query", help="Query an existing FAISS index with new text")
    query_parser.add_argument(
        "query_text",
        help="Text to search against the FAISS index",
    )
    query_parser.add_argument(
        "-i",
        "--index-path",
        type=Path,
        default=Path("vector_store") / INDEX_FILENAME,
        help="Path to the saved FAISS index (.bin)",
    )
    query_parser.add_argument(
        "-m",
        "--metadata-path",
        type=Path,
        default=Path("vector_store") / METADATA_FILENAME,
        help="Path to the metadata JSON produced during index construction",
    )
    query_parser.add_argument(
        "-c",
        "--config-path",
        type=Path,
        default=Path("vector_store") / CONFIG_FILENAME,
        help="Path to the index configuration JSON",
    )
    query_parser.add_argument(
        "--model-name",
        default=None,
        help="SentenceTransformer model to use for query embeddings. Defaults to the one stored in config",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar articles to return",
    )
    query_parser.add_argument(
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


def iter_chunks(path: Path, fmt: str) -> Iterator[dict]:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError("Chunk JSON (non-JSONL) must be a list of article objects")
        for article in payload:
            yield article


def load_chunk_metadata(chunks_path: Path) -> Dict[str, dict]:
    fmt = detect_json_format(chunks_path)
    metadata: Dict[str, dict] = {}
    for article in iter_chunks(chunks_path, fmt):
        article_id = str(article.get("id") or article.get("page_id") or "")
        if not article_id:
            continue
        chunks = article.get("chunks") or []
        preview = ""
        if isinstance(chunks, list) and chunks:
            preview = str(chunks[0])[:300]
        metadata[article_id] = {
            "id": article_id,
            "title": article.get("title", ""),
            "preview": preview,
            "chunk_count": len(chunks),
        }
    return metadata


def list_embedding_files(directory: Path) -> List[Path]:
    files = sorted(
        [path for path in directory.iterdir() if path.suffix.lower() in {".npy", ".pt"}]
    )
    if not files:
        raise FileNotFoundError(f"No embedding files found in {directory}")
    return files


def load_embedding_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        tensor = torch.load(path)
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            raise TypeError(f"Unexpected object stored in {path}: {type(tensor)!r}")
    if array.ndim == 1:
        array = array[np.newaxis, :]
    return array.astype("float32", copy=False)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return embeddings / norms


def read_progress(progress_path: Path) -> Optional[BuildState]:
    if not progress_path.is_file():
        return None
    with progress_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    processed_files = payload.get("processed_files", [])
    total_vectors = payload.get("total_vectors", 0)
    return BuildState(processed_files=processed_files, total_vectors=total_vectors)


def write_progress(progress_path: Path, state: BuildState) -> None:
    payload = {
        "processed_files": state.processed_files,
        "total_vectors": state.total_vectors,
    }
    temp_path = progress_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(progress_path)


def load_existing_index(index_path: Path, metadata_path: Path) -> Tuple[faiss.IndexFlatIP, List[dict]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    index = faiss.read_index(str(index_path))
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = []
    return index, metadata


def save_metadata(metadata_path: Path, metadata: Sequence[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = metadata_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(list(metadata), handle, ensure_ascii=False, indent=2)
    temp_path.replace(metadata_path)


def save_config(config_path: Path, model_name: str, dim: int, count: int) -> None:
    payload = {
        "model_name": model_name,
        "dim": dim,
        "index_type": "IndexFlatIP",
        "count": count,
        "normalize": True,
    }
    temp_path = config_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(config_path)


def _sorted_ids(meta: Dict[str, dict]) -> List[str]:
    def keyfun(k: str):
        try:
            return int(k)
        except Exception:
            return k
    return sorted(meta.keys(), key=keyfun)


def build_index_from_npz(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    if args.save_interval <= 0:
        raise ValueError("save-interval must be greater than zero")

    output_dir: Path = args.output_dir
    progress_path = args.progress_file or (output_dir / PROGRESS_FILENAME)
    index_path = output_dir / INDEX_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    config_path = output_dir / CONFIG_FILENAME

    output_dir.mkdir(parents=True, exist_ok=True)

    with args.metadata_json.open("r", encoding="utf-8") as fh:
        meta_map: Dict[str, dict] = json.load(fh)

    state = read_progress(progress_path) if args.resume else None
    if state is None:
        state = BuildState(processed_files=[], total_vectors=0)
    processed_set = set(state.processed_files)

    if args.resume and index_path.exists():
        index, metadata = load_existing_index(index_path, metadata_path)
        logging.info("Loaded existing index with %d vectors", index.ntotal)
    else:
        index = None
        metadata = []

    files_added_since_save = 0

    # Load NPZ lazily
    npz = np.load(args.npz_path)
    keys = list(npz.files)
    use_per_id = all(k in meta_map for k in keys)

    if not use_per_id:
        # consolidated array
        arr_name = "embeddings" if "embeddings" in keys else ("arr_0" if "arr_0" in keys else None)
        if arr_name is None:
            raise RuntimeError(f"NPZ does not contain an 'embeddings' or 'arr_0' array. Keys: {keys}")
        emb = np.array(npz[arr_name])
        if emb.ndim == 1:
            emb = emb[None, :]
        if "ids" in keys:
            ids = [str(x) for x in npz["ids"]]
        else:
            logging.warning("NPZ has no 'ids'; assuming order matches sorted ids from metadata")
            ids = _sorted_ids(meta_map)
        if len(ids) != emb.shape[0]:
            raise RuntimeError(f"Mismatch: {len(ids)} ids for {emb.shape[0]} embeddings")

        # iterate in order
        for _id, vec in zip(ids, emb):
            if _id in processed_set:
                continue
            arr = np.array(vec, dtype="float32")
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = normalize_embeddings(arr)

            if index is None:
                index = faiss.IndexFlatIP(arr.shape[1])
                logging.info("Created IndexFlatIP with dimension %d", arr.shape[1])
            if arr.shape[1] != index.d:
                raise ValueError(f"Embedding dimension mismatch for id {_id}: expected {index.d}, got {arr.shape[1]}")

            index.add(arr)
            vectors_added = arr.shape[0]
            state.total_vectors += vectors_added

            info = meta_map.get(_id, {})
            md = {
                "id": str(_id),
                "title": "",
                "preview": "",
                "chunk_count": int(info.get("chunks") or info.get("chunk_count") or vectors_added),
            }
            metadata.append(md)
            state.processed_files.append(_id)
            processed_set.add(_id)
            files_added_since_save += 1
            if files_added_since_save >= args.save_interval:
                faiss.write_index(index, str(index_path))
                save_metadata(metadata_path, metadata)
                write_progress(progress_path, state)
                logging.info(
                    "Progress saved after %d new ids (%d vectors total).",
                    files_added_since_save,
                    state.total_vectors,
                )
                files_added_since_save = 0
    else:
        # per-id arrays inside npz
        # sort keys by numeric id when possible
        def keyfun(k: str):
            try:
                return int(k)
            except Exception:
                return k
        for _id in sorted(keys, key=keyfun):
            if _id in processed_set:
                continue
            arr = np.array(npz[_id], dtype="float32")
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = normalize_embeddings(arr)

            if index is None:
                index = faiss.IndexFlatIP(arr.shape[1])
                logging.info("Created IndexFlatIP with dimension %d", arr.shape[1])
            if arr.shape[1] != index.d:
                raise ValueError(f"Embedding dimension mismatch for id {_id}: expected {index.d}, got {arr.shape[1]}")

            index.add(arr)
            vectors_added = arr.shape[0]
            state.total_vectors += vectors_added

            info = meta_map.get(_id, {})
            md = {
                "id": str(_id),
                "title": "",
                "preview": "",
                "chunk_count": int(info.get("chunks") or info.get("chunk_count") or vectors_added),
            }
            metadata.append(md)
            state.processed_files.append(_id)
            processed_set.add(_id)
            files_added_since_save += 1
            if files_added_since_save >= args.save_interval:
                faiss.write_index(index, str(index_path))
                save_metadata(metadata_path, metadata)
                write_progress(progress_path, state)
                logging.info(
                    "Progress saved after %d new ids (%d vectors total).",
                    files_added_since_save,
                    state.total_vectors,
                )
                files_added_since_save = 0

    if index is None:
        raise RuntimeError("No embeddings were added to the index. Check the NPZ file.")

    faiss.write_index(index, str(index_path))
    save_metadata(metadata_path, metadata)
    # Default model name (unknown from NPZ); store dimension and count
    save_config(config_path, DEFAULT_MODEL, index.d, index.ntotal)
    write_progress(progress_path, state)

    logging.info("Index built with %d vectors and saved to %s", index.ntotal, index_path)
    logging.info("Metadata saved to %s", metadata_path)
    logging.info("Config saved to %s", config_path)


def build_index(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    if args.save_interval <= 0:
        raise ValueError("save-interval must be greater than zero")

    embeddings_dir: Path = args.embeddings_dir
    output_dir: Path = args.output_dir
    progress_path = args.progress_file or (output_dir / PROGRESS_FILENAME)
    index_path = output_dir / INDEX_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    config_path = output_dir / CONFIG_FILENAME

    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list_embedding_files(embeddings_dir)

    chunk_metadata = load_chunk_metadata(args.chunks_path)
    logging.info("Loaded chunk metadata for %d articles", len(chunk_metadata))

    state = read_progress(progress_path) if args.resume else None
    if state is None:
        state = BuildState(processed_files=[], total_vectors=0)

    processed_set = set(state.processed_files)

    if args.resume and index_path.exists():
        index, metadata = load_existing_index(index_path, metadata_path)
        logging.info("Loaded existing index with %d vectors", index.ntotal)
    else:
        index = None
        metadata = []

    files_added_since_save = 0

    for file_path in all_files:
        if file_path.name in processed_set:
            continue

        embeddings = load_embedding_array(file_path)
        embeddings = normalize_embeddings(embeddings)

        if index is None:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            logging.info("Created IndexFlatIP with dimension %d", dimension)

        if embeddings.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension mismatch for {file_path}: expected {index.d}, got {embeddings.shape[1]}"
            )

        index.add(embeddings)
        vectors_added = embeddings.shape[0]
        state.total_vectors += vectors_added

        article_id = file_path.stem
        meta_entry = chunk_metadata.get(article_id, {
            "id": article_id,
            "title": "",
            "preview": "",
            "chunk_count": int(embeddings.shape[0]),
        })
        meta_entry = dict(meta_entry)
        meta_entry["embedding_path"] = str(file_path.resolve())
        metadata.append(meta_entry)

        state.processed_files.append(file_path.name)
        processed_set.add(file_path.name)

        logging.debug("Indexed %s (%d vectors)", file_path.name, vectors_added)

        files_added_since_save += 1
        if files_added_since_save >= args.save_interval:
            faiss.write_index(index, str(index_path))
            save_metadata(metadata_path, metadata)
            write_progress(progress_path, state)
            logging.info(
                "Progress saved after %d new files (%d vectors total).",
                files_added_since_save,
                state.total_vectors,
            )
            files_added_since_save = 0

    if index is None:
        raise RuntimeError("No embeddings were added to the index. Check the embeddings directory.")

    faiss.write_index(index, str(index_path))
    save_metadata(metadata_path, metadata)
    save_config(config_path, DEFAULT_MODEL, index.d, index.ntotal)
    write_progress(progress_path, state)

    logging.info("Index built with %d vectors and saved to %s", index.ntotal, index_path)
    logging.info("Metadata saved to %s", metadata_path)
    logging.info("Config saved to %s", config_path)


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_query_resources(
    index_path: Path,
    metadata_path: Path,
    config_path: Path,
    model_name_override: Optional[str],
) -> Tuple[faiss.Index, List[dict], SentenceTransformer]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    index = faiss.read_index(str(index_path))

    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = []

    config_model = DEFAULT_MODEL
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config_model = config.get("model_name", config_model)
    model_name = model_name_override or config_model

    device = choose_device()
    model = SentenceTransformer(model_name, device=device)
    logging.info("Loaded query model '%s' on device %s", model_name, device)
    return index, metadata, model


def query_index(
    index: faiss.Index,
    metadata: Sequence[dict],
    model: SentenceTransformer,
    query_text: str,
    top_k: int,
) -> List[dict]:
    embedding = model.encode(
        [query_text],
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    scores, indices = index.search(embedding, top_k)
    results: List[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        results.append(entry)
    return results


def handle_query(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    index, metadata, model = load_query_resources(
        args.index_path, args.metadata_path, args.config_path, args.model_name
    )
    results = query_index(index, metadata, model, args.query_text, args.top_k)
    if not results:
        logging.info("No results found.")
        return
    for rank, item in enumerate(results, start=1):
        preview = item.get("preview", "")
        logging.info(
            "%d. id=%s score=%.4f title=%s preview=%s",
            rank,
            item.get("id"),
            item.get("score"),
            item.get("title", ""),
            preview[:120],
        )


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_index(args)
    elif args.command == "build-from-npz":
        build_index_from_npz(args)
    elif args.command == "query":
        handle_query(args)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
