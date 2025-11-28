#!/Iteratorusr/bin/env python3
"""Optimized hierarchical chunk embeddings for Wikipedia with GPU acceleration and compressed storage."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


@dataclass
class ChunkBatch:
    article_ids: list[str]
    chunks: list[list[str]]
    indices: list[int]


class ArticleDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.offsets = []
        self._build_index()

    def _build_index(self):
        """Indexa a posição (byte offsets) de cada linha no arquivo JSONL."""
        # usa modo binário para garantir que tell() funcione
        with open(self.path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        """Carrega apenas 1 artigo do arquivo usando o offset em bytes."""
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        # decodifica a linha (bytes -> str) antes de json.loads
        article = json.loads(line.decode("utf-8"))

        # --- AQUI ESTÁ A CORREÇÃO LÓGICA ---
        # Pega a lista direto da chave "chunks" do seu JSONL
        chunks = article.get("chunks", [])

        return {"id": str(article.get("id", f"article_{idx:06d}")), "chunks": chunks, "index": idx}


def collate_fn(batch):
    """Custom collate to handle variable chunks per article."""
    return batch


# def load_articles(path: Path) -> Iterator[dict]:
#     """Stream articles one by one to reduce RAM usage."""
#     suffix = path.suffix.lower()
#     if suffix in {".jsonl", ".jsonlines", ".ndjson"}:
#         with path.open("r", encoding="utf-8") as f:
#             for line in f:
#                 if line.strip():
#                     yield json.loads(line)
#     else:
#         with path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#             for article in (data if isinstance(data, list) else [data]):
#                 yield article


def process_embeddings_optimized(
    model: SentenceTransformer,
    dataloader: DataLoader,
    output_dir: Path,
    storage_mode: str,
    checkpoint_interval: int,
    use_fp16: bool = True
):
    """Process embeddings with batching optimization and compressed storage."""

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "written": 0, "total_chunks": 0}
    checkpoint_path = output_dir / "checkpoint.json"

    # Enable mixed precision if available
    if use_fp16 and torch.cuda.is_available():
        model = model.half()
        logging.info("Mixed precision (FP16) enabled")

    # Storage containers
    if storage_mode == "compressed":
        all_embeddings = {}  # article_id -> embeddings
        metadata = {}  # article_id -> {chunks: int, shape: tuple}

    # Collect chunks for batch processing
    chunk_buffer = []
    metadata_buffer = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing articles")):
        for item in batch:
            article_id = item["id"]
            chunks = item["chunks"]

            if not chunks:
                stats["processed"] += 1
                continue

            # Add to buffer for batch processing
            chunk_buffer.extend(chunks)
            metadata_buffer.append({
                "id": article_id,
                "chunk_count": len(chunks),
                "start_idx": len(chunk_buffer) - len(chunks)
            })

        # Process buffer when it reaches optimal size
        if len(chunk_buffer) >= 8192:  # Process ~1024 chunks at once
            if storage_mode == "compressed":
                _process_chunk_buffer_compressed(
                    model, chunk_buffer, metadata_buffer,
                    all_embeddings, metadata, stats
                )
            else:
                _process_chunk_buffer_individual(
                    model, chunk_buffer, metadata_buffer,
                    output_dir, stats
                )
            chunk_buffer.clear()
            metadata_buffer.clear()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save checkpoint periodically
        if stats["processed"] > 0 and stats["processed"] % checkpoint_interval == 0:
            _save_checkpoint(checkpoint_path, stats)

            # For compressed mode, save intermediate file
            if storage_mode == "compressed":
                _save_compressed_batch(
                    output_dir, all_embeddings, metadata, stats["processed"])
                all_embeddings.clear()
                metadata.clear()

    # Process remaining chunks
    if chunk_buffer:
        if storage_mode == "compressed":
            _process_chunk_buffer_compressed(
                model, chunk_buffer, metadata_buffer,
                all_embeddings, metadata, stats
            )
        else:
            _process_chunk_buffer_individual(
                model, chunk_buffer, metadata_buffer,
                output_dir, stats
            )

    # Save final compressed file
    if storage_mode == "compressed":
        _save_compressed_batch(output_dir, all_embeddings,
                               metadata, stats["processed"], final=True)
        _merge_compressed_batches(output_dir)

    _save_checkpoint(checkpoint_path, stats, finished=True)
    return stats


def _process_chunk_buffer_compressed(model, chunks, meta_buffer, all_embeddings, metadata, stats):
    """Process chunks and store in memory for compressed saving."""
    if not chunks:
        return

    # Generate embeddings for all chunks at once
    embeddings = model.encode(
        chunks,
        batch_size=4096,  # Optimal for T4 GPU
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=model.device
    )

    # Store embeddings by article
    for meta in meta_buffer:
        article_id = meta["id"]
        start = meta["start_idx"]
        count = meta["chunk_count"]

        article_embeddings = embeddings[start:start + count]
        all_embeddings[article_id] = article_embeddings.astype(np.float32)
        metadata[article_id] = {
            "chunks": count,
            "shape": article_embeddings.shape,
            "dtype": "float32"
        }

        stats["processed"] += 1
        stats["written"] += 1
        stats["total_chunks"] += count

    del embeddings
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _process_chunk_buffer_individual(model, chunks, meta_buffer, output_dir, stats):
    """Process chunks and save individual files."""
    if not chunks:
        return

    # Generate embeddings for all chunks at once
    embeddings = model.encode(
        chunks,
        batch_size=1024,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    # Split and save individual files
    for meta in meta_buffer:
        article_id = meta["id"]
        start = meta["start_idx"]
        count = meta["chunk_count"]

        article_embeddings = embeddings[start:start + count]
        output_path = output_dir / f"{article_id}.npy"
        np.save(output_path, article_embeddings)

        stats["processed"] += 1
        stats["written"] += 1
        stats["total_chunks"] += count

    del embeddings
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_compressed_batch(output_dir: Path, embeddings_dict: dict, metadata: dict, batch_id: int, final: bool = False):
    """Save a batch of embeddings in compressed format."""
    if not embeddings_dict:
        return

    suffix = "final" if final else f"batch_{batch_id:06d}"

    # Save embeddings in compressed npz
    embeddings_path = output_dir / f"embeddings_{suffix}.npz"
    np.savez_compressed(embeddings_path, **embeddings_dict)

    # Save metadata
    metadata_path = output_dir / f"metadata_{suffix}.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Calculate compression stats
    uncompressed_size = sum(emb.nbytes for emb in embeddings_dict.values())
    compressed_size = embeddings_path.stat().st_size
    ratio = compressed_size / uncompressed_size * 100 if uncompressed_size > 0 else 0

    logging.info(f"Saved batch {suffix}: {len(embeddings_dict)} articles")
    logging.info(f"  Uncompressed: {uncompressed_size / 1024**2:.1f} MB")
    logging.info(
        f"  Compressed: {compressed_size / 1024**2:.1f} MB ({ratio:.1f}%)")


def _merge_compressed_batches(output_dir: Path):
    """Merge all batch files into single compressed archive."""
    batch_files = sorted(output_dir.glob("embeddings_batch_*.npz"))

    if not batch_files:
        return

    logging.info(f"Merging {len(batch_files)} batch files...")

    all_embeddings = {}
    all_metadata = {}

    # Load all batches
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with np.load(batch_file) as data:
            all_embeddings.update({k: data[k] for k in data.files})

        # Load corresponding metadata
        meta_file = batch_file.parent / \
            batch_file.name.replace(
                "embeddings_", "metadata_").replace(".npz", ".json")
        if meta_file.exists():
            with meta_file.open() as f:
                all_metadata.update(json.load(f))

    # Save merged file
    final_path = output_dir / "embeddings_all.npz"
    np.savez_compressed(final_path, **all_embeddings)

    # Save merged metadata
    meta_path = output_dir / "metadata_all.json"
    with meta_path.open("w") as f:
        json.dump(all_metadata, f, indent=2)

    # Calculate final stats
    uncompressed = sum(emb.nbytes for emb in all_embeddings.values())
    compressed = final_path.stat().st_size

    logging.info(f"✓ Merged file saved: {final_path}")
    logging.info(f"  Total articles: {len(all_embeddings):,}")
    logging.info(f"  Uncompressed: {uncompressed / 1024**3:.2f} GB")
    logging.info(
        f"  Compressed: {compressed / 1024**3:.2f} GB ({compressed/uncompressed*100:.1f}%)")

    # Clean up batch files
    logging.info("Cleaning up batch files...")
    for batch_file in batch_files:
        batch_file.unlink()
        meta_file = batch_file.parent / \
            batch_file.name.replace(
                "embeddings_", "metadata_").replace(".npz", ".json")
        if meta_file.exists():
            meta_file.unlink()


def _save_checkpoint(path: Path, stats: dict, finished: bool = False):
    """Save processing checkpoint."""
    checkpoint = {**stats, "finished": finished}
    with path.open("w") as f:
        json.dump(checkpoint, f, indent=2)
    logging.info(
        f"Checkpoint: {stats['processed']} articles, {stats['total_chunks']} chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Wikipedia embedding generator")
    parser.add_argument("articles_path", type=Path,
                        help="Path to articles JSON/JSONL")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("embeddings"),
                        help="Output directory")
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Model name")
    parser.add_argument("--chunk-size", type=int,
                        default=400, help="Tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int,
                        default=50, help="Overlap tokens")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers for I/O parallelization")
    parser.add_argument("--checkpoint-interval", type=int, default=512,
                        help="Save checkpoint every N articles")
    parser.add_argument("--storage-mode", choices=["compressed", "individual"],
                        default="compressed",
                        help="Storage format: compressed (single .npz) or individual (.npy files)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    # Load model and tokenizer
    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load articles
    logging.info(f"Loading articles from {args.articles_path}")
    dataset = ArticleDataset(args.articles_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Article batches for I/O
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda")
    )

    # Process embeddings
    logging.info(
        f"Starting embedding generation (storage: {args.storage_mode})...")
    stats = process_embeddings_optimized(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        storage_mode=args.storage_mode,
        checkpoint_interval=args.checkpoint_interval,
        use_fp16=not args.no_fp16
    )

    logging.info(f"✓ Complete! Processed {stats['processed']:,} articles")
    logging.info(f"✓ Generated {stats['total_chunks']:,} chunk embeddings")
    logging.info(
        f"✓ Written {stats['written']:,} embeddings to {args.output_dir}")

    if args.storage_mode == "compressed":
        final_file = args.output_dir / "embeddings_all.npz"
        if final_file.exists():
            size_gb = final_file.stat().st_size / 1024**3
            logging.info(
                f"✓ Final compressed file: {final_file} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
