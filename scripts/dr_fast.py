#!/usr/bin/env python3
"""Run DeepReviewer Fast Mode through the project DeepReviewer class."""

from __future__ import annotations

import argparse 
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_researcher import DeepReviewer


DEFAULT_MODEL = (
    ROOT
    / ".hf_cache/models--WestlakeNLP--DeepReviewer-7B/"
    "snapshots/cc31e8d72f62ad24c46f9a255f0cec31f18e8949"
)


def raise_csv_field_limit() -> None:
    # Full paper sources in the dataset can exceed Python's conservative default.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepReviewer Fast Mode with vLLM.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local model path or HF model id.")
    parser.add_argument("--csv", default="data/deepreview/test_2024.csv")
    parser.add_argument("--row", type=int, default=0, help="Start row for CSV input.")
    parser.add_argument("--num-rows", type=int, default=1, help="Number of CSV rows to run.")
    parser.add_argument("--all", action="store_true", help="Run all rows from --row to EOF.")
    parser.add_argument("--paper-file", help="Optional plain text/LaTeX paper file.")
    parser.add_argument("--output", default="outputs/fast_row0.md")
    parser.add_argument("--output-dir", help="Directory for multi-row outputs.")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tokenizer-mode", default="slow")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


def load_papers(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.paper_file:
        return [(args.row, Path(args.paper_file).read_text(encoding="utf-8", errors="replace"))]

    papers = []
    with open(args.csv, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < args.row:
                continue
            if not args.all and len(papers) >= args.num_rows:
                break
            messages = json.loads(row["inputs"])
            user_messages = [m["content"] for m in messages if m.get("role") == "user"]
            if not user_messages:
                raise ValueError(f"No user message found in {args.csv} row {args.row}")
            papers.append((idx, user_messages[-1]))

    if not papers:
        raise IndexError(f"No CSV rows found from row {args.row} in {args.csv}")

    return papers


def output_path_for(args: argparse.Namespace, row_idx: int, multi_row: bool) -> Path:
    if not multi_row:
        return Path(args.output)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output).parent
    return output_dir / f"fast_row{row_idx}.md"


def main() -> None:
    raise_csv_field_limit()

    args = parse_args()
    papers = load_papers(args)
    paper_texts = [paper_text for _, paper_text in papers]

    reviewer = DeepReviewer(
        model_size="7B",
        custom_model_name=args.model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tokenizer_mode=args.tokenizer_mode,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        dtype=args.dtype,
    )

    reviews = reviewer.evaluate(
        paper_texts,
        mode="Fast Mode",
        reviewer_num=4,
        max_tokens=args.max_tokens,
    )

    multi_row = len(papers) > 1
    written_paths = []
    for (row_idx, _), review in zip(papers, reviews):
        output_path = output_path_for(args, row_idx, multi_row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(review["raw_text"] + "\n", encoding="utf-8")
        written_paths.append(output_path)

    print(f"model: {args.model}")
    print(f"backend: {args.backend}")
    print(f"mode: Fast Mode")
    print(f"rows: {papers[0][0]}..{papers[-1][0]} ({len(papers)} total)")
    print(f"max_model_len: {args.max_model_len}")
    print(f"max_num_seqs: {args.max_num_seqs}")
    print(f"tokenizer_mode: {args.tokenizer_mode}")
    print(f"enforce_eager: {args.enforce_eager}")
    print(f"disable_custom_all_reduce: {args.disable_custom_all_reduce}")
    print(f"max_tokens: {args.max_tokens}")
    for output_path in written_paths:
        print(f"output: {output_path.resolve()}")
    print(reviews[0]["raw_text"][:2000])


if __name__ == "__main__":
    main()
