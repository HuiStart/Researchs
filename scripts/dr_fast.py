#!/usr/bin/env python3
"""Run DeepReviewer Fast Mode through the project DeepReviewer class."""

from __future__ import annotations

import argparse # 解析命令行参数
import csv
import importlib.metadata
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path    # 优雅地处理文件路径

ROOT = Path(__file__).resolve().parents[1]  # 获取当前脚本所在目录的上两级目录（通常为项目根目录）
sys.path.insert(0, str(ROOT))   # 将项目根目录插入模块搜索路径的最前端，确保可以导入项目内的 ai_researcher 模块
from ai_researcher import DeepReviewer

DEFAULT_MODEL = (
    ROOT
    / ".hf_cache/models--WestlakeNLP--DeepReviewer-7B/"
    "snapshots/cc31e8d72f62ad24c46f9a255f0cec31f18e8949"
)

SCORE_KEYS = ("rating", "soundness", "presentation", "contribution")
MIN_VLLM_VERSION = (0, 7, 2)


def raise_csv_field_limit() -> None:
    # Full paper sources in the dataset can exceed Python's conservative default.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        match = re.match(r"\d+", part)
        if not match:
            break
        parts.append(int(match.group(0)))
    return tuple(parts)


def require_supported_vllm() -> None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit("vLLM backend requested, but vllm is not installed.") from exc

    if parse_version_tuple(version) < MIN_VLLM_VERSION:
        min_version = ".".join(str(part) for part in MIN_VLLM_VERSION)
        raise SystemExit(
            f"vLLM backend requires vllm>={min_version} for this Qwen2-based DeepReviewer model. "
            f"Current vllm=={version} generated invalid repeated-zero outputs in this environment. "
            "Use --backend transformers, or upgrade vllm in the hdmag environment."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepReviewer Fast Mode with vLLM.")
    parser.add_argument("-m","--model", default=str(DEFAULT_MODEL), help="Local model path or HF model id.") # 默认为本地的deepreview-7B
    parser.add_argument("-d","--csv", default="data/deepreview/test_2024.csv")
    parser.add_argument("-r","--row", type=int, default=0, help="Start row for CSV input.")
    parser.add_argument("-n","--num-rows", type=int, default=1, help="Number of CSV rows to run.")
    parser.add_argument("--all", action="store_true", help="Run all rows from --row to EOF.")
    parser.add_argument("--paper-file", help="Optional plain text/LaTeX paper file.（自己指定本地文件来运行）")

    # 用于单条样本输出，或者在多条样本时提供默认输出目录的参考路径
    parser.add_argument("-o","--output", default="outputs/",help="Output file for one row, or output directory for multiple rows.")   

    # transformers: 用Transformers 加载模型     vllm: 用 vLLM 加载模型，适合高吞吐、tensor parallel、多 GPU
    # 如果你想用 --tensor-parallel-size 2，必须显式写 --backend vllm
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"],help="选择推理后端")

    # 1: 每生成完一个样本就立刻打印分数对比和 running metrics
    # 大于 1: 每生成完一个 batch 就打印一次分数对比和 running metrics
    parser.add_argument("-b","--generation-batch-size", type=int, default=1,help="每次送给模型的样本数量。对于 transformers 后端，必须为 1。对于 vLLM 后端，可以大于 1 来提高吞吐量，但会增加每个样本的延迟。")
    parser.add_argument("-tps","--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=13000,help="模型最大上下文长度，包含输入 prompt 和输出 tokens")
    parser.add_argument("-t","--max-tokens", type=int, default=4096, help="每条 review 最多生成多少 token")

    # vllm参数
    parser.add_argument("--max-num-seqs", type=int, default=32, help="vllm同时掉调度的最大序列数量，通常设置为 GPU 数量的倍数以获得最佳性能")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


def first_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def mean_or_none(values) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def normalize_decision(value: str | None) -> str | None:
    if not value:
        return None
    value = value.lower().strip()
    if "accept" in value:
        return "accept"
    if "reject" in value:
        return "reject"
    return value


def parse_reference(row: dict) -> dict:
    ref_scores = {}
    raw_ratings = []

    try:
        raw_ratings = [first_number(value) for value in json.loads(row.get("rating") or "[]")]
        raw_ratings = [value for value in raw_ratings if value is not None]
    except json.JSONDecodeError:
        raw_ratings = []
    ref_scores["rating"] = mean_or_none(raw_ratings)

    try:
        reviewer_comments = json.loads(row.get("reviewer_comments") or "[]")
    except json.JSONDecodeError:
        reviewer_comments = []

    if ref_scores["rating"] is None:
        ref_scores["rating"] = mean_or_none(
            first_number(comment.get("rating")) for comment in reviewer_comments
        )

    for key in ("soundness", "presentation", "contribution"):
        ref_scores[key] = mean_or_none(
            first_number((comment.get("content") or {}).get(key))
            for comment in reviewer_comments
        )

    return {
        "id": row.get("id", ""),
        "mode": row.get("mode", ""),
        "year": row.get("year", ""),
        "decision": row.get("decision", ""),
        "normalized_decision": normalize_decision(row.get("decision")),
        "scores": ref_scores,
        "raw_ratings": raw_ratings,
    }


def parse_prediction(review: dict) -> dict:
    meta_review = review.get("meta_review") or {}
    pred_scores = {
        key: first_number(meta_review.get(key))
        for key in SCORE_KEYS
    }
    raw_text = review.get("raw_text", "")
    for key in SCORE_KEYS:
        if pred_scores[key] is not None:
            continue
        section_match = re.search(
            rf"## {key.title()}:\s*(.*?)(?=\n## |\Z)",
            raw_text,
            re.DOTALL,
        )
        if section_match:
            pred_scores[key] = first_number(section_match.group(1))

    decision = review.get("decision", "")
    if not decision:
        decision_match = re.search(r"## Decision:\s*(.*?)(?=\n## |\Z)", raw_text, re.DOTALL)
        if decision_match:
            decision = decision_match.group(1).strip().splitlines()[0]

    return {
        "scores": pred_scores,
        "decision": decision,
        "normalized_decision": normalize_decision(decision),
    }


def format_score(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def binary_decision(value: str | None) -> int | None:
    if value == "accept":
        return 1
    if value == "reject":
        return 0
    return None


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_diffs = [x - x_mean for x in xs]
    y_diffs = [y - y_mean for y in ys]
    denom = math.sqrt(sum(x * x for x in x_diffs) * sum(y * y for y in y_diffs))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(x_diffs, y_diffs)) / denom


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def macro_f1(true_values: list[int], pred_values: list[int]) -> float | None:
    if not true_values or len(true_values) != len(pred_values):
        return None
    f1_values = []
    for label in (0, 1):
        tp = sum(1 for t, p in zip(true_values, pred_values) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_values, pred_values) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_values, pred_values) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1_values) / len(f1_values)


def calculate_metrics(records: list[dict]) -> dict:
    metrics = {}
    for key in SCORE_KEYS:
        pairs = [
            (record["ref"]["scores"].get(key), record["pred"]["scores"].get(key))
            for record in records
            if record["ref"]["scores"].get(key) is not None
            and record["pred"]["scores"].get(key) is not None
        ]
        if pairs:
            refs = [pair[0] for pair in pairs]
            preds = [pair[1] for pair in pairs]
            errors = [pred - ref for ref, pred in pairs]
            metrics[f"{key}_mae"] = sum(abs(error) for error in errors) / len(errors)
            metrics[f"{key}_mse"] = sum(error * error for error in errors) / len(errors)
            metrics[f"{key}_spearman"] = spearman(refs, preds)
        else:
            metrics[f"{key}_mae"] = None
            metrics[f"{key}_mse"] = None
            metrics[f"{key}_spearman"] = None

        total_pairs = 0
        correct_pairs = 0
        for left, right in combinations(records, 2):
            left_ref = left["ref"]["scores"].get(key)
            right_ref = right["ref"]["scores"].get(key)
            left_pred = left["pred"]["scores"].get(key)
            right_pred = right["pred"]["scores"].get(key)
            if None in {left_ref, right_ref, left_pred, right_pred}:
                continue
            total_pairs += 1
            if (left_ref > right_ref) == (left_pred > right_pred):
                correct_pairs += 1
        metrics[f"{key}_pairwise_acc"] = correct_pairs / total_pairs if total_pairs else None

    true_decisions = []
    pred_decisions = []
    for record in records:
        true_value = binary_decision(record["ref"]["normalized_decision"])
        pred_value = binary_decision(record["pred"]["normalized_decision"])
        if true_value is None or pred_value is None:
            continue
        true_decisions.append(true_value)
        pred_decisions.append(pred_value)

    if true_decisions:
        metrics["decision_accuracy"] = sum(
            1 for t, p in zip(true_decisions, pred_decisions) if t == p
        ) / len(true_decisions)
        metrics["decision_f1"] = macro_f1(true_decisions, pred_decisions)
    else:
        metrics["decision_accuracy"] = None
        metrics["decision_f1"] = None

    return metrics


def print_sample_report(sample: dict, review: dict, output_path: Path, records: list[dict]) -> None:
    pred = parse_prediction(review)
    record = {
        "row_idx": sample["row_idx"],
        "id": sample.get("id", ""),
        "ref": sample["ref"],
        "pred": pred,
    }
    records.append(record)

    print(f"\n[{len(records)}] row={sample['row_idx']} id={sample.get('id', 'N/A')} output={output_path}")
    for key in SCORE_KEYS:
        ref_score = sample["ref"]["scores"].get(key)
        pred_score = pred["scores"].get(key)
        error = None if ref_score is None or pred_score is None else pred_score - ref_score
        print(
            f"  {key:<12} pred={format_score(pred_score)} "
            f"ref={format_score(ref_score)} "
            f"err={format_score(error)}"
        )
    if sample["ref"].get("raw_ratings"):
        print(f"  reference reviewer ratings: {sample['ref']['raw_ratings']}")

    ref_decision = sample["ref"].get("normalized_decision")
    pred_decision = pred.get("normalized_decision")
    match = "yes" if ref_decision and pred_decision and ref_decision == pred_decision else "no"
    print(f"  decision     pred={pred_decision or 'N/A'} ref={ref_decision or 'N/A'} match={match}")

    metrics = calculate_metrics(records)
    print(
        "  running      "
        f"Rating MAE={format_score(metrics['rating_mae'])} "
        f"MSE={format_score(metrics['rating_mse'])} "
        f"Spearman={format_score(metrics['rating_spearman'])} "
        f"Decision Acc={format_score(metrics['decision_accuracy'])}"
    )


def print_metrics_summary(records: list[dict]) -> None:
    if not records:
        return
    metrics = calculate_metrics(records)
    print("\nEvaluation summary")
    print("| Metric | MAE | MSE | Spearman | Pairwise Acc |")
    print("|---|---:|---:|---:|---:|")
    for key in SCORE_KEYS:
        print(
            f"| {key.title()} | {format_score(metrics[f'{key}_mae'])} "
            f"| {format_score(metrics[f'{key}_mse'])} "
            f"| {format_score(metrics[f'{key}_spearman'])} "
            f"| {format_score(metrics[f'{key}_pairwise_acc'])} |"
        )
    print(
        f"Decision Accuracy: {format_score(metrics['decision_accuracy'])}  "
        f"Decision Macro-F1: {format_score(metrics['decision_f1'])}"
    )


def load_papers(args: argparse.Namespace) -> list[dict]:
    if args.paper_file:
        return [{
            "row_idx": args.row,
            "paper_text": Path(args.paper_file).read_text(encoding="utf-8", errors="replace"),
            "id": Path(args.paper_file).stem,
            "ref": {
                "id": Path(args.paper_file).stem,
                "mode": "",
                "year": "",
                "decision": "",
                "normalized_decision": None,
                "scores": {key: None for key in SCORE_KEYS},
                "raw_ratings": [],
            },
        }]

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
            ref = parse_reference(row)
            papers.append({
                "row_idx": idx,
                "paper_text": user_messages[-1],
                "id": ref["id"],
                "ref": ref,
            })

    if not papers:
        raise IndexError(f"No CSV rows found from row {args.row} in {args.csv}")

    return papers


def output_path_for(args: argparse.Namespace, row_idx: int, multi_row: bool) -> Path:
    if not multi_row:
        return Path(args.output)
    output = Path(args.output)
    output_dir = output.parent if output.suffix else output
    return output_dir / f"fast_row{row_idx}.md"


def main() -> None:
    raise_csv_field_limit()

    args = parse_args()
    papers = load_papers(args)
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1")
    if args.backend == "vllm":
        require_supported_vllm()
    if args.backend == "vllm" and args.dtype == "auto":
        print("dtype: auto requested with vLLM; using float16 to avoid bfloat16 degeneration on RTX A4000.")
        args.dtype = "float16"

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

    multi_row = len(papers) > 1
    written_paths = []
    metric_records = []

    print(f"model: {args.model}")
    print(f"backend: {args.backend}")
    print(f"mode: Fast Mode")
    print(f"rows: {papers[0]['row_idx']}..{papers[-1]['row_idx']} ({len(papers)} total)")
    print(f"max_model_len: {args.max_model_len}")
    print(f"max_num_seqs: {args.max_num_seqs}")
    print(f"tokenizer_mode: {args.tokenizer_mode}")
    print(f"enforce_eager: {args.enforce_eager}")
    print(f"disable_custom_all_reduce: {args.disable_custom_all_reduce}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"generation_batch_size: {args.generation_batch_size}")

    for start in range(0, len(papers), args.generation_batch_size):
        batch = papers[start:start + args.generation_batch_size]
        reviews = reviewer.evaluate(
            [sample["paper_text"] for sample in batch],
            mode="Fast Mode",
            reviewer_num=4,
            max_tokens=args.max_tokens,
        )

        for sample, review in zip(batch, reviews):
            row_idx = sample["row_idx"]
            output_path = output_path_for(args, row_idx, multi_row)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(review["raw_text"] + "\n", encoding="utf-8")
            written_paths.append(output_path)
            print_sample_report(sample, review, output_path.resolve(), metric_records)

    print_metrics_summary(metric_records)

    print("\nWritten outputs")
    for output_path in written_paths:
        print(f"output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
