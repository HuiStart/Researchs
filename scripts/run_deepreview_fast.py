#!/usr/bin/env python3
"""Run DeepReviewer in Fast Mode with a Transformers backend.

This runner is intentionally independent from ``ai_researcher.DeepReviewer``
because the package import currently requires vLLM. Fast Mode does not require
OpenScholar or external LLM APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from huggingface_hub.errors import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer


FAST_SYSTEM_PROMPT = (
    "You are an expert academic reviewer tasked with providing a thorough and "
    "balanced evaluation of research papers. Your thinking mode is Fast Mode. "
    "In this mode, you should quickly provide the review results."
)


SMOKE_PAPER = r"""
\title{A Small Demonstration Paper for Automated Review}
\begin{abstract}
This paper proposes a lightweight method for evaluating whether a local review
model can be loaded and prompted successfully. The method introduces a simple
prompting protocol and reports qualitative examples.
\end{abstract}

\section{Introduction}
Automated paper review systems can help researchers obtain early feedback, but
their outputs must be treated as assistance rather than replacement for human
judgment. This demonstration paper contains enough structure for a smoke test.

\section{Method}
We format a short paper-like input, pass it to the reviewer in Fast Mode, and
inspect whether the model returns a structured review with strengths,
weaknesses, questions, a rating, confidence, and a decision.

\section{Experiments}
The experiment is a functional smoke test. It does not claim scientific novelty
or benchmark performance.

\section{Conclusion}
The expected outcome is a coherent review that identifies the limited novelty
and weak experimental validation of this demonstration paper.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepReviewer-7B Fast Mode.")
    parser.add_argument(
        "--model",
        default="WestlakeNLP/DeepReviewer-7B",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--paper-file",
        help="Text/Markdown/LaTeX paper file to review. If omitted, a smoke-test paper is used.",
    )
    parser.add_argument(
        "--csv",
        help="Optional DeepReview CSV file. The script extracts the user message from the selected row.",
    )
    parser.add_argument("--row", type=int, default=0, help="CSV row index to use with --csv.")
    parser.add_argument(
        "--output",
        default="outputs/deepreview_fast_output.md",
        help="Path for the generated review.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help="Hugging Face cache directory. Keep this on a large disk.",
    )
    parser.add_argument(
        "--hf-token-file",
        help="Optional file containing a Hugging Face token. The token is read but never printed.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=24000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="float16",
        help="Model dtype. float16 is the safest choice for RTX A4000.",
    )
    return parser.parse_args()


def read_hf_token(token_file: str | None) -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if not token_file:
        return None
    text = Path(token_file).read_text(encoding="utf-8", errors="ignore").strip()
    match = re.search(r"hf_[A-Za-z0-9_-]+", text)
    return match.group(0) if match else (text or None)


def load_paper_text(args: argparse.Namespace) -> str:
    if args.paper_file:
        return Path(args.paper_file).read_text(encoding="utf-8", errors="replace")

    if args.csv:
        with open(args.csv, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx != args.row:
                    continue
                messages = json.loads(row["inputs"])
                user_messages = [m["content"] for m in messages if m.get("role") == "user"]
                if not user_messages:
                    raise ValueError(f"No user message found in {args.csv} row {args.row}")
                return user_messages[-1]
        raise IndexError(f"CSV row {args.row} not found in {args.csv}")

    return SMOKE_PAPER


def dtype_from_arg(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def build_prompt(tokenizer, paper_text: str, max_input_tokens: int) -> str:
    messages = [
        {"role": "system", "content": FAST_SYSTEM_PROMPT},
        {"role": "user", "content": paper_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System:\n{FAST_SYSTEM_PROMPT}\n\nUser:\n{paper_text}\n\nAssistant:\n"

    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(input_ids) <= max_input_tokens:
        return prompt

    keep = input_ids[-max_input_tokens:]
    return tokenizer.decode(keep, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    token = read_hf_token(args.hf_token_file)
    paper_text = load_paper_text(args)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=str(cache_dir),
            token=token,
            trust_remote_code=True,
        )
    except (GatedRepoError, OSError) as exc:
        raise SystemExit(
            "Cannot access the DeepReviewer model on Hugging Face. "
            "Make sure your account has accepted the model terms and pass a valid token "
            "with --hf-token-file or HF_TOKEN. Original error: "
            f"{exc}"
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = build_prompt(tokenizer, paper_text, args.max_input_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            cache_dir=str(cache_dir),
            token=token,
            trust_remote_code=True,
            torch_dtype=dtype_from_arg(args.dtype),
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except (GatedRepoError, OSError) as exc:
        raise SystemExit(
            "Cannot download/load the DeepReviewer weights. "
            "Check Hugging Face access, token permissions, and available disk space. "
            f"Original error: {exc}"
        ) from exc
    model.eval()

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    review = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    output_path.write_text(review + "\n", encoding="utf-8")

    print(f"model: {args.model}")
    print(f"input_tokens: {inputs['input_ids'].shape[-1]}")
    print(f"output_tokens: {new_tokens.shape[-1]}")
    print(f"review_path: {output_path.resolve()}")
    print("\n" + review[:2000])


if __name__ == "__main__":
    main()
