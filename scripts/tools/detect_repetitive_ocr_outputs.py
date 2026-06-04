#!/usr/bin/env python3
"""Detect repetitive OCR/VLM outputs in MinerU-style sample directories."""

from __future__ import annotations

import argparse
import csv
import re
import zlib
from pathlib import Path


TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]")
SPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def compression_ratio(text: str) -> float:
    normalized = SPACE_RE.sub(" ", text).strip()
    if not normalized:
        return 1.0
    encoded = normalized.encode("utf-8", errors="ignore")
    return len(zlib.compress(encoded)) / len(encoded)


def max_same_token_run(tokens: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        previous = token
        longest = max(longest, current)
    return longest


def max_repeated_ngram(tokens: list[str], max_n: int = 12) -> tuple[int, int, int, str]:
    """Return repeat count, ngram size, repeated token span, and example sequence."""
    best_repeats = 0
    best_n = 0
    best_span = 0
    best_seq = ""
    token_count = len(tokens)

    for ngram_size in range(1, max_n + 1):
        if token_count < ngram_size * 2:
            continue

        windows = [
            tuple(tokens[index : index + ngram_size])
            for index in range(0, token_count - ngram_size + 1)
        ]
        index = 0
        while index < len(windows):
            repeats = 1
            while (
                index + repeats * ngram_size < len(windows)
                and windows[index + repeats * ngram_size] == windows[index]
            ):
                repeats += 1

            span = repeats * ngram_size
            if repeats >= 2 and span > best_span:
                best_repeats = repeats
                best_n = ngram_size
                best_span = span
                best_seq = " ".join(windows[index])

            index += repeats * ngram_size if repeats > 1 else 1

    return best_repeats, best_n, best_span, best_seq


def max_nonspace_char_run(text: str) -> tuple[int, str]:
    normalized = SPACE_RE.sub(" ", text)
    longest = 0
    longest_char = ""
    for match in re.finditer(r"(\S)\1{9,}", normalized):
        run_len = len(match.group(0))
        if run_len > longest:
            longest = run_len
            longest_char = match.group(1)
    return longest, longest_char


def classify(metrics: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    same_run = int(metrics["same_token_run"])
    ngram_span = int(metrics["repeated_ngram_span"])
    ngram_repeats = int(metrics["repeated_ngram_repeats"])
    compression = float(metrics["compression_ratio"])
    token_count = int(metrics["token_count"])
    char_run = int(metrics["char_run"])
    repeated_example = str(metrics["repeated_ngram_example"])

    if same_run >= 50:
        reasons.append(f"same_token_run={same_run}")
    if ngram_span >= 140:
        reasons.append(
            "repeated_ngram_span="
            f"{ngram_span}(n={metrics['repeated_ngram_n']},rep={ngram_repeats})"
        )
    if ngram_repeats >= 12 and ngram_span >= 80 and compression <= 0.25:
        reasons.append(f"low_entropy_repeated_ngram={ngram_span}")
    if compression <= 0.12 and ngram_span >= 35 and token_count >= 500:
        reasons.append(f"very_low_compression={compression:.3f}")
    if char_run >= 80:
        reasons.append(f"char_run={char_run}x{metrics['char_run_token']}")

    # Dotted TOCs, sports-rating leader lines, and page dividers are legitimate
    # document content in OmniDocBench. Keep other repeated punctuation such as
    # plus signs because the audited cases are formula/table decoding loops.
    leader_tokens = {".", "\u00b7", "\u2014"}
    if (
        repeated_example in leader_tokens
        and ngram_span == same_run
        and compression > 0.18
    ):
        return []

    return reasons


def analyze_markdown(path: Path, root: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = tokenize(text)
    repeated_count, repeated_n, repeated_span, repeated_seq = max_repeated_ngram(tokens)
    char_run, char_run_token = max_nonspace_char_run(text)

    sample_dir = path.parents[1]
    metrics: dict[str, object] = {
        "sample_id": sample_dir.name,
        "md_path": str(path),
        "relative_md_path": str(path.relative_to(root)),
        "char_count": len(text),
        "token_count": len(tokens),
        "compression_ratio": compression_ratio(text),
        "same_token_run": max_same_token_run(tokens),
        "repeated_ngram_repeats": repeated_count,
        "repeated_ngram_n": repeated_n,
        "repeated_ngram_span": repeated_span,
        "repeated_ngram_example": repeated_seq,
        "char_run": char_run,
        "char_run_token": char_run_token,
    }
    metrics["reasons"] = ";".join(classify(metrics))
    return metrics


def iter_markdown_outputs(root: Path) -> list[Path]:
    return sorted(root.glob("*/vlm/*.md"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run output root containing sample/vlm/*.md files")
    parser.add_argument("--tsv", type=Path, required=True, help="Path for detailed TSV output")
    parser.add_argument("--list", type=Path, required=True, help="Path for sample-id list output")
    args = parser.parse_args()

    root = args.root.resolve()
    rows = []
    for markdown_path in iter_markdown_outputs(root):
        metrics = analyze_markdown(markdown_path, root)
        if metrics["reasons"]:
            rows.append(metrics)

    rows.sort(
        key=lambda row: (
            -int(row["repeated_ngram_span"]),
            -int(row["same_token_run"]),
            str(row["sample_id"]),
        )
    )

    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    args.list.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "reasons",
        "token_count",
        "char_count",
        "compression_ratio",
        "same_token_run",
        "repeated_ngram_span",
        "repeated_ngram_n",
        "repeated_ngram_repeats",
        "repeated_ngram_example",
        "char_run",
        "char_run_token",
        "relative_md_path",
        "md_path",
    ]
    with args.tsv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    with args.list.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(f"{row['sample_id']}\n")

    print(f"scanned={len(iter_markdown_outputs(root))} flagged={len(rows)}")
    print(f"tsv={args.tsv}")
    print(f"list={args.list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
