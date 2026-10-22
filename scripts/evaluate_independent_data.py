#!/usr/bin/env python3
"""Evaluate a private prediction/label table without exporting raw rows.

The input path should point to private normalized evaluation storage. The
output is an aggregate JSON report containing metrics, approved subgroup
aggregates, and the input file hash; it never copies event or customer IDs.

Example:
    python scripts/evaluate_independent_data.py \
      --input /private/evaluation/real_eval_v1.csv \
      --output reports/real_eval_v1_summary.json \
      --as-of 2026-10-20T00:00:00Z \
      --evaluation-type real_historical_temporal \
      --subgroup-field merchant_age_band
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from backend.services.independent_evaluation import EvaluationInputError, evaluate_prediction_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise EvaluationInputError("input must be a CSV, JSON, or JSONL file")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Private prediction/label table")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON report path")
    parser.add_argument("--as-of", required=True, help="UTC evaluation cutoff")
    parser.add_argument("--evaluation-type", required=True, help="Named evaluation lane")
    parser.add_argument("--subgroup-field", action="append", default=[], help="Approved subgroup field; may repeat")
    parser.add_argument("--review-budget", action="append", type=float, default=[0.02, 0.05], help="Review fraction; may repeat")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")
    try:
        frame = _read_input(input_path)
        report = evaluate_prediction_frame(
            frame,
            as_of=args.as_of,
            evaluation_type=args.evaluation_type,
            subgroup_fields=args.subgroup_field,
            review_budgets=args.review_budget,
        )
    except EvaluationInputError as exc:
        raise SystemExit(f"Evaluation input rejected: {exc}") from exc

    report["dataset_manifest"] = {
        "source_name": input_path.name,
        "sha256": _sha256(input_path),
        "raw_rows_exported": False,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "sample_count": report["metrics"]["sample_count"], "raw_rows_exported": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
