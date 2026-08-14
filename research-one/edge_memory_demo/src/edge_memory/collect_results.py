from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def flatten_metrics(prefix: str, metrics: Dict[str, float], row: Dict[str, object]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def collect_one(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: List[Dict[str, object]] = []
    config = payload.get("config", {})
    base_metrics = payload.get("base_metrics", {})
    dense_metrics = payload.get("dense_metrics", {})
    threshold_metrics = payload.get("threshold_metrics", {})

    if threshold_metrics:
        for threshold, metrics in threshold_metrics.items():
            row: Dict[str, object] = {
                "source": str(path),
                "run_dir": str(path.parent),
                "threshold": threshold,
            }
            for key in [
                "model_name_or_path",
                "num_facts",
                "num_experts",
                "top_k_experts",
                "router_label_noise",
                "base_train_size",
                "memory_train_size",
                "test_size",
                "base_epochs",
                "memory_epochs",
                "learning_rate",
            ]:
                if key in config:
                    row[key] = config[key]
            flatten_metrics("base", base_metrics, row)
            flatten_metrics("dense", dense_metrics, row)
            flatten_metrics("conditional", metrics, row)
            rows.append(row)
    else:
        row = {"source": str(path), "run_dir": str(path.parent)}
        rows.append(row)
    return rows


def run(input_dir: Path, output_path: Path) -> None:
    json_files = sorted(input_dir.rglob("llm_synthetic_metrics.json"))
    if not json_files:
        raise FileNotFoundError(f"No llm_synthetic_metrics.json files found under {input_dir}")

    rows: List[Dict[str, object]] = []
    for path in json_files:
        rows.extend(collect_one(path))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}", flush=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LLM synthetic experiment JSON files into one CSV.")
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output", default="outputs/summary/llm_results_summary.csv")
    return parser.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(Path(args.input_dir), Path(args.output))


if __name__ == "__main__":
    main()

