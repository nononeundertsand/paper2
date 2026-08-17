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


def experiment_type_from_path(path: Path) -> str:
    if path.name == "llm_synthetic_metrics.json":
        return "llm_synthetic"
    if path.name == "real_qa_metrics.json":
        return "real_qa"
    if path.name == "scheduler_metrics.json":
        return "scheduler"
    return "unknown"


def collect_memory_result(path: Path) -> List[Dict[str, object]]:
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
                "experiment_type": experiment_type_from_path(path),
                "source": str(path),
                "run_dir": str(path.parent),
                "threshold": threshold,
            }
            for key in [
                "model_name_or_path",
                "seed",
                "dataset_name",
                "dataset_config",
                "dataset_split",
                "general_source",
                "general_dataset_name",
                "general_dataset_config",
                "general_dataset_split",
                "num_facts",
                "common_fact_ratio",
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
            if "num_labels" in payload:
                row["num_labels"] = payload["num_labels"]
            if "num_common_facts" in payload:
                row["num_common_facts"] = payload["num_common_facts"]
            if "num_tail_facts" in payload:
                row["num_tail_facts"] = payload["num_tail_facts"]
            if "num_general_labels" in payload:
                row["num_general_labels"] = payload["num_general_labels"]
            flatten_metrics("base", base_metrics, row)
            flatten_metrics("dense", dense_metrics, row)
            flatten_metrics("conditional", metrics, row)
            if "conditional_activation_rate" in row and "top_k_experts" in row:
                row["conditional_expert_cost"] = float(row["conditional_activation_rate"]) * float(row["top_k_experts"])
            rows.append(row)
    else:
        row = {"experiment_type": experiment_type_from_path(path), "source": str(path), "run_dir": str(path.parent)}
        rows.append(row)
    return rows


def collect_scheduler_result(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows: List[Dict[str, object]] = []
    config = payload.get("config", {})
    for strategy, metrics in payload.get("results", {}).items():
        row: Dict[str, object] = {
            "experiment_type": "scheduler",
            "source": str(path),
            "run_dir": str(path.parent),
            "strategy": strategy,
        }
        for key, value in config.items():
            row[key] = value
        flatten_metrics("scheduler", metrics, row)
        rows.append(row)
    return rows


def collect_one(path: Path) -> List[Dict[str, object]]:
    if path.name == "scheduler_metrics.json":
        return collect_scheduler_result(path)
    return collect_memory_result(path)


def run(input_dir: Path, output_path: Path, include_scheduler: bool) -> None:
    patterns = ["llm_synthetic_metrics.json", "real_qa_metrics.json"]
    if include_scheduler:
        patterns.append("scheduler_metrics.json")
    json_files: List[Path] = []
    for pattern in patterns:
        json_files.extend(input_dir.rglob(pattern))
    json_files = sorted(set(json_files))
    if not json_files:
        raise FileNotFoundError(f"No experiment metric JSON files found under {input_dir}")

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
    parser = argparse.ArgumentParser(description="Collect experiment JSON files into one CSV.")
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output", default="outputs/summary/results_summary.csv")
    parser.add_argument("--include-scheduler", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(Path(args.input_dir), Path(args.output), include_scheduler=args.include_scheduler)


if __name__ == "__main__":
    main()
