from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_GROUP_KEYS = [
    "experiment_type",
    "dataset_name",
    "general_dataset_name",
    "num_facts",
    "num_experts",
    "top_k_experts",
    "router_label_noise",
    "threshold",
    "strategy",
]

DEFAULT_METRICS = [
    "base_accuracy",
    "base_acc_common_fact",
    "base_acc_tail_fact",
    "base_acc_general",
    "base_rank_qa_em",
    "base_rank_qa_f1",
    "dense_accuracy",
    "dense_acc_common_fact",
    "dense_acc_tail_fact",
    "dense_acc_general",
    "dense_rank_qa_em",
    "dense_rank_qa_f1",
    "conditional_accuracy",
    "conditional_acc_common_fact",
    "conditional_acc_tail_fact",
    "conditional_acc_general",
    "conditional_activation_rate",
    "conditional_router_f1",
    "conditional_rank_qa_em",
    "conditional_rank_qa_f1",
    "conditional_rank_qa_mrr",
    "conditional_rank_qa_hits5",
    "conditional_expert_cost",
    "scheduler_total_utility",
    "scheduler_gain_per_storage",
    "scheduler_saving_per_energy",
]


def maybe_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def run(input_path: Path, output_path: Path, group_keys: List[str], metrics: List[str]) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups[key].append(row)

    output_rows: List[dict] = []
    for key, group_rows in groups.items():
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        out["n"] = len(group_rows)
        for metric in metrics:
            values = [value for value in (maybe_float(row.get(metric, "")) for row in group_rows) if value is not None]
            if values:
                mean, std = mean_std(values)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
        output_rows.append(out)

    fieldnames = sorted({key for row in output_rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} aggregate rows to {output_path}", flush=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment summary CSV into mean/std tables.")
    parser.add_argument("--input", default="outputs/e2e/summary/e2e_full_summary.csv")
    parser.add_argument("--output", default="outputs/e2e/summary/e2e_full_aggregate.csv")
    parser.add_argument("--group-keys", default=",".join(DEFAULT_GROUP_KEYS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    return parser.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    group_keys = [item.strip() for item in args.group_keys.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    run(Path(args.input), Path(args.output), group_keys, metrics)


if __name__ == "__main__":
    main()

