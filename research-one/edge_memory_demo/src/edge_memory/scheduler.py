from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SchedulerConfig:
    output_dir: str = "outputs/scheduler_run"
    seed: int = 42
    num_items: int = 400
    storage_budget: float = 120.0
    energy_budget: float = 160.0
    min_utility: float = 0.15


@dataclass(frozen=True)
class KnowledgeItem:
    item_id: int
    freq: float
    gain: float
    stability: float
    retrieval_cost: float
    offline_gain: float
    write_energy: float
    storage_cost: float
    privacy_risk: float
    deletion_risk: float


def generate_items(cfg: SchedulerConfig) -> List[KnowledgeItem]:
    rng = random.Random(cfg.seed)
    items: List[KnowledgeItem] = []
    for idx in range(cfg.num_items):
        hot = rng.random() < 0.25
        freq = rng.uniform(0.45, 1.0) if hot else rng.uniform(0.0, 0.45)
        gain = rng.betavariate(2.0, 3.0)
        stability = rng.betavariate(4.0, 2.0)
        retrieval_cost = rng.uniform(0.1, 1.0)
        offline_gain = rng.uniform(0.0, 1.0)
        write_energy = rng.uniform(0.2, 1.5)
        storage_cost = rng.uniform(0.2, 2.0)
        privacy_risk = rng.betavariate(1.5, 4.0)
        deletion_risk = rng.betavariate(1.5, 5.0)
        items.append(
            KnowledgeItem(
                item_id=idx,
                freq=freq,
                gain=gain,
                stability=stability,
                retrieval_cost=retrieval_cost,
                offline_gain=offline_gain,
                write_energy=write_energy,
                storage_cost=storage_cost,
                privacy_risk=privacy_risk,
                deletion_risk=deletion_risk,
            )
        )
    return items


def consolidation_utility(item: KnowledgeItem) -> float:
    return (
        1.25 * item.freq * item.gain
        + 0.85 * item.freq * item.retrieval_cost
        + 0.55 * item.offline_gain
        + 0.35 * item.stability
        - 0.65 * item.write_energy
        - 0.45 * item.storage_cost
        - 1.20 * item.privacy_risk
        - 0.90 * item.deletion_risk
    )


def frequency_score(item: KnowledgeItem) -> float:
    return item.freq


def all_write_score(item: KnowledgeItem) -> float:
    return 1.0


def select_items(
    items: List[KnowledgeItem],
    cfg: SchedulerConfig,
    strategy: str,
) -> List[KnowledgeItem]:
    if strategy == "resource_aware":
        scored = [(consolidation_utility(item), item) for item in items]
        scored = [(score, item) for score, item in scored if score >= cfg.min_utility]
    elif strategy == "frequency_only":
        scored = [(frequency_score(item), item) for item in items]
    elif strategy == "all_write":
        scored = [(all_write_score(item), item) for item in items]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    scored.sort(key=lambda pair: pair[0] / max(pair[1].storage_cost + pair[1].write_energy, 1e-6), reverse=True)

    selected: List[KnowledgeItem] = []
    used_storage = 0.0
    used_energy = 0.0
    for score, item in scored:
        if strategy == "frequency_only" and item.freq < 0.45:
            continue
        next_storage = used_storage + item.storage_cost
        next_energy = used_energy + item.write_energy
        if next_storage > cfg.storage_budget or next_energy > cfg.energy_budget:
            continue
        selected.append(item)
        used_storage = next_storage
        used_energy = next_energy
    return selected


def summarize(items: List[KnowledgeItem]) -> Dict[str, float]:
    if not items:
        return {
            "num_written": 0,
            "expected_accuracy_gain": 0.0,
            "retrieval_saving": 0.0,
            "storage": 0.0,
            "write_energy": 0.0,
            "privacy_risk": 0.0,
            "deletion_risk": 0.0,
            "mean_utility": 0.0,
            "total_utility": 0.0,
            "gain_per_storage": 0.0,
            "saving_per_energy": 0.0,
        }
    expected_accuracy_gain = sum(item.freq * item.gain for item in items)
    retrieval_saving = sum(item.freq * item.retrieval_cost for item in items)
    storage = sum(item.storage_cost for item in items)
    write_energy = sum(item.write_energy for item in items)
    total_utility = sum(consolidation_utility(item) for item in items)
    return {
        "num_written": len(items),
        "expected_accuracy_gain": expected_accuracy_gain,
        "retrieval_saving": retrieval_saving,
        "storage": storage,
        "write_energy": write_energy,
        "privacy_risk": sum(item.privacy_risk for item in items) / len(items),
        "deletion_risk": sum(item.deletion_risk for item in items) / len(items),
        "mean_utility": total_utility / len(items),
        "total_utility": total_utility,
        "gain_per_storage": expected_accuracy_gain / max(storage, 1e-8),
        "saving_per_energy": retrieval_saving / max(write_energy, 1e-8),
    }


def run(cfg: SchedulerConfig) -> Dict[str, Dict[str, float]]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = generate_items(cfg)
    results = {}
    for strategy in ["all_write", "frequency_only", "resource_aware"]:
        selected = select_items(items, cfg, strategy)
        results[strategy] = summarize(selected)

    payload = {
        "config": asdict(cfg),
        "results": results,
        "description": "A trace-free simulation for resource-aware episodic-to-parametric memory consolidation.",
    }
    with (output_dir / "scheduler_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for strategy, metrics in results.items():
        text = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"[scheduler:{strategy}] {text}", flush=True)
    return results


def parse_args(argv: Iterable[str]) -> SchedulerConfig:
    parser = argparse.ArgumentParser(description="Resource-aware memory consolidation simulation.")
    parser.add_argument("--output-dir", default=SchedulerConfig.output_dir)
    parser.add_argument("--seed", type=int, default=SchedulerConfig.seed)
    parser.add_argument("--num-items", type=int, default=SchedulerConfig.num_items)
    parser.add_argument("--storage-budget", type=float, default=SchedulerConfig.storage_budget)
    parser.add_argument("--energy-budget", type=float, default=SchedulerConfig.energy_budget)
    parser.add_argument("--min-utility", type=float, default=SchedulerConfig.min_utility)
    args = parser.parse_args(list(argv))
    return SchedulerConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_items=args.num_items,
        storage_budget=args.storage_budget,
        energy_budget=args.energy_budget,
        min_utility=args.min_utility,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
