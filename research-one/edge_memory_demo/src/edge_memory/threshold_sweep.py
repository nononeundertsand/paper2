from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from edge_memory.data import SyntheticConfig, SyntheticMemoryDataset, SyntheticWorld
from edge_memory.model import BaseClassifier, ConditionalMemoryModel, ModelConfig
from edge_memory.train import (
    TrainConfig,
    build_loader,
    evaluate_base,
    evaluate_memory,
    get_device,
    print_metrics,
    set_seed,
    train_base,
    train_memory,
)


def parse_thresholds(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = ["threshold"] + [key for key in rows[0].keys() if key != "threshold"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_threshold_sweep(cfg: TrainConfig, thresholds: List[float]) -> Dict[str, object]:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device)
    print(f"Using device: {device}", flush=True)
    print(f"Sweep thresholds: {', '.join(f'{value:.2f}' for value in thresholds)}", flush=True)

    world_cfg = SyntheticConfig(
        num_facts=cfg.num_facts,
        common_fact_ratio=cfg.common_fact_ratio,
        seed=cfg.seed,
    )
    world = SyntheticWorld(world_cfg)
    base_train = SyntheticMemoryDataset(world, "base_train", cfg.base_train_size, cfg.seed + 1)
    memory_train = SyntheticMemoryDataset(world, "memory_train", cfg.memory_train_size, cfg.seed + 2)
    test = SyntheticMemoryDataset(world, "test", cfg.test_size, cfg.seed + 3)

    base_loader = build_loader(base_train, cfg.batch_size, world.vocab.pad_id, shuffle=True)
    memory_loader = build_loader(memory_train, cfg.batch_size, world.vocab.pad_id, shuffle=True)
    test_loader = build_loader(test, cfg.batch_size, world.vocab.pad_id, shuffle=False)

    model_cfg = ModelConfig(
        vocab_size=len(world.vocab),
        num_labels=len(world.labels),
        hidden_size=cfg.hidden_size,
        num_experts=cfg.num_experts,
        top_k_experts=cfg.top_k_experts,
    )

    base = BaseClassifier(model_cfg, world.vocab.pad_id).to(device)
    train_base(base, base_loader, cfg, device)
    base_metrics = evaluate_base(base, test_loader, device)
    print_metrics("base", base_metrics)

    memory_model = ConditionalMemoryModel(base, model_cfg).to(device)
    train_memory(memory_model, memory_loader, cfg, device)
    dense_metrics = evaluate_memory(memory_model, test_loader, device, cfg.threshold, mode="dense")
    print_metrics("dense_memory", dense_metrics)

    rows: List[Dict[str, float]] = []
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        metrics = evaluate_memory(memory_model, test_loader, device, threshold, mode="conditional")
        row = {"threshold": threshold}
        row.update(metrics)
        rows.append(row)
        threshold_metrics[f"{threshold:.4f}"] = metrics
        print_metrics(f"conditional_threshold_{threshold:.2f}", metrics)

    payload = {
        "config": asdict(cfg),
        "thresholds": thresholds,
        "world_config": asdict(world_cfg),
        "model_config": asdict(model_cfg),
        "base_metrics": base_metrics,
        "dense_metrics": dense_metrics,
        "threshold_metrics": threshold_metrics,
        "notes": {
            "common_facts": len(world.common_facts),
            "tail_facts": len(world.tail_facts),
            "vocab_size": len(world.vocab),
            "num_labels": len(world.labels),
        },
    }
    with (output_dir / "threshold_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    save_csv(output_dir / "threshold_sweep.csv", rows)
    torch.save(base.state_dict(), output_dir / "base_model.pt")
    torch.save(memory_model.state_dict(), output_dir / "conditional_memory.pt")
    return payload


def parse_args(argv: Iterable[str]) -> Tuple[TrainConfig, List[float]]:
    parser = argparse.ArgumentParser(description="Sweep read-router thresholds after one training run.")
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--output-dir", default="outputs/threshold_sweep")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-facts", type=int, default=TrainConfig.num_facts)
    parser.add_argument("--common-fact-ratio", type=float, default=TrainConfig.common_fact_ratio)
    parser.add_argument("--base-train-size", type=int, default=TrainConfig.base_train_size)
    parser.add_argument("--memory-train-size", type=int, default=TrainConfig.memory_train_size)
    parser.add_argument("--test-size", type=int, default=TrainConfig.test_size)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--num-experts", type=int, default=TrainConfig.num_experts)
    parser.add_argument("--top-k-experts", type=int, default=TrainConfig.top_k_experts)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--base-epochs", type=int, default=TrainConfig.base_epochs)
    parser.add_argument("--memory-epochs", type=int, default=TrainConfig.memory_epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--router-loss-weight", type=float, default=TrainConfig.router_loss_weight)
    parser.add_argument("--sparsity-weight", type=float, default=TrainConfig.sparsity_weight)
    parser.add_argument("--load-balance-weight", type=float, default=TrainConfig.load_balance_weight)
    parser.add_argument("--device", default=TrainConfig.device)
    args = parser.parse_args(list(argv))

    thresholds = parse_thresholds(args.thresholds)
    cfg = TrainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_facts=args.num_facts,
        common_fact_ratio=args.common_fact_ratio,
        base_train_size=args.base_train_size,
        memory_train_size=args.memory_train_size,
        test_size=args.test_size,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k_experts=args.top_k_experts,
        batch_size=args.batch_size,
        base_epochs=args.base_epochs,
        memory_epochs=args.memory_epochs,
        learning_rate=args.learning_rate,
        router_loss_weight=args.router_loss_weight,
        sparsity_weight=args.sparsity_weight,
        load_balance_weight=args.load_balance_weight,
        threshold=0.5,
        device=args.device,
    )
    return cfg, thresholds


def main(argv: Optional[Iterable[str]] = None) -> None:
    cfg, thresholds = parse_args(sys.argv[1:] if argv is None else argv)
    run_threshold_sweep(cfg, thresholds)


if __name__ == "__main__":
    main()
