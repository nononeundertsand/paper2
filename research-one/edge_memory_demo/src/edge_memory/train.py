from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from edge_memory.data import (
    SAMPLE_COMMON_FACT,
    SAMPLE_GENERAL,
    SAMPLE_TAIL_FACT,
    SAMPLE_TYPE_NAMES,
    SyntheticConfig,
    SyntheticMemoryDataset,
    SyntheticWorld,
    collate_batch,
)
from edge_memory.model import BaseClassifier, ConditionalMemoryModel, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "outputs/synthetic_run"
    seed: int = 42
    num_facts: int = 240
    common_fact_ratio: float = 0.65
    base_train_size: int = 6000
    memory_train_size: int = 7000
    test_size: int = 2500
    hidden_size: int = 128
    num_experts: int = 8
    top_k_experts: int = 2
    batch_size: int = 128
    base_epochs: int = 8
    memory_epochs: int = 10
    learning_rate: float = 2e-3
    router_loss_weight: float = 0.6
    sparsity_weight: float = 0.03
    load_balance_weight: float = 0.01
    router_label_noise: float = 0.0
    threshold: float = 0.5
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loader(dataset: SyntheticMemoryDataset, batch_size: int, pad_id: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_batch, pad_id=pad_id),
    )


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def train_base(
    model: BaseClassifier,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    for epoch in range(1, cfg.base_epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(out["logits"], batch["labels"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch["labels"].numel()
            total += batch["labels"].numel()
            correct += (out["logits"].argmax(dim=-1) == batch["labels"]).sum().item()

        print(
            f"[base] epoch={epoch:02d} loss={total_loss / max(total, 1):.4f} "
            f"acc={correct / max(total, 1):.4f}",
            flush=True,
        )


def load_balance_loss(expert_scores: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(expert_scores, dim=-1)
    mean_probs = probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    return F.mse_loss(mean_probs, target)


def train_memory(
    model: ConditionalMemoryModel,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.01)

    for epoch in range(1, cfg.memory_epochs + 1):
        total_loss = 0.0
        total_task = 0.0
        total_router = 0.0
        total_read = 0.0
        total = 0

        for batch in loader:
            batch = batch_to_device(batch, device)
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                threshold=cfg.threshold,
                soft_read=True,
            )
            task_loss = F.cross_entropy(out["final_logits"], batch["labels"])
            read_labels = batch["read_labels"]
            if cfg.router_label_noise > 0.0:
                flip_mask = torch.rand_like(read_labels) < cfg.router_label_noise
                read_labels = torch.where(flip_mask, 1.0 - read_labels, read_labels)
            router_loss = F.binary_cross_entropy_with_logits(out["read_logit"], read_labels)
            sparsity_loss = out["read_prob"].mean()
            balance_loss = load_balance_loss(out["expert_scores"])
            loss = (
                task_loss
                + cfg.router_loss_weight * router_loss
                + cfg.sparsity_weight * sparsity_loss
                + cfg.load_balance_weight * balance_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            batch_size = batch["labels"].numel()
            total_loss += loss.item() * batch_size
            total_task += task_loss.item() * batch_size
            total_router += router_loss.item() * batch_size
            total_read += out["read_prob"].sum().item()
            total += batch_size

        print(
            f"[memory] epoch={epoch:02d} loss={total_loss / max(total, 1):.4f} "
            f"task={total_task / max(total, 1):.4f} router={total_router / max(total, 1):.4f} "
            f"mean_read_prob={total_read / max(total, 1):.4f}",
            flush=True,
        )


@torch.no_grad()
def evaluate_base(model: BaseClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    return _evaluate_logits(
        loader=loader,
        device=device,
        predict_fn=lambda batch: {
            "logits": model(batch["input_ids"], batch["attention_mask"])["logits"],
            "read_prob": torch.zeros_like(batch["labels"], dtype=torch.float32),
            "read_mask": torch.zeros_like(batch["labels"], dtype=torch.float32),
        },
    )


@torch.no_grad()
def evaluate_memory(
    model: ConditionalMemoryModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    mode: str,
) -> Dict[str, float]:
    model.eval()

    def predict_fn(batch: dict) -> dict:
        force_read = None
        soft_read = False
        if mode == "base":
            force_read = False
        elif mode == "dense":
            force_read = True
        elif mode == "conditional":
            force_read = None
            soft_read = False
        elif mode == "soft":
            force_read = None
            soft_read = True
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            threshold=threshold,
            force_read=force_read,
            soft_read=soft_read,
        )
        return {
            "logits": out["final_logits"],
            "read_prob": out["read_prob"],
            "read_mask": out["read_mask"],
        }

    return _evaluate_logits(loader=loader, device=device, predict_fn=predict_fn)


def _evaluate_logits(loader: DataLoader, device: torch.device, predict_fn) -> Dict[str, float]:
    total = 0
    correct = 0
    read_sum = 0.0
    read_mask_sum = 0.0
    router_tp = router_fp = router_fn = 0
    by_type = {
        SAMPLE_COMMON_FACT: [0, 0],
        SAMPLE_TAIL_FACT: [0, 0],
        SAMPLE_GENERAL: [0, 0],
    }

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = predict_fn(batch)
        pred = out["logits"].argmax(dim=-1)
        labels = batch["labels"]
        is_correct = pred == labels

        total += labels.numel()
        correct += is_correct.sum().item()
        read_sum += out["read_prob"].sum().item()
        read_mask_sum += out["read_mask"].sum().item()

        route_pred = out["read_mask"] > 0.5
        route_gold = batch["read_labels"] > 0.5
        router_tp += (route_pred & route_gold).sum().item()
        router_fp += (route_pred & ~route_gold).sum().item()
        router_fn += (~route_pred & route_gold).sum().item()

        for sample_type in by_type:
            mask = batch["sample_types"] == sample_type
            by_type[sample_type][0] += (is_correct & mask).sum().item()
            by_type[sample_type][1] += mask.sum().item()

    precision = router_tp / max(router_tp + router_fp, 1)
    recall = router_tp / max(router_tp + router_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        "accuracy": correct / max(total, 1),
        "mean_read_prob": read_sum / max(total, 1),
        "activation_rate": read_mask_sum / max(total, 1),
        "router_precision": precision,
        "router_recall": recall,
        "router_f1": f1,
    }
    for sample_type, (type_correct, type_total) in by_type.items():
        metrics[f"acc_{SAMPLE_TYPE_NAMES[sample_type]}"] = type_correct / max(type_total, 1)
    return metrics


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    ordered = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"[eval:{name}] {ordered}", flush=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run(cfg: TrainConfig) -> Dict[str, Dict[str, float]]:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device)
    print(f"Using device: {device}", flush=True)

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

    metrics = {
        "base": base_metrics,
        "dense_memory": evaluate_memory(memory_model, test_loader, device, cfg.threshold, mode="dense"),
        "conditional_memory": evaluate_memory(memory_model, test_loader, device, cfg.threshold, mode="conditional"),
        "soft_conditional_memory": evaluate_memory(memory_model, test_loader, device, cfg.threshold, mode="soft"),
    }
    for name, value in metrics.items():
        if name != "base":
            print_metrics(name, value)

    payload = {
        "config": asdict(cfg),
        "world_config": asdict(world_cfg),
        "model_config": asdict(model_cfg),
        "metrics": metrics,
        "notes": {
            "common_facts": len(world.common_facts),
            "tail_facts": len(world.tail_facts),
            "vocab_size": len(world.vocab),
            "num_labels": len(world.labels),
        },
    }
    save_json(output_dir / "metrics.json", payload)
    torch.save(base.state_dict(), output_dir / "base_model.pt")
    torch.save(memory_model.state_dict(), output_dir / "conditional_memory.pt")
    return metrics


def parse_args(argv: Iterable[str]) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Synthetic edge parametric memory feasibility demo.")
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
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
    parser.add_argument("--router-label-noise", type=float, default=TrainConfig.router_label_noise)
    parser.add_argument("--threshold", type=float, default=TrainConfig.threshold)
    parser.add_argument("--device", default=TrainConfig.device)
    args = parser.parse_args(list(argv))
    return TrainConfig(
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
        router_label_noise=args.router_label_noise,
        threshold=args.threshold,
        device=args.device,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    run(cfg)


if __name__ == "__main__":
    main()
