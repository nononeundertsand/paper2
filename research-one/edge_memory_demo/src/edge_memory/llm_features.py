from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from edge_memory.data import (
    SAMPLE_COMMON_FACT,
    SAMPLE_GENERAL,
    SAMPLE_TAIL_FACT,
    SAMPLE_TYPE_NAMES,
    SyntheticConfig,
    SyntheticMemoryDataset,
    SyntheticWorld,
)
from edge_memory.model import MemoryExpert
from edge_memory.threshold_sweep import parse_thresholds, save_csv
from edge_memory.train import TrainConfig, batch_to_device, get_device, print_metrics, set_seed


@dataclass(frozen=True)
class LLMFeatureConfig:
    model_name_or_path: str
    output_dir: str = "outputs/llm_synthetic"
    seed: int = 42
    num_facts: int = 160
    common_fact_ratio: float = 0.65
    base_train_size: int = 1000
    memory_train_size: int = 1400
    test_size: int = 700
    num_experts: int = 8
    top_k_experts: int = 2
    batch_size: int = 64
    feature_batch_size: int = 16
    base_epochs: int = 8
    memory_epochs: int = 10
    learning_rate: float = 2e-3
    router_loss_weight: float = 0.6
    sparsity_weight: float = 0.03
    load_balance_weight: float = 0.01
    router_label_noise: float = 0.0
    max_length: int = 64
    pooling: str = "mean"
    thresholds: str = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
    device: str = "auto"
    fp16: bool = False
    trust_remote_code: bool = False


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        read_labels: torch.Tensor,
        sample_types: torch.Tensor,
    ) -> None:
        self.features = features.float()
        self.labels = labels.long()
        self.read_labels = read_labels.float()
        self.sample_types = sample_types.long()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> dict:
        return {
            "features": self.features[index],
            "labels": self.labels[index],
            "read_labels": self.read_labels[index],
            "sample_types": self.sample_types[index],
        }


class FeatureBaseClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"hidden": features, "logits": self.classifier(features)}


class FeatureConditionalMemoryModel(nn.Module):
    def __init__(
        self,
        base: FeatureBaseClassifier,
        hidden_size: int,
        num_labels: int,
        num_experts: int,
        top_k_experts: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        for param in self.base.parameters():
            param.requires_grad = False

        self.read_router = nn.Sequential(
            nn.LayerNorm(hidden_size + 2),
            nn.Linear(hidden_size + 2, max(hidden_size // 2, 32)),
            nn.GELU(),
            nn.Linear(max(hidden_size // 2, 32), 1),
        )
        self.expert_router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_experts),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.experts = nn.ModuleList([MemoryExpert(hidden_size, num_labels, 0.1) for _ in range(num_experts)])

    def forward(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
        force_read: Optional[bool] = None,
        soft_read: bool = True,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_out = self.base(features)
            hidden = base_out["hidden"]
            base_logits = base_out["logits"]

        base_probs = F.softmax(base_logits, dim=-1)
        entropy = -(base_probs * base_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        top2 = torch.topk(base_logits, k=2, dim=-1).values
        logit_gap = top2[:, :1] - top2[:, 1:2]
        read_features = torch.cat([hidden, entropy, logit_gap], dim=-1)
        read_logit = self.read_router(read_features).squeeze(-1)
        read_prob = torch.sigmoid(read_logit)

        expert_scores = self.expert_router(hidden)
        k = min(self.top_k_experts, self.num_experts)
        top_values, top_indices = torch.topk(expert_scores, k=k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)
        expert_logits = torch.stack([expert(hidden) for expert in self.experts], dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.num_labels)
        selected_logits = torch.gather(expert_logits, dim=1, index=gather_index)
        memory_logits = (selected_logits * top_weights.unsqueeze(-1)).sum(dim=1)

        alpha = torch.sigmoid(self.alpha_head(hidden))
        mixed_logits = alpha * base_logits + (1.0 - alpha) * memory_logits

        if force_read is True:
            final_logits = mixed_logits
            read_mask = torch.ones_like(read_prob)
        elif force_read is False:
            final_logits = base_logits
            read_mask = torch.zeros_like(read_prob)
        elif soft_read:
            final_logits = read_prob.unsqueeze(-1) * mixed_logits + (1.0 - read_prob.unsqueeze(-1)) * base_logits
            read_mask = (read_prob >= threshold).float()
        else:
            read_mask = (read_prob >= threshold).float()
            final_logits = read_mask.unsqueeze(-1) * mixed_logits + (1.0 - read_mask.unsqueeze(-1)) * base_logits

        return {
            "base_logits": base_logits,
            "memory_logits": memory_logits,
            "final_logits": final_logits,
            "read_logit": read_logit,
            "read_prob": read_prob,
            "read_mask": read_mask,
            "expert_scores": expert_scores,
        }


def collate_features(samples: List[dict]) -> dict:
    return {
        "features": torch.stack([sample["features"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
        "read_labels": torch.stack([sample["read_labels"] for sample in samples]),
        "sample_types": torch.stack([sample["sample_types"] for sample in samples]),
    }


def load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for real LLM experiments. "
            "Install it with: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def extract_features(
    raw_dataset: SyntheticMemoryDataset,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> FeatureDataset:
    texts = [sample["text"] for sample in raw_dataset.samples]
    labels = torch.stack([sample["label"] for sample in raw_dataset.samples])
    read_labels = torch.stack([sample["read_label"] for sample in raw_dataset.samples])
    sample_types = torch.stack([sample["sample_type"] for sample in raw_dataset.samples])
    features: List[torch.Tensor] = []

    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        elif pooling == "last":
            lengths = encoded["attention_mask"].sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), lengths]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        features.append(pooled.detach().float().cpu())

    return FeatureDataset(
        features=torch.cat(features, dim=0),
        labels=labels,
        read_labels=read_labels,
        sample_types=sample_types,
    )


def build_loader(dataset: FeatureDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_features)


def load_balance_loss(expert_scores: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(expert_scores, dim=-1)
    mean_probs = probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    return F.mse_loss(mean_probs, target)


def train_base(model: FeatureBaseClassifier, loader: DataLoader, cfg: LLMFeatureConfig, device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    for epoch in range(1, cfg.base_epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            out = model(batch["features"])
            loss = F.cross_entropy(out["logits"], batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["labels"].numel()
            total += batch["labels"].numel()
            correct += (out["logits"].argmax(dim=-1) == batch["labels"]).sum().item()
        print(f"[llm-base] epoch={epoch:02d} loss={total_loss / max(total, 1):.4f} acc={correct / max(total, 1):.4f}", flush=True)


def train_memory(
    model: FeatureConditionalMemoryModel,
    loader: DataLoader,
    cfg: LLMFeatureConfig,
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
            out = model(batch["features"], soft_read=True)
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
            f"[llm-memory] epoch={epoch:02d} loss={total_loss / max(total, 1):.4f} "
            f"task={total_task / max(total, 1):.4f} router={total_router / max(total, 1):.4f} "
            f"mean_read_prob={total_read / max(total, 1):.4f}",
            flush=True,
        )


@torch.no_grad()
def evaluate(loader: DataLoader, device: torch.device, predict_fn) -> Dict[str, float]:
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


def evaluate_base(model: FeatureBaseClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    return evaluate(
        loader,
        device,
        lambda batch: {
            "logits": model(batch["features"])["logits"],
            "read_prob": torch.zeros_like(batch["labels"], dtype=torch.float32),
            "read_mask": torch.zeros_like(batch["labels"], dtype=torch.float32),
        },
    )


def evaluate_memory(
    model: FeatureConditionalMemoryModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    mode: str,
) -> Dict[str, float]:
    model.eval()

    def predict(batch: dict) -> dict:
        if mode == "dense":
            out = model(batch["features"], threshold=threshold, force_read=True, soft_read=False)
        elif mode == "conditional":
            out = model(batch["features"], threshold=threshold, force_read=None, soft_read=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return {"logits": out["final_logits"], "read_prob": out["read_prob"], "read_mask": out["read_mask"]}

    return evaluate(loader, device, predict)


def run(cfg: LLMFeatureConfig) -> Dict[str, object]:
    AutoModelForCausalLM, AutoTokenizer = load_transformers()
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device)
    dtype = torch.float16 if cfg.fp16 and device.type == "cuda" else torch.float32
    print(f"Using device: {device}", flush=True)
    print(f"Loading LLM: {cfg.model_name_or_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    ).to(device)
    llm.eval()

    world_cfg = SyntheticConfig(num_facts=cfg.num_facts, common_fact_ratio=cfg.common_fact_ratio, seed=cfg.seed)
    world = SyntheticWorld(world_cfg)
    raw_base = SyntheticMemoryDataset(world, "base_train", cfg.base_train_size, cfg.seed + 1)
    raw_memory = SyntheticMemoryDataset(world, "memory_train", cfg.memory_train_size, cfg.seed + 2)
    raw_test = SyntheticMemoryDataset(world, "test", cfg.test_size, cfg.seed + 3)

    print("Extracting LLM features...", flush=True)
    base_dataset = extract_features(raw_base, llm, tokenizer, device, cfg.feature_batch_size, cfg.max_length, cfg.pooling)
    memory_dataset = extract_features(raw_memory, llm, tokenizer, device, cfg.feature_batch_size, cfg.max_length, cfg.pooling)
    test_dataset = extract_features(raw_test, llm, tokenizer, device, cfg.feature_batch_size, cfg.max_length, cfg.pooling)
    del llm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hidden_size = base_dataset.features.size(-1)
    base_loader = build_loader(base_dataset, cfg.batch_size, shuffle=True)
    memory_loader = build_loader(memory_dataset, cfg.batch_size, shuffle=True)
    test_loader = build_loader(test_dataset, cfg.batch_size, shuffle=False)

    base = FeatureBaseClassifier(hidden_size, len(world.labels)).to(device)
    train_base(base, base_loader, cfg, device)
    base_metrics = evaluate_base(base, test_loader, device)
    print_metrics("llm_feature_base", base_metrics)

    memory_model = FeatureConditionalMemoryModel(
        base=base,
        hidden_size=hidden_size,
        num_labels=len(world.labels),
        num_experts=cfg.num_experts,
        top_k_experts=cfg.top_k_experts,
    ).to(device)
    train_memory(memory_model, memory_loader, cfg, device)

    dense_metrics = evaluate_memory(memory_model, test_loader, device, 0.5, mode="dense")
    print_metrics("llm_feature_dense_memory", dense_metrics)

    thresholds = parse_thresholds(cfg.thresholds)
    rows: List[Dict[str, float]] = []
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        metrics = evaluate_memory(memory_model, test_loader, device, threshold, mode="conditional")
        row = {"threshold": threshold}
        row.update(metrics)
        rows.append(row)
        threshold_metrics[f"{threshold:.4f}"] = metrics
        print_metrics(f"llm_feature_conditional_threshold_{threshold:.2f}", metrics)

    payload = {
        "config": asdict(cfg),
        "world_config": asdict(world_cfg),
        "hidden_size": hidden_size,
        "num_labels": len(world.labels),
        "base_metrics": base_metrics,
        "dense_metrics": dense_metrics,
        "threshold_metrics": threshold_metrics,
    }
    with (output_dir / "llm_synthetic_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    save_csv(output_dir / "llm_threshold_sweep.csv", rows)
    torch.save(base.state_dict(), output_dir / "llm_feature_base.pt")
    torch.save(memory_model.state_dict(), output_dir / "llm_feature_conditional_memory.pt")
    return payload


def parse_args(argv: Iterable[str]) -> LLMFeatureConfig:
    parser = argparse.ArgumentParser(description="Real LLM hidden-state validation for conditional parametric memory.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", default=LLMFeatureConfig.output_dir)
    parser.add_argument("--seed", type=int, default=LLMFeatureConfig.seed)
    parser.add_argument("--num-facts", type=int, default=LLMFeatureConfig.num_facts)
    parser.add_argument("--common-fact-ratio", type=float, default=LLMFeatureConfig.common_fact_ratio)
    parser.add_argument("--base-train-size", type=int, default=LLMFeatureConfig.base_train_size)
    parser.add_argument("--memory-train-size", type=int, default=LLMFeatureConfig.memory_train_size)
    parser.add_argument("--test-size", type=int, default=LLMFeatureConfig.test_size)
    parser.add_argument("--num-experts", type=int, default=LLMFeatureConfig.num_experts)
    parser.add_argument("--top-k-experts", type=int, default=LLMFeatureConfig.top_k_experts)
    parser.add_argument("--batch-size", type=int, default=LLMFeatureConfig.batch_size)
    parser.add_argument("--feature-batch-size", type=int, default=LLMFeatureConfig.feature_batch_size)
    parser.add_argument("--base-epochs", type=int, default=LLMFeatureConfig.base_epochs)
    parser.add_argument("--memory-epochs", type=int, default=LLMFeatureConfig.memory_epochs)
    parser.add_argument("--learning-rate", type=float, default=LLMFeatureConfig.learning_rate)
    parser.add_argument("--router-loss-weight", type=float, default=LLMFeatureConfig.router_loss_weight)
    parser.add_argument("--sparsity-weight", type=float, default=LLMFeatureConfig.sparsity_weight)
    parser.add_argument("--load-balance-weight", type=float, default=LLMFeatureConfig.load_balance_weight)
    parser.add_argument("--router-label-noise", type=float, default=LLMFeatureConfig.router_label_noise)
    parser.add_argument("--max-length", type=int, default=LLMFeatureConfig.max_length)
    parser.add_argument("--pooling", choices=["mean", "last"], default=LLMFeatureConfig.pooling)
    parser.add_argument("--thresholds", default=LLMFeatureConfig.thresholds)
    parser.add_argument("--device", default=LLMFeatureConfig.device)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(list(argv))
    return LLMFeatureConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        seed=args.seed,
        num_facts=args.num_facts,
        common_fact_ratio=args.common_fact_ratio,
        base_train_size=args.base_train_size,
        memory_train_size=args.memory_train_size,
        test_size=args.test_size,
        num_experts=args.num_experts,
        top_k_experts=args.top_k_experts,
        batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        base_epochs=args.base_epochs,
        memory_epochs=args.memory_epochs,
        learning_rate=args.learning_rate,
        router_loss_weight=args.router_loss_weight,
        sparsity_weight=args.sparsity_weight,
        load_balance_weight=args.load_balance_weight,
        router_label_noise=args.router_label_noise,
        max_length=args.max_length,
        pooling=args.pooling,
        thresholds=args.thresholds,
        device=args.device,
        fp16=args.fp16,
        trust_remote_code=args.trust_remote_code,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    run(cfg)


if __name__ == "__main__":
    main()

