from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from edge_memory.data import SAMPLE_COMMON_FACT, SAMPLE_GENERAL, SAMPLE_TAIL_FACT
from edge_memory.llm_features import (
    FeatureBaseClassifier,
    FeatureConditionalMemoryModel,
    build_loader,
    evaluate_base,
    evaluate_memory,
    extract_features,
    load_transformers,
    train_base,
    train_memory,
)
from edge_memory.threshold_sweep import parse_thresholds, save_csv
from edge_memory.train import get_device, print_metrics, set_seed


GENERAL_TASKS = [
    ("reply yes", "general_yes"),
    ("reply no", "general_no"),
    ("sentiment good", "general_positive"),
    ("sentiment bad", "general_negative"),
    ("parity even", "general_even"),
    ("parity odd", "general_odd"),
]


@dataclass(frozen=True)
class RealQAConfig:
    model_name_or_path: str
    output_dir: str = "outputs/real_qa"
    seed: int = 42
    dataset_name: str = "squad"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    local_jsonl: Optional[str] = None
    question_field: str = "question"
    answer_field: str = "answers"
    num_facts: int = 120
    common_fact_ratio: float = 0.6
    base_train_size: int = 4000
    memory_train_size: int = 6000
    test_size: int = 1200
    num_experts: int = 8
    top_k_experts: int = 2
    batch_size: int = 64
    feature_batch_size: int = 16
    base_epochs: int = 20
    memory_epochs: int = 30
    learning_rate: float = 3e-3
    router_loss_weight: float = 0.6
    sparsity_weight: float = 0.03
    load_balance_weight: float = 0.01
    router_label_noise: float = 0.0
    max_length: int = 96
    pooling: str = "mean"
    thresholds: str = "0.30,0.50,0.70"
    device: str = "auto"
    fp16: bool = False
    trust_remote_code: bool = False
    max_source_examples: int = 50000


@dataclass(frozen=True)
class QAFact:
    fact_id: int
    question: str
    answer: str
    is_common: bool


class RawTextDataset:
    def __init__(self, samples: List[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class RealQAWorld:
    def __init__(self, facts: List[QAFact]) -> None:
        self.facts = facts
        self.common_facts = [fact for fact in facts if fact.is_common]
        self.tail_facts = [fact for fact in facts if not fact.is_common]
        self.labels = [fact.answer for fact in facts] + [label for _, label in GENERAL_TASKS]
        self.num_fact_labels = len(facts)
        self.num_labels = len(self.labels)

    def render_fact_prompt(self, fact: QAFact, rng: random.Random) -> str:
        templates = [
            "{question}",
            "Question: {question}",
            "Answer this question: {question}",
            "Please answer the question: {question}",
            "Knowledge question: {question}",
        ]
        return rng.choice(templates).format(question=fact.question.strip())

    def render_general_prompt(self, rng: random.Random) -> Tuple[str, int]:
        text, _ = rng.choice(GENERAL_TASKS)
        variants = [
            text,
            f"please {text}",
            f"task {text}",
            f"{text} now",
        ]
        label = self.num_fact_labels + [item[0] for item in GENERAL_TASKS].index(text)
        return rng.choice(variants), label


def normalize_answer(answer: str) -> str:
    return " ".join(str(answer).strip().split())


def extract_answer(raw_answer) -> Optional[str]:
    if raw_answer is None:
        return None
    if isinstance(raw_answer, str):
        return normalize_answer(raw_answer)
    if isinstance(raw_answer, dict):
        texts = raw_answer.get("text") or raw_answer.get("texts") or raw_answer.get("answer")
        return extract_answer(texts)
    if isinstance(raw_answer, (list, tuple)):
        for item in raw_answer:
            answer = extract_answer(item)
            if answer:
                return answer
    return normalize_answer(str(raw_answer))


def load_local_jsonl(path: Path, question_field: str, answer_field: str, limit: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = normalize_answer(obj.get(question_field, ""))
            answer = extract_answer(obj.get(answer_field))
            if question and answer:
                pairs.append((question, answer))
    return pairs


def load_hf_dataset(cfg: RealQAConfig) -> List[Tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HuggingFace QA datasets. Install it with: pip install datasets"
        ) from exc

    if cfg.dataset_config:
        dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    else:
        dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    pairs: List[Tuple[str, str]] = []
    for obj in dataset:
        if len(pairs) >= cfg.max_source_examples:
            break
        question = normalize_answer(obj.get(cfg.question_field, ""))
        answer = extract_answer(obj.get(cfg.answer_field))
        if question and answer:
            pairs.append((question, answer))
    return pairs


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out: List[Tuple[str, str]] = []
    for question, answer in pairs:
        key = (question.lower(), answer.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((question, answer))
    return out


def build_world(cfg: RealQAConfig) -> RealQAWorld:
    if cfg.local_jsonl:
        pairs = load_local_jsonl(Path(cfg.local_jsonl), cfg.question_field, cfg.answer_field, cfg.max_source_examples)
    else:
        pairs = load_hf_dataset(cfg)

    pairs = deduplicate_pairs(pairs)
    if len(pairs) < cfg.num_facts:
        raise ValueError(f"Need at least {cfg.num_facts} QA pairs, got {len(pairs)}")

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    selected = pairs[: cfg.num_facts]
    common_cutoff = int(cfg.num_facts * cfg.common_fact_ratio)
    facts = [
        QAFact(fact_id=idx, question=question, answer=answer, is_common=idx < common_cutoff)
        for idx, (question, answer) in enumerate(selected)
    ]
    return RealQAWorld(facts)


def build_samples(world: RealQAWorld, split: str, size: int, seed: int) -> RawTextDataset:
    rng = random.Random(seed)
    samples: List[dict] = []
    for _ in range(size):
        p = rng.random()
        if split == "base_train":
            kind = SAMPLE_GENERAL if p < 0.35 else SAMPLE_COMMON_FACT
        elif split == "memory_train":
            if p < 0.45:
                kind = SAMPLE_TAIL_FACT
            elif p < 0.70:
                kind = SAMPLE_COMMON_FACT
            else:
                kind = SAMPLE_GENERAL
        elif split == "test":
            if p < 0.34:
                kind = SAMPLE_TAIL_FACT
            elif p < 0.67:
                kind = SAMPLE_COMMON_FACT
            else:
                kind = SAMPLE_GENERAL
        else:
            raise ValueError(f"Unknown split: {split}")

        if kind == SAMPLE_TAIL_FACT:
            fact = rng.choice(world.tail_facts)
            text = world.render_fact_prompt(fact, rng)
            label = fact.fact_id
            read_label = 1.0
        elif kind == SAMPLE_COMMON_FACT:
            fact = rng.choice(world.common_facts)
            text = world.render_fact_prompt(fact, rng)
            label = fact.fact_id
            read_label = 0.0
        else:
            text, label = world.render_general_prompt(rng)
            read_label = 0.0

        samples.append(
            {
                "text": text,
                "label": torch.tensor(label, dtype=torch.long),
                "read_label": torch.tensor(read_label, dtype=torch.float32),
                "sample_type": torch.tensor(kind, dtype=torch.long),
            }
        )
    return RawTextDataset(samples)


def run(cfg: RealQAConfig) -> Dict[str, object]:
    AutoModelForCausalLM, AutoTokenizer = load_transformers()
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device)
    dtype = torch.float16 if cfg.fp16 and device.type == "cuda" else torch.float32

    print(f"Using device: {device}", flush=True)
    print(f"Loading real QA data: {cfg.local_jsonl or cfg.dataset_name}", flush=True)
    world = build_world(cfg)
    print(
        f"Loaded facts: total={len(world.facts)}, common={len(world.common_facts)}, tail={len(world.tail_facts)}",
        flush=True,
    )

    print(f"Loading LLM: {cfg.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    ).to(device)
    llm.eval()

    raw_base = build_samples(world, "base_train", cfg.base_train_size, cfg.seed + 1)
    raw_memory = build_samples(world, "memory_train", cfg.memory_train_size, cfg.seed + 2)
    raw_test = build_samples(world, "test", cfg.test_size, cfg.seed + 3)

    print("Extracting LLM features from real QA prompts...", flush=True)
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

    base = FeatureBaseClassifier(hidden_size, world.num_labels).to(device)
    train_base(base, base_loader, cfg, device)
    base_metrics = evaluate_base(base, test_loader, device)
    print_metrics("real_qa_base", base_metrics)

    memory_model = FeatureConditionalMemoryModel(
        base=base,
        hidden_size=hidden_size,
        num_labels=world.num_labels,
        num_experts=cfg.num_experts,
        top_k_experts=cfg.top_k_experts,
    ).to(device)
    train_memory(memory_model, memory_loader, cfg, device)
    dense_metrics = evaluate_memory(memory_model, test_loader, device, 0.5, mode="dense")
    print_metrics("real_qa_dense_memory", dense_metrics)

    thresholds = parse_thresholds(cfg.thresholds)
    rows: List[Dict[str, float]] = []
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        metrics = evaluate_memory(memory_model, test_loader, device, threshold, mode="conditional")
        row = {"threshold": threshold}
        row.update(metrics)
        rows.append(row)
        threshold_metrics[f"{threshold:.4f}"] = metrics
        print_metrics(f"real_qa_conditional_threshold_{threshold:.2f}", metrics)

    fact_preview = [
        {"fact_id": fact.fact_id, "question": fact.question, "answer": fact.answer, "is_common": fact.is_common}
        for fact in world.facts[: min(20, len(world.facts))]
    ]
    payload = {
        "config": asdict(cfg),
        "hidden_size": hidden_size,
        "num_labels": world.num_labels,
        "num_facts": len(world.facts),
        "num_common_facts": len(world.common_facts),
        "num_tail_facts": len(world.tail_facts),
        "base_metrics": base_metrics,
        "dense_metrics": dense_metrics,
        "threshold_metrics": threshold_metrics,
        "fact_preview": fact_preview,
    }
    with (output_dir / "real_qa_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    save_csv(output_dir / "real_qa_threshold_sweep.csv", rows)
    torch.save(base.state_dict(), output_dir / "real_qa_base.pt")
    torch.save(memory_model.state_dict(), output_dir / "real_qa_conditional_memory.pt")
    return payload


def parse_args(argv: Iterable[str]) -> RealQAConfig:
    parser = argparse.ArgumentParser(description="Real QA data validation for conditional parametric memory.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", default=RealQAConfig.output_dir)
    parser.add_argument("--seed", type=int, default=RealQAConfig.seed)
    parser.add_argument("--dataset-name", default=RealQAConfig.dataset_name)
    parser.add_argument("--dataset-config", default=RealQAConfig.dataset_config)
    parser.add_argument("--dataset-split", default=RealQAConfig.dataset_split)
    parser.add_argument("--local-jsonl", default=RealQAConfig.local_jsonl)
    parser.add_argument("--question-field", default=RealQAConfig.question_field)
    parser.add_argument("--answer-field", default=RealQAConfig.answer_field)
    parser.add_argument("--num-facts", type=int, default=RealQAConfig.num_facts)
    parser.add_argument("--common-fact-ratio", type=float, default=RealQAConfig.common_fact_ratio)
    parser.add_argument("--base-train-size", type=int, default=RealQAConfig.base_train_size)
    parser.add_argument("--memory-train-size", type=int, default=RealQAConfig.memory_train_size)
    parser.add_argument("--test-size", type=int, default=RealQAConfig.test_size)
    parser.add_argument("--num-experts", type=int, default=RealQAConfig.num_experts)
    parser.add_argument("--top-k-experts", type=int, default=RealQAConfig.top_k_experts)
    parser.add_argument("--batch-size", type=int, default=RealQAConfig.batch_size)
    parser.add_argument("--feature-batch-size", type=int, default=RealQAConfig.feature_batch_size)
    parser.add_argument("--base-epochs", type=int, default=RealQAConfig.base_epochs)
    parser.add_argument("--memory-epochs", type=int, default=RealQAConfig.memory_epochs)
    parser.add_argument("--learning-rate", type=float, default=RealQAConfig.learning_rate)
    parser.add_argument("--router-loss-weight", type=float, default=RealQAConfig.router_loss_weight)
    parser.add_argument("--sparsity-weight", type=float, default=RealQAConfig.sparsity_weight)
    parser.add_argument("--load-balance-weight", type=float, default=RealQAConfig.load_balance_weight)
    parser.add_argument("--router-label-noise", type=float, default=RealQAConfig.router_label_noise)
    parser.add_argument("--max-length", type=int, default=RealQAConfig.max_length)
    parser.add_argument("--pooling", choices=["mean", "last"], default=RealQAConfig.pooling)
    parser.add_argument("--thresholds", default=RealQAConfig.thresholds)
    parser.add_argument("--device", default=RealQAConfig.device)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-source-examples", type=int, default=RealQAConfig.max_source_examples)
    args = parser.parse_args(list(argv))
    return RealQAConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        seed=args.seed,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        local_jsonl=args.local_jsonl,
        question_field=args.question_field,
        answer_field=args.answer_field,
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
        max_source_examples=args.max_source_examples,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    run(cfg)


if __name__ == "__main__":
    main()

