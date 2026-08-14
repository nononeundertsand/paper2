from __future__ import annotations

import argparse
import json
import random
import re
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
from edge_memory.train import batch_to_device, get_device, print_metrics, set_seed


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
    general_source: str = "hf"
    general_dataset_name: str = "ag_news"
    general_dataset_config: Optional[str] = None
    general_dataset_split: str = "train"
    general_local_jsonl: Optional[str] = None
    general_text_field: str = "text"
    general_label_field: str = "label"
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


@dataclass(frozen=True)
class GeneralExample:
    text: str
    label_name: str


class RawTextDataset:
    def __init__(self, samples: List[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class RealQAWorld:
    def __init__(self, facts: List[QAFact], general_examples: List[GeneralExample]) -> None:
        self.facts = facts
        self.common_facts = [fact for fact in facts if fact.is_common]
        self.tail_facts = [fact for fact in facts if not fact.is_common]
        self.general_examples = general_examples
        self.general_label_names = sorted({example.label_name for example in general_examples})
        self.general_label_to_offset = {
            label_name: offset for offset, label_name in enumerate(self.general_label_names)
        }
        self.labels = [fact.answer for fact in facts] + self.general_label_names
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
        example = rng.choice(self.general_examples)
        label = self.num_fact_labels + self.general_label_to_offset[example.label_name]
        return example.text, label

    @property
    def has_general_examples(self) -> bool:
        return bool(self.general_examples)


def normalize_answer(answer: str) -> str:
    return " ".join(str(answer).strip().split())


def normalize_for_eval(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_for_eval(prediction).split()
    gold_tokens = normalize_for_eval(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        count = common.get(token, 0)
        if count > 0:
            overlap += 1
            common[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_for_eval(prediction) == normalize_for_eval(gold))


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


def load_hf_general_examples(cfg: RealQAConfig) -> List[GeneralExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HuggingFace general datasets. Install it with: pip install datasets"
        ) from exc

    if cfg.general_dataset_config:
        dataset = load_dataset(cfg.general_dataset_name, cfg.general_dataset_config, split=cfg.general_dataset_split)
    else:
        dataset = load_dataset(cfg.general_dataset_name, split=cfg.general_dataset_split)

    examples: List[GeneralExample] = []
    for obj in dataset:
        if len(examples) >= cfg.max_source_examples:
            break
        text = normalize_answer(obj.get(cfg.general_text_field, ""))
        raw_label = obj.get(cfg.general_label_field)
        if text == "" or raw_label is None:
            continue
        label_name = f"general_{cfg.general_dataset_name}_{raw_label}"
        examples.append(GeneralExample(text=text, label_name=label_name))
    return examples


def load_local_general_jsonl(
    path: Path,
    text_field: str,
    label_field: str,
    limit: int,
) -> List[GeneralExample]:
    examples: List[GeneralExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(examples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = normalize_answer(obj.get(text_field, ""))
            raw_label = obj.get(label_field)
            if text == "" or raw_label is None:
                continue
            examples.append(GeneralExample(text=text, label_name=f"general_local_{raw_label}"))
    return examples


def build_synthetic_general_examples() -> List[GeneralExample]:
    examples: List[GeneralExample] = []
    for text, label in GENERAL_TASKS:
        variants = [
            text,
            f"please {text}",
            f"task {text}",
            f"{text} now",
        ]
        for variant in variants:
            examples.append(GeneralExample(text=variant, label_name=label))
    return examples


def build_general_examples(cfg: RealQAConfig) -> List[GeneralExample]:
    if cfg.general_source == "none":
        return []
    if cfg.general_source == "synthetic":
        return build_synthetic_general_examples()
    if cfg.general_source == "hf":
        return load_hf_general_examples(cfg)
    if cfg.general_source == "jsonl":
        if not cfg.general_local_jsonl:
            raise ValueError("--general-local-jsonl is required when --general-source jsonl")
        return load_local_general_jsonl(
            Path(cfg.general_local_jsonl),
            cfg.general_text_field,
            cfg.general_label_field,
            cfg.max_source_examples,
        )
    raise ValueError(f"Unknown general_source: {cfg.general_source}")


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
    general_examples = build_general_examples(cfg)
    if cfg.general_source != "none" and not general_examples:
        raise ValueError(f"No general examples loaded from source: {cfg.general_source}")
    return RealQAWorld(facts, general_examples)


def sample_kind(split: str, rng: random.Random, has_general: bool) -> int:
    p = rng.random()
    if split == "base_train":
        if has_general and p < 0.35:
            return SAMPLE_GENERAL
        return SAMPLE_COMMON_FACT
    if split == "memory_train":
        if p < 0.45:
            return SAMPLE_TAIL_FACT
        if p < 0.70 or not has_general:
            return SAMPLE_COMMON_FACT
        return SAMPLE_GENERAL
    if split == "test":
        if p < 0.34:
            return SAMPLE_TAIL_FACT
        if p < 0.67 or not has_general:
            return SAMPLE_COMMON_FACT
        return SAMPLE_GENERAL
    raise ValueError(f"Unknown split: {split}")


def build_samples(world: RealQAWorld, split: str, size: int, seed: int) -> RawTextDataset:
    rng = random.Random(seed)
    samples: List[dict] = []
    for _ in range(size):
        kind = sample_kind(split, rng, world.has_general_examples)

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


@torch.no_grad()
def evaluate_candidate_answer_ranking(
    loader,
    device: torch.device,
    world: RealQAWorld,
    predict_logits,
) -> Dict[str, float]:
    answer_texts = world.labels[: world.num_fact_labels]
    totals = {
        "qa": {"n": 0, "em": 0.0, "f1": 0.0, "mrr": 0.0, "hits1": 0.0, "hits5": 0.0},
        "common": {"n": 0, "em": 0.0, "f1": 0.0},
        "tail": {"n": 0, "em": 0.0, "f1": 0.0},
    }

    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = predict_logits(batch)[:, : world.num_fact_labels]
        labels = batch["labels"]
        sample_types = batch["sample_types"]
        qa_mask = (sample_types == SAMPLE_COMMON_FACT) | (sample_types == SAMPLE_TAIL_FACT)
        if qa_mask.sum().item() == 0:
            continue

        qa_logits = logits[qa_mask]
        qa_labels = labels[qa_mask]
        qa_types = sample_types[qa_mask]
        ranked = torch.argsort(qa_logits, dim=-1, descending=True)
        top1 = ranked[:, 0]

        for row in range(qa_labels.numel()):
            pred_id = int(top1[row].item())
            gold_id = int(qa_labels[row].item())
            pred_answer = answer_texts[pred_id]
            gold_answer = answer_texts[gold_id]
            em = exact_match(pred_answer, gold_answer)
            f1 = token_f1(pred_answer, gold_answer)

            matches = torch.nonzero(ranked[row] == gold_id, as_tuple=False)
            rank = int(matches[0].item()) + 1 if matches.numel() > 0 else world.num_fact_labels + 1
            totals["qa"]["n"] += 1
            totals["qa"]["em"] += em
            totals["qa"]["f1"] += f1
            totals["qa"]["mrr"] += 1.0 / rank
            totals["qa"]["hits1"] += float(rank <= 1)
            totals["qa"]["hits5"] += float(rank <= 5)

            bucket = "tail" if int(qa_types[row].item()) == SAMPLE_TAIL_FACT else "common"
            totals[bucket]["n"] += 1
            totals[bucket]["em"] += em
            totals[bucket]["f1"] += f1

    qa_n = max(totals["qa"]["n"], 1)
    common_n = max(totals["common"]["n"], 1)
    tail_n = max(totals["tail"]["n"], 1)
    return {
        "rank_qa_em": totals["qa"]["em"] / qa_n,
        "rank_qa_f1": totals["qa"]["f1"] / qa_n,
        "rank_qa_mrr": totals["qa"]["mrr"] / qa_n,
        "rank_qa_hits1": totals["qa"]["hits1"] / qa_n,
        "rank_qa_hits5": totals["qa"]["hits5"] / qa_n,
        "rank_common_em": totals["common"]["em"] / common_n,
        "rank_common_f1": totals["common"]["f1"] / common_n,
        "rank_tail_em": totals["tail"]["em"] / tail_n,
        "rank_tail_f1": totals["tail"]["f1"] / tail_n,
    }


def add_ranking_metrics(
    metrics: Dict[str, float],
    loader,
    device: torch.device,
    world: RealQAWorld,
    predict_logits,
) -> Dict[str, float]:
    metrics.update(evaluate_candidate_answer_ranking(loader, device, world, predict_logits))
    return metrics


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
    print(
        f"Loaded general examples: total={len(world.general_examples)}, labels={len(world.general_label_names)}, "
        f"source={cfg.general_source}",
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
    add_ranking_metrics(
        base_metrics,
        test_loader,
        device,
        world,
        lambda batch: base(batch["features"])["logits"],
    )
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
    add_ranking_metrics(
        dense_metrics,
        test_loader,
        device,
        world,
        lambda batch: memory_model(batch["features"], threshold=0.5, force_read=True, soft_read=False)["final_logits"],
    )
    print_metrics("real_qa_dense_memory", dense_metrics)

    thresholds = parse_thresholds(cfg.thresholds)
    rows: List[Dict[str, float]] = []
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        metrics = evaluate_memory(memory_model, test_loader, device, threshold, mode="conditional")
        add_ranking_metrics(
            metrics,
            test_loader,
            device,
            world,
            lambda batch, threshold=threshold: memory_model(
                batch["features"],
                threshold=threshold,
                force_read=None,
                soft_read=False,
            )["final_logits"],
        )
        row = {"threshold": threshold}
        row.update(metrics)
        rows.append(row)
        threshold_metrics[f"{threshold:.4f}"] = metrics
        print_metrics(f"real_qa_conditional_threshold_{threshold:.2f}", metrics)

    fact_preview = [
        {"fact_id": fact.fact_id, "question": fact.question, "answer": fact.answer, "is_common": fact.is_common}
        for fact in world.facts[: min(20, len(world.facts))]
    ]
    general_preview = [
        {"text": example.text, "label_name": example.label_name}
        for example in world.general_examples[: min(20, len(world.general_examples))]
    ]
    payload = {
        "config": asdict(cfg),
        "hidden_size": hidden_size,
        "num_labels": world.num_labels,
        "num_facts": len(world.facts),
        "num_common_facts": len(world.common_facts),
        "num_tail_facts": len(world.tail_facts),
        "num_general_examples": len(world.general_examples),
        "num_general_labels": len(world.general_label_names),
        "base_metrics": base_metrics,
        "dense_metrics": dense_metrics,
        "threshold_metrics": threshold_metrics,
        "fact_preview": fact_preview,
        "general_preview": general_preview,
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
    parser.add_argument("--general-source", choices=["hf", "jsonl", "synthetic", "none"], default=RealQAConfig.general_source)
    parser.add_argument("--general-dataset-name", default=RealQAConfig.general_dataset_name)
    parser.add_argument("--general-dataset-config", default=RealQAConfig.general_dataset_config)
    parser.add_argument("--general-dataset-split", default=RealQAConfig.general_dataset_split)
    parser.add_argument("--general-local-jsonl", default=RealQAConfig.general_local_jsonl)
    parser.add_argument("--general-text-field", default=RealQAConfig.general_text_field)
    parser.add_argument("--general-label-field", default=RealQAConfig.general_label_field)
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
        general_source=args.general_source,
        general_dataset_name=args.general_dataset_name,
        general_dataset_config=args.general_dataset_config,
        general_dataset_split=args.general_dataset_split,
        general_local_jsonl=args.general_local_jsonl,
        general_text_field=args.general_text_field,
        general_label_field=args.general_label_field,
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
