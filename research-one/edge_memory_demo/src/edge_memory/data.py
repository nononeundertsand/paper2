from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

SAMPLE_COMMON_FACT = 0
SAMPLE_TAIL_FACT = 1
SAMPLE_GENERAL = 2

SAMPLE_TYPE_NAMES = {
    SAMPLE_COMMON_FACT: "common_fact",
    SAMPLE_TAIL_FACT: "tail_fact",
    SAMPLE_GENERAL: "general",
}


@dataclass(frozen=True)
class SyntheticConfig:
    num_facts: int = 240
    common_fact_ratio: float = 0.65
    num_general_labels: int = 6
    seed: int = 42


@dataclass(frozen=True)
class Fact:
    fact_id: int
    domain: str
    entity: str
    answer: str
    is_common: bool


class Vocab:
    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.add(PAD_TOKEN)
        self.add(UNK_TOKEN)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def update(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.add(token)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unk) for token in tokens]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)


class SyntheticWorld:
    def __init__(self, cfg: SyntheticConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.vocab = Vocab()
        self.facts = self._build_facts()
        self.general_tasks = self._build_general_tasks()
        self.labels = [fact.answer for fact in self.facts]
        self.labels.extend(label for _, label in self.general_tasks)
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self._build_vocab()

    @property
    def common_facts(self) -> List[Fact]:
        return [fact for fact in self.facts if fact.is_common]

    @property
    def tail_facts(self) -> List[Fact]:
        return [fact for fact in self.facts if not fact.is_common]

    def _build_facts(self) -> List[Fact]:
        domains = ["capital", "currency", "founder", "river", "language", "landmark"]
        common_cutoff = int(self.cfg.num_facts * self.cfg.common_fact_ratio)
        facts: List[Fact] = []
        for idx in range(self.cfg.num_facts):
            domain = domains[idx % len(domains)]
            entity = f"entity_{idx:04d}"
            answer = f"answer_{idx:04d}"
            facts.append(
                Fact(
                    fact_id=idx,
                    domain=domain,
                    entity=entity,
                    answer=answer,
                    is_common=idx < common_cutoff,
                )
            )
        return facts

    def _build_general_tasks(self) -> List[tuple[List[str], str]]:
        base_tasks = [
            (["reply", "yes"], "general_yes"),
            (["reply", "no"], "general_no"),
            (["sentiment", "good"], "general_positive"),
            (["sentiment", "bad"], "general_negative"),
            (["parity", "even"], "general_even"),
            (["parity", "odd"], "general_odd"),
        ]
        return base_tasks[: self.cfg.num_general_labels]

    def _build_vocab(self) -> None:
        templates = [
            ["what", "is", "the", "{domain}", "of", "{entity}", "?"],
            ["tell", "me", "the", "{domain}", "for", "{entity}", "?"],
            ["need", "{domain}", "about", "{entity}"],
            ["{entity}", "{domain}", "is", "what", "?"],
        ]
        fixed_tokens = set()
        for template in templates:
            fixed_tokens.update(token for token in template if not token.startswith("{"))
        self.vocab.update(fixed_tokens)
        for fact in self.facts:
            self.vocab.update([fact.domain, fact.entity, fact.answer])
        for prompt, label in self.general_tasks:
            self.vocab.update(prompt)
            self.vocab.add(label)

    def render_fact_prompt(self, fact: Fact, rng: random.Random) -> List[str]:
        templates = [
            ["what", "is", "the", fact.domain, "of", fact.entity, "?"],
            ["tell", "me", "the", fact.domain, "for", fact.entity, "?"],
            ["need", fact.domain, "about", fact.entity],
            [fact.entity, fact.domain, "is", "what", "?"],
        ]
        return list(rng.choice(templates))

    def render_general_prompt(self, rng: random.Random) -> tuple[List[str], str]:
        prompt, label = rng.choice(self.general_tasks)
        variants = [
            list(prompt),
            ["please", *prompt],
            [*prompt, "now"],
            ["task", *prompt],
        ]
        for token in ["please", "now", "task"]:
            self.vocab.add(token)
        return list(rng.choice(variants)), label


class SyntheticMemoryDataset(Dataset):
    def __init__(
        self,
        world: SyntheticWorld,
        split: str,
        size: int,
        seed: int,
    ) -> None:
        self.world = world
        self.split = split
        self.size = size
        self.rng = random.Random(seed)
        self.samples = [self._sample_one() for _ in range(size)]

    def _sample_kind(self) -> int:
        p = self.rng.random()
        if self.split == "base_train":
            return SAMPLE_GENERAL if p < 0.35 else SAMPLE_COMMON_FACT
        if self.split == "memory_train":
            if p < 0.45:
                return SAMPLE_TAIL_FACT
            if p < 0.70:
                return SAMPLE_COMMON_FACT
            return SAMPLE_GENERAL
        if self.split == "test":
            if p < 0.34:
                return SAMPLE_TAIL_FACT
            if p < 0.67:
                return SAMPLE_COMMON_FACT
            return SAMPLE_GENERAL
        raise ValueError(f"Unknown split: {self.split}")

    def _sample_one(self) -> dict:
        kind = self._sample_kind()
        if kind == SAMPLE_TAIL_FACT:
            fact = self.rng.choice(self.world.tail_facts)
            tokens = self.world.render_fact_prompt(fact, self.rng)
            label = self.world.label_to_id[fact.answer]
            read_label = 1.0
        elif kind == SAMPLE_COMMON_FACT:
            fact = self.rng.choice(self.world.common_facts)
            tokens = self.world.render_fact_prompt(fact, self.rng)
            label = self.world.label_to_id[fact.answer]
            read_label = 0.0
        else:
            tokens, general_label = self.world.render_general_prompt(self.rng)
            label = self.world.label_to_id[general_label]
            read_label = 0.0

        return {
            "input_ids": torch.tensor(self.world.vocab.encode(tokens), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "read_label": torch.tensor(read_label, dtype=torch.float32),
            "sample_type": torch.tensor(kind, dtype=torch.long),
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def collate_batch(samples: List[dict], pad_id: int) -> dict:
    max_len = max(sample["input_ids"].numel() for sample in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for row, sample in enumerate(samples):
        seq = sample["input_ids"]
        input_ids[row, : seq.numel()] = seq
        attention_mask[row, : seq.numel()] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack([sample["label"] for sample in samples]),
        "read_labels": torch.stack([sample["read_label"] for sample in samples]),
        "sample_types": torch.stack([sample["sample_type"] for sample in samples]),
    }

