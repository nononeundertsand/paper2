from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_labels: int
    hidden_size: int = 128
    num_experts: int = 8
    top_k_experts: int = 2
    dropout: float = 0.1


class BagOfWordsEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, pad_id: int, dropout: float) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class BaseClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int) -> None:
        super().__init__()
        self.encoder = BagOfWordsEncoder(cfg.vocab_size, cfg.hidden_size, pad_id, cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.classifier(hidden)
        return {"hidden": hidden, "logits": logits}


class MemoryExpert(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class ConditionalMemoryModel(nn.Module):
    def __init__(self, base: BaseClassifier, cfg: ModelConfig) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        for param in self.base.parameters():
            param.requires_grad = False

        self.read_router = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size + 2),
            nn.Linear(cfg.hidden_size + 2, cfg.hidden_size // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_size // 2, 1),
        )
        self.expert_router = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.num_experts),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.experts = nn.ModuleList(
            [MemoryExpert(cfg.hidden_size, cfg.num_labels, cfg.dropout) for _ in range(cfg.num_experts)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        force_read: Optional[bool] = None,
        threshold: float = 0.5,
        soft_read: bool = True,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_out = self.base(input_ids, attention_mask)
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
        k = min(self.cfg.top_k_experts, self.cfg.num_experts)
        top_values, top_indices = torch.topk(expert_scores, k=k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        expert_logits = []
        for expert in self.experts:
            expert_logits.append(expert(hidden))
        stacked_logits = torch.stack(expert_logits, dim=1)

        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.cfg.num_labels)
        selected_logits = torch.gather(stacked_logits, dim=1, index=gather_index)
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
            "mixed_logits": mixed_logits,
            "final_logits": final_logits,
            "read_logit": read_logit,
            "read_prob": read_prob,
            "read_mask": read_mask,
            "expert_scores": expert_scores,
            "top_indices": top_indices,
            "alpha": alpha.squeeze(-1),
        }

