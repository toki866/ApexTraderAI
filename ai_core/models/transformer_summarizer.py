# -*- coding: utf-8 -*-
'''
transformer_summarizer.py

StepD' (StepD-prime) transformer summarizer used to compress fixed-length sequences into embeddings.

Design notes:
- Input is a padded sequence (B, T, F) with an attention mask (B, T) where 1=valid token, 0=padding.
- Produces an embedding vector (B, Demb) used downstream (e.g., StepE RL observation).
- Also provides a small supervised head for training (binary up/down by default).
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TransformerSummarizerConfig:
    feature_dim: int
    max_len: int = 20
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    embedding_dim: int = 32
    num_classes: int = 2  # binary by default
    use_cls_token: bool = False  # masked-mean pooling by default


class TransformerSummarizer(nn.Module):
    def __init__(self, cfg: TransformerSummarizerConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        if cfg.use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.normal_(self.cls, mean=0.0, std=0.02)
            max_len_eff = cfg.max_len + 1
        else:
            self.cls = None
            max_len_eff = cfg.max_len

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.emb_proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.embedding_dim),
        )
        self.head = nn.Linear(cfg.embedding_dim, cfg.num_classes)

        self.register_buffer(
            "_pos_idx",
            torch.arange(max_len_eff, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        return_embedding: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Args:
            x: (B, T, F)
            attn_mask: (B, T) with 1=valid, 0=pad
            return_embedding: if True returns (logits, embedding); else (logits, None)
        '''
        if x.dim() != 3:
            raise ValueError(f"x must be (B,T,F). got {tuple(x.shape)}")
        if attn_mask.dim() != 2:
            raise ValueError(f"attn_mask must be (B,T). got {tuple(attn_mask.shape)}")

        B, T, _ = x.shape
        if T != self.cfg.max_len:
            raise ValueError(f"Expected T==max_len({self.cfg.max_len}) but got T={T}")

        h = self.in_proj(x)  # (B,T,D)
        pos = self.pos_emb(self._pos_idx[:T]).unsqueeze(0)  # (1,T,D)
        h = h + pos

        key_padding_mask = (attn_mask == 0)  # (B,T) True=PAD

        if self.cfg.use_cls_token:
            cls = self.cls.expand(B, -1, -1)  # (B,1,D)
            h = torch.cat([cls, h], dim=1)  # (B,T+1,D)

            cls_valid = torch.ones((B, 1), device=attn_mask.device, dtype=attn_mask.dtype)
            attn_mask_eff = torch.cat([cls_valid, attn_mask], dim=1)  # (B,T+1)
            key_padding_mask = (attn_mask_eff == 0)
        else:
            attn_mask_eff = attn_mask

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,Teff,D)

        if self.cfg.use_cls_token:
            pooled = h[:, 0, :]
        else:
            m = attn_mask_eff.unsqueeze(-1).to(h.dtype)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (h * m).sum(dim=1) / denom

        emb = self.emb_proj(pooled)
        logits = self.head(emb)
        return (logits, emb) if return_embedding else (logits, None)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, emb = self.forward(x, attn_mask, return_embedding=True)
        assert emb is not None
        return emb
