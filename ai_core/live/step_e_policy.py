from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ai_core.services.step_e_service import DiffPolicyNet


@dataclass
class StepEPolicy:
    model_path: Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"StepE model not found: {self.model_path}")

        map_location = "cpu"
        ckpt = torch.load(self.model_path, map_location=map_location)
        self.cfg: Dict[str, object] = dict(ckpt.get("cfg", {}) or {})
        self.obs_cols: List[str] = list(ckpt.get("obs_cols", []) or [])
        if not self.obs_cols:
            raise ValueError(f"{self.model_path} has empty obs_cols")

        self.mu = np.asarray(ckpt.get("mu"), dtype=np.float32)
        self.sd = np.asarray(ckpt.get("sd"), dtype=np.float32)
        if self.mu.shape[0] != len(self.obs_cols) or self.sd.shape[0] != len(self.obs_cols):
            raise ValueError(
                f"{self.model_path} scaler dim mismatch: obs={len(self.obs_cols)} mu={self.mu.shape} sd={self.sd.shape}"
            )
        self.sd = np.where(np.abs(self.sd) < 1e-8, 1.0, self.sd).astype(np.float32)

        requested_device = str(self.device or "cpu").strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        hidden_dim = int(self.cfg.get("hidden_dim", 64) or 64)
        self.net = DiffPolicyNet(in_dim=len(self.obs_cols) + 1, hidden_dim=hidden_dim)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.net.to(self.device)

        self.pos_limit = float(self.cfg.get("pos_limit", 1.0) or 1.0)

    def predict(self, obs_dict: Dict[str, float], pos_prev: float = 0.0, pos_limit: Optional[float] = None) -> float:
        vec = np.array([float(obs_dict.get(c, 0.0) or 0.0) for c in self.obs_cols], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        x = (vec - self.mu) / self.sd
        x = np.concatenate([x, np.array([float(pos_prev)], dtype=np.float32)], axis=0)

        with torch.no_grad():
            tx = torch.from_numpy(x).to(self.device).unsqueeze(0)
            raw = self.net(tx)
            ratio = torch.tanh(raw).squeeze(0).item()

        plim = float(self.pos_limit if pos_limit is None else pos_limit)
        return float(np.clip(ratio, -plim, plim))
