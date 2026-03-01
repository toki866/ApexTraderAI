from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class FeatureStore:
    output_root: str = "output"
    mode: str = "sim"
    symbol: str = "SOXL"

    def __post_init__(self) -> None:
        self.output_root = str(self.output_root)
        m = str(self.mode or "sim").strip().lower()
        if m in {"ops", "prod", "production", "real"}:
            m = "live"
        self.mode = m
        self._merged_cache: Optional[pd.DataFrame] = None
        self._emb_cache: Dict[str, pd.DataFrame] = {}

    @property
    def out_root(self) -> Path:
        return Path(self.output_root)

    def _load_base(self) -> pd.DataFrame:
        if self._merged_cache is not None:
            return self._merged_cache

        base = self.out_root / "stepA" / self.mode
        p_tr = pd.read_csv(base / f"stepA_prices_train_{self.symbol}.csv")
        p_te = pd.read_csv(base / f"stepA_prices_test_{self.symbol}.csv")
        t_tr = pd.read_csv(base / f"stepA_tech_train_{self.symbol}.csv")
        t_te = pd.read_csv(base / f"stepA_tech_test_{self.symbol}.csv")

        prices = pd.concat([p_tr, p_te], ignore_index=True)
        tech = pd.concat([t_tr, t_te], ignore_index=True)
        for df in (prices, tech):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

        merged = prices.merge(tech, on="Date", how="left", suffixes=("", "_tech"))
        merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        merged = self._ensure_core_features(merged)

        for c in merged.columns:
            if c == "Date":
                continue
            merged[c] = pd.to_numeric(merged[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        self._merged_cache = merged
        return merged

    def _ensure_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "Gap" not in out.columns and "gap" not in out.columns:
            close_prev = out["Close"].astype(float).shift(1)
            out["Gap"] = out["Open"].astype(float) / close_prev.replace(0, np.nan) - 1.0
        elif "gap" in out.columns and "Gap" not in out.columns:
            out["Gap"] = out["gap"]

        if "ATR_norm" not in out.columns and "atr_norm" not in out.columns and "atr_norm_14" not in out.columns:
            high = out["High"].astype(float)
            low = out["Low"].astype(float)
            close_prev = out["Close"].astype(float).shift(1)
            tr = pd.concat([(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean()
            out["ATR_norm"] = atr / close_prev.replace(0, np.nan)
        elif "atr_norm_14" in out.columns and "ATR_norm" not in out.columns:
            out["ATR_norm"] = out["atr_norm_14"]
        elif "atr_norm" in out.columns and "ATR_norm" not in out.columns:
            out["ATR_norm"] = out["atr_norm"]

        if "oc_ret" not in out.columns:
            out["oc_ret"] = out["Close"].astype(float) / out["Open"].astype(float).replace(0, np.nan) - 1.0

        for c in ["Gap", "ATR_norm", "oc_ret"]:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        return out

    def _load_profile_embeddings(self, profile: str) -> pd.DataFrame:
        if profile in self._emb_cache:
            return self._emb_cache[profile]

        bases = [self.out_root / "stepD_prime" / self.mode / "embeddings", self.out_root / "stepDprime" / self.mode / "embeddings"]
        p_all = [b / f"stepDprime_{profile}_{self.symbol}_embeddings_all.csv" for b in bases]
        p_tr = [b / f"stepDprime_{profile}_{self.symbol}_embeddings_train.csv" for b in bases]
        p_te = [b / f"stepDprime_{profile}_{self.symbol}_embeddings_test.csv" for b in bases]

        def _read_one(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)
            dcol = next((c for c in df.columns if str(c).strip().lower() == "date"), df.columns[0])
            if dcol != "Date":
                df = df.rename(columns={dcol: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            emb_cols = sorted([c for c in df.columns if str(c).startswith("emb_")])
            sub = df[["Date"] + emb_cols].copy()
            sub = sub.rename(columns={c: f"dprime_{profile}_emb_{i:03d}" for i, c in enumerate(emb_cols)})
            return sub

        for p in p_all:
            if p.exists():
                emb = _read_one(p)
                self._emb_cache[profile] = emb
                return emb
        for ptr, pte in zip(p_tr, p_te):
            if ptr.exists() and pte.exists():
                emb = pd.concat([_read_one(ptr), _read_one(pte)], ignore_index=True)
                emb = emb.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
                self._emb_cache[profile] = emb
                return emb
        raise FileNotFoundError(f"Missing StepD' embeddings for profile={profile} symbol={self.symbol}")

    @staticmethod
    def infer_profile_from_obs_cols(obs_cols: Iterable[str]) -> Set[str]:
        profiles: Set[str] = set()
        for c in obs_cols:
            s = str(c)
            if not s.startswith("dprime_") or "_emb_" not in s:
                continue
            profiles.add(s[len("dprime_") : s.index("_emb_")])
        return profiles

    def get_row(self, date: str | pd.Timestamp, obs_cols: Iterable[str]) -> Dict[str, float]:
        dt = pd.to_datetime(date, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"invalid date={date}")
        dt = dt.normalize()

        df = self._load_base()
        profiles = self.infer_profile_from_obs_cols(obs_cols)
        for p in sorted(profiles):
            emb = self._load_profile_embeddings(p)
            df = df.merge(emb, on="Date", how="left")

        row = df[df["Date"] == dt]
        if row.empty:
            raise KeyError(f"Date not found in FeatureStore: {dt.date()}")
        rec = row.iloc[-1].to_dict()

        out: Dict[str, float] = {}
        for c in obs_cols:
            v = rec.get(c, 0.0)
            try:
                f = float(v)
            except Exception:
                f = 0.0
            if not np.isfinite(f):
                f = 0.0
            out[str(c)] = f
        return out
