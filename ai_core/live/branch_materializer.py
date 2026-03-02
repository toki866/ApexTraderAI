from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ai_core.config.step_b_config import StepBConfig
from ai_core.live.branch_specs import BRANCHES
from ai_core.services.step_b_service import StepBService
from ai_core.services.step_dprime_service import StepDPrimeConfig, StepDPrimeService
from ai_core.types.common import DateRange


@dataclass
class MaterializeResult:
    stepb_executed: bool
    stepb_paths: List[str]
    stepb_pred_k: int
    stepb_horizons: List[int]
    dprime_executed_profiles: List[str]
    dprime_paths: List[str]


class BranchMaterializer:
    def __init__(self, output_root: str = "output") -> None:
        self.output_root = Path(output_root)

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        m = str(mode or "live").strip().lower()
        if m in {"ops", "prod", "production", "real"}:
            return "live"
        return m

    def _check_stepa_live_artifacts(self, symbol: str, mode: str, target_dt: pd.Timestamp) -> Dict[str, str]:
        stepa_dir = self.output_root / "stepA" / mode
        req = [
            stepa_dir / f"stepA_prices_train_{symbol}.csv",
            stepa_dir / f"stepA_prices_test_{symbol}.csv",
            stepa_dir / f"stepA_tech_train_{symbol}.csv",
            stepa_dir / f"stepA_tech_test_{symbol}.csv",
            stepa_dir / f"stepA_split_summary_{symbol}.csv",
        ]
        missing = [str(p) for p in req if not p.exists()]
        if missing:
            raise FileNotFoundError(f"StepA artifacts missing: {missing}")

        split_df = pd.read_csv(stepa_dir / f"stepA_split_summary_{symbol}.csv")
        if {"key", "value"}.issubset(split_df.columns):
            kv = {str(k): str(v) for k, v in zip(split_df["key"], split_df["value"])}
        else:
            kv = {c: str(split_df.iloc[0][c]) for c in split_df.columns}
        test_start = pd.to_datetime(kv.get("test_start"), errors="coerce")
        test_end = pd.to_datetime(kv.get("test_end"), errors="coerce")
        if pd.isna(test_start) or pd.isna(test_end):
            raise ValueError("stepA_split_summary missing test_start/test_end")
        test_start = test_start.normalize()
        test_end = test_end.normalize()
        if not (test_start <= target_dt <= test_end):
            raise RuntimeError(f"target_date={target_dt.date()} is outside StepA test window [{test_start.date()}..{test_end.date()}]")

        return {
            "train_start": str(pd.to_datetime(kv["train_start"]).date()),
            "train_end": str(pd.to_datetime(kv["train_end"]).date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        }

    def _collect_stepb_horizons(self, branches: Iterable[str]) -> List[int]:
        hs = set()
        for b in branches:
            spec = BRANCHES[b]
            hs.update(spec.stepb_required_steps)
        return sorted(int(h) for h in hs)

    def _run_stepb(self, symbol: str, mode: str, split: Dict[str, str], pred_k: int) -> List[str]:
        dr = DateRange(
            train_start=date.fromisoformat(split["train_start"]),
            train_end=date.fromisoformat(split["train_end"]),
            test_start=date.fromisoformat(split["test_start"]),
            test_end=date.fromisoformat(split["test_end"]),
        )
        cfg = StepBConfig(symbol=symbol, date_range=dr, run_mode=mode)
        StepBService(app_config=None).run(cfg)
        stepb_dir = self.output_root / "stepB" / mode
        paths = sorted(str(p) for p in stepb_dir.glob(f"stepB_pred_pathseq_*_h{pred_k:02d}_{symbol}.csv"))
        if not paths:
            raise FileNotFoundError(f"StepB pathseq not generated under {stepb_dir}")
        return paths

    def _run_dprime(self, symbol: str, mode: str, profiles: Sequence[str], pred_k: int, l_past: int) -> List[str]:
        cfg = StepDPrimeConfig(
            symbol=symbol,
            mode=mode,
            output_root=str(self.output_root),
            profiles=tuple(sorted(set(profiles))),
            pred_k=int(pred_k),
            l_past=int(l_past),
        )
        StepDPrimeService().run(cfg)

        out_paths: List[str] = []
        for p in cfg.profiles:
            state_path = self.output_root / "stepDprime" / mode / f"stepDprime_state_test_{p}_{symbol}.csv"
            if not state_path.exists():
                raise FileNotFoundError(f"StepDPrime output missing: {state_path}")
            out_paths.append(str(state_path))
        return out_paths

    def materialize(
        self,
        symbol: str,
        mode: str,
        target_date: str,
        branches_final: Sequence[str],
        refresh_stepb: bool = True,
        refresh_dprime: bool = True,
        pred_k: int = 20,
        l_past: int = 63,
    ) -> MaterializeResult:
        m = self._normalize_mode(mode)
        target_dt = pd.to_datetime(target_date, errors="coerce")
        if pd.isna(target_dt):
            raise ValueError(f"invalid target_date={target_date}")
        target_dt = target_dt.normalize()

        unknown = [b for b in branches_final if b not in BRANCHES]
        if unknown:
            raise ValueError(f"unknown branches: {unknown}")

        split = self._check_stepa_live_artifacts(symbol=symbol, mode=m, target_dt=target_dt)

        stepb_required = any(BRANCHES[b].stepb_required for b in branches_final)
        horizons = self._collect_stepb_horizons(branches_final)
        stepb_paths: List[str] = []
        if stepb_required and refresh_stepb:
            stepb_paths = self._run_stepb(symbol=symbol, mode=m, split=split, pred_k=pred_k)

        dprime_profiles = [BRANCHES[b].dprime_profile for b in branches_final]
        dprime_paths: List[str] = []
        if refresh_dprime:
            dprime_paths = self._run_dprime(symbol=symbol, mode=m, profiles=dprime_profiles, pred_k=pred_k, l_past=l_past)

        return MaterializeResult(
            stepb_executed=bool(stepb_required and refresh_stepb),
            stepb_paths=stepb_paths,
            stepb_pred_k=int(pred_k),
            stepb_horizons=horizons,
            dprime_executed_profiles=sorted(set(dprime_profiles)) if refresh_dprime else [],
            dprime_paths=dprime_paths,
        )
