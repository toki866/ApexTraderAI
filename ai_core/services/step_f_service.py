"""StepF router/MARL integration layer.

Role boundary note:
- StepE generates candidate expert actions/ratios.
- StepF performs router / MARL final selection and integration for final decisions.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.errors import PerformanceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from ai_core.clusterers.ticc_clusterer import TICCClusterer, TICCUnavailableError
from ai_core.services.step_dprime_service import _compute_base_features
from ai_core.utils.cluster_stats import compute_cluster_stats, derive_valid_and_rare_clusters
from ai_core.utils.metrics_utils import compute_split_metrics
from ai_core.utils.pipeline_artifact_utils import resolve_stepdprime_root
try:
    from ai_core.utils.timing_logger import TimingLogger
except Exception:  # pragma: no cover
    from contextlib import contextmanager

    class TimingLogger:  # type: ignore[override]
        @staticmethod
        def disabled() -> "TimingLogger":
            return TimingLogger()

        @contextmanager
        def stage(self, _name: str):
            yield

@dataclass
class StepFRouterConfig:
    output_root: str = "output"
    agents: str = ""
    mode: str = "sim"
    phase2_state_path: Optional[str] = None
    use_z_pred: bool = False
    z_pred_source: str = "dprime_mix_h02"
    robust_scaler: bool = True
    pca_n_components: int = 30
    clusterer_type: str = "ticc_raw20_stable"
    ticc_num_clusters: int = 8
    raw20_num_clusters: int = 20
    k_eff_min: int = 12
    cluster_share_min: float = 0.01
    cluster_mean_run_min: float = 3.0
    cluster_main_source: str = "stable"
    enable_raw20_monitor: bool = True
    ticc_window_size: int = 5
    ticc_lambda: float = 0.11
    ticc_beta: float = 600.0
    ticc_max_iter: int = 100
    ticc_threshold: float = 2e-5
    past_window_days: int = 63
    past_resample_len: int = 20
    safe_set: str = "dprime_bnf_h01,dprime_all_features_h01"
    topK: int = 3
    min_samples_regime: int = 20
    fallback_set: str = "all"
    topk_filter_ev_positive: bool = True
    shrink_k: int = 30
    softmax_beta: float = 1.0
    ema_alpha: float = 0.5
    pos_limit: float = 1.0
    trade_cost_bps: float = 15.0
    pos_l2_lambda: float = 0.0
    reward_mode: str = "legacy"
    stepf_compare_reward_modes: bool = False
    stepf_reward_modes: str = ""
    lambda_switch: float = 0.0005
    lambda_churn: float = 0.0002
    lambda_regret: float = 0.2
    lambda_dd: float = 0.05
    eps_ir: float = 1e-8
    verbose: bool = True
    retrain: str = "off"
    branch_id: str = "default"
    input_mode: str = ""
    model_source_dir: str = ""
    device: str = "auto"


StepFConfig = StepFRouterConfig


@dataclass
class StepFResult:
    daily_log_path: str
    summary_path: str
    ratio_path: str = ""
    run_summary_path: str = ""


@dataclass
class StepFModeRunRecord:
    mode: str
    status: str
    output_dir: str
    step_e_root: str
    step_e_daily_log_count: int
    traceback_path: str = ""
    error: str = ""
    files_present: List[str] | None = None


class StepFService:
    DEFAULT_COMPARE_REWARD_MODES: Tuple[str, ...] = (
        "legacy",
        "profit_basic",
        "profit_regret",
        "profit_light_risk",
    )

    def __init__(self, app_config):
        self.app_config = app_config

    def _timing(self) -> TimingLogger:
        t = getattr(self.app_config, "_timing_logger", None)
        return t if isinstance(t, TimingLogger) else TimingLogger.disabled()

    @staticmethod
    def _log_stepf_multi(message: str) -> None:
        print(message)
        print(f"[ONE_TAP]{message}")

    @staticmethod
    def _log_stepf_entry(message: str) -> None:
        print(message)
        print(f"[ONE_TAP]{message}")

    @staticmethod
    def evaluate_final_outputs(output_root: Path, mode: str, symbol: str) -> Dict[str, object]:
        """Evaluate StepF final artifact health at end-of-step.

        Fatal only when required artifact/shape is broken.
        Non-fatal notes (e.g. split missing) are warnings.
        """
        base = Path(output_root) / "stepF" / mode
        equity_path = base / f"stepF_equity_marl_{symbol}.csv"
        daily_log_router_path = base / f"stepF_daily_log_router_{symbol}.csv"
        daily_log_marl_path = base / f"stepF_daily_log_marl_{symbol}.csv"
        summary_path = base / f"stepF_summary_router_{symbol}.json"
        audit_path = base / f"stepF_audit_router_{symbol}.json"

        warnings: List[str] = []
        errors: List[str] = []
        row_count = 0

        print(f"[STEPF_FINAL] equity_path={equity_path} exists={equity_path.exists()}")
        print(f"[STEPF_FINAL] daily_log_router_path={daily_log_router_path} exists={daily_log_router_path.exists()}")
        print(f"[STEPF_FINAL] daily_log_marl_path={daily_log_marl_path} exists={daily_log_marl_path.exists()}")
        print(f"[STEPF_FINAL] summary_path={summary_path} exists={summary_path.exists()}")
        print(f"[STEPF_FINAL] audit_path={audit_path} exists={audit_path.exists()}")

        if not equity_path.exists():
            errors.append("missing_equity_csv")
        else:
            try:
                eq_df = pd.read_csv(equity_path)
                row_count = int(len(eq_df))
                if row_count <= 0:
                    errors.append("equity_csv_empty")
                required_cols = ["Date", "ratio", "ret", "equity"]
                missing_cols = [c for c in required_cols if c not in eq_df.columns]
                if missing_cols:
                    errors.append(f"equity_csv_missing_cols:{','.join(missing_cols)}")
            except Exception as exc:
                errors.append(f"equity_csv_unreadable:{type(exc).__name__}")

        if not daily_log_router_path.exists():
            warnings.append("missing_daily_log_router")
        if not daily_log_marl_path.exists():
            warnings.append("missing_daily_log_marl")
        if not summary_path.exists():
            warnings.append("missing_summary_router")
        if not audit_path.exists():
            warnings.append("missing_audit_router")
        else:
            try:
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                note = str(summary_payload.get("note", "")).strip()
                if note and "split missing" in note.lower():
                    warnings.append(f"split_warning:{note}")
            except Exception as exc:
                warnings.append(f"summary_unreadable:{type(exc).__name__}")

        final_status = "fail" if errors else ("warn" if warnings else "complete")
        return_code = 1 if errors else 0
        print(f"[STEPF_FINAL] row_count={row_count}")
        print(f"[STEPF_FINAL] final_status={final_status}")
        print(f"[STEPF_FINAL] return_code={return_code}")
        if warnings:
            print(f"[STEPF_FINAL] warnings={';'.join(warnings)}")
        if errors:
            print(f"[STEPF_FINAL] errors={';'.join(errors)}")

        return {
            "final_status": final_status,
            "return_code": return_code,
            "warnings": warnings,
            "errors": errors,
            "row_count": row_count,
            "equity_path": str(equity_path),
            "daily_log_router_path": str(daily_log_router_path),
            "daily_log_marl_path": str(daily_log_marl_path),
            "summary_path": str(summary_path),
            "audit_path": str(audit_path),
        }

    def run(self, date_range, symbol: str, mode: Optional[str] = None) -> StepFResult:
        out_root: Optional[Path] = None
        stepf_dir: Optional[Path] = None
        try:
            timing = self._timing()
            with timing.stage("stepF.total"):
                self._log_stepf_entry("[STEPF_ENTRY] begin")
            pre_output_root = self._resolve_output_root(getattr(self.app_config, "output_root", "output"))
            pre_mode = str(mode or getattr(date_range, "mode", None) or "sim").strip().lower()
            if pre_mode in {"ops", "prod", "production", "real"}:
                pre_mode = "live"
            pre_stepf_root = pre_output_root / "stepF"
            pre_stepf_root.mkdir(parents=True, exist_ok=True)
            (pre_stepf_root / "sim").mkdir(parents=True, exist_ok=True)
            (pre_stepf_root / pre_mode).mkdir(parents=True, exist_ok=True)
            self._log_stepf_entry("[STEPF_ENTRY] before_config_load")
            cfg: StepFRouterConfig = getattr(self.app_config, "stepF", None)
            if cfg is None:
                raise ValueError("app_config.stepF is missing")
            self._log_stepf_entry("[STEPF_ENTRY] after_config_load")

            resolved_mode = str(mode or getattr(date_range, "mode", None) or cfg.mode or "sim").strip().lower()
            if resolved_mode in {"ops", "prod", "production", "real"}:
                resolved_mode = "live"
            out_root = self._resolve_output_root(cfg.output_root)
            stepf_root = out_root / "stepF"
            stepf_dir = stepf_root / resolved_mode
            stepf_root.mkdir(parents=True, exist_ok=True)
            stepf_dir.mkdir(parents=True, exist_ok=True)

            self._log_stepf_entry(f"[STEPF_ENTRY] output_root={out_root}")
            self._log_stepf_entry(f"[STEPF_ENTRY] stepf_dir={stepf_dir}")

            self._log_stepf_entry("[STEPF_ENTRY] before_reward_modes_resolve")
            compare_enabled, reward_modes = self._resolve_reward_modes(cfg)
            self._log_stepf_entry(f"[STEPF_ENTRY] compare_enabled={str(compare_enabled).lower()}")
            self._log_stepf_entry(f"[STEPF_ENTRY] reward_modes={','.join(reward_modes)}")
            self._log_stepf_entry(f"[STEPF_ENTRY] config_path={getattr(self.app_config, '_config_path', '(unknown)')}")
            self._log_stepf_entry(f"[STEPF_ENTRY] symbol={symbol}")

            self._log_stepf_multi(f"[STEPF_MULTI] compare_enabled={str(compare_enabled).lower()}")
            self._log_stepf_multi(f"[STEPF_MULTI] reward_modes={','.join(reward_modes)}")

            self._log_stepf_entry("[STEPF_ENTRY] before_stepe_daily_logs_resolve")
            step_e_root = out_root / "stepE" / resolved_mode
            step_e_daily_logs = sorted(step_e_root.glob(f"stepE_daily_log_*_{symbol}.csv"))
            self._log_stepf_entry("[STEPF_ENTRY] after_stepe_daily_logs_resolve")
            self._log_stepf_multi(f"[STEPF_MULTI] mode_input_stepE_root={step_e_root}")
            self._log_stepf_multi(f"[STEPF_MULTI] mode_input_stepE_daily_log_count={len(step_e_daily_logs)}")

            self._log_stepf_entry("[STEPF_ENTRY] output_dir_prepare_done")
            self._log_stepf_entry("[STEPF_ENTRY] before_mode_loop")
            primary_result: Optional[StepFResult] = None
            mode_records: List[StepFModeRunRecord] = []
            for reward_mode in reward_modes:
                with timing.stage("stepF.reward_mode.total", agent_id=str(reward_mode), meta={"reward_mode": str(reward_mode), "stage_group": "reward_mode", "agent_kind": "reward_mode"}):
                    mode_cfg = deepcopy(cfg)
                    mode_cfg.reward_mode = reward_mode
                    persist_primary_outputs = primary_result is None
                    self._log_stepf_entry(f"[STEPF_ENTRY] before_mode={reward_mode}")
                    self._log_stepf_multi(f"[STEPF_MULTI] mode_start={reward_mode}")
                    retrain = "on" if str(getattr(mode_cfg, "retrain", "off")).lower() == "on" else "off"
                    mode_dir = self._reward_dir(out_root=out_root, mode=resolved_mode, retrain=retrain, reward_mode=reward_mode)
                    self._log_stepf_multi(f"[STEPF_MULTI] mode_output_dir={mode_dir}")
                    self._log_stepf_multi(f"[STEPF_MULTI] mode_input_stepE_root={step_e_root}")
                    self._log_stepf_multi(f"[STEPF_MULTI] mode_input_stepE_daily_log_count={len(step_e_daily_logs)}")

                    try:
                        self._log_stepf_entry("[STEPF_ENTRY] before_service_run")
                        mode_result = self._run_router(
                            mode_cfg,
                            date_range,
                            symbol=symbol,
                            mode=resolved_mode,
                            persist_primary_outputs=persist_primary_outputs,
                        )
                        self._log_stepf_entry("[STEPF_ENTRY] after_service_run")
                        written_files = [
                            f"stepF_equity_marl_{symbol}.csv",
                            f"stepF_daily_log_router_{symbol}.csv",
                            f"stepF_daily_log_marl_{symbol}.csv",
                            f"stepF_summary_router_{symbol}.json",
                        ]
                        self._log_stepf_multi(f"[STEPF_MULTI] mode_written_files={','.join(written_files)}")
                        self._log_stepf_multi(f"[STEPF_MULTI] mode_success={reward_mode}")
                        mode_records.append(
                            StepFModeRunRecord(
                                mode=reward_mode,
                                status="SUCCESS",
                                output_dir=str(mode_dir),
                                step_e_root=str(step_e_root),
                                step_e_daily_log_count=len(step_e_daily_logs),
                                files_present=[p.name for p in sorted(mode_dir.glob("*"))],
                            )
                        )
                        if primary_result is None:
                            primary_result = mode_result
                    except Exception as exc:
                        mode_dir.mkdir(parents=True, exist_ok=True)
                        tb_text = traceback.format_exc()
                        traceback_path = mode_dir / f"stepF_traceback_{symbol}.log"
                        traceback_path.write_text(tb_text, encoding="utf-8")
                        files_present = [p.name for p in sorted(mode_dir.glob("*"))]
                        self._log_stepf_multi(f"[STEPF_MULTI] mode_fail={reward_mode} exc={type(exc).__name__}: {exc}")
                        self._log_stepf_multi(f"[STEPF_MULTI] mode_traceback_path={traceback_path}")
                        self._log_stepf_multi(f"[STEPF_MULTI] mode_existing_files={','.join(files_present) if files_present else '(none)'}")
                        print(tb_text)
                        print(f"[ONE_TAP][STEPF_MULTI] mode_fail_traceback_begin={reward_mode}")
                        print(tb_text)
                        print(f"[ONE_TAP][STEPF_MULTI] mode_fail_traceback_end={reward_mode}")
                        mode_records.append(
                            StepFModeRunRecord(
                                mode=reward_mode,
                                status="FAIL",
                                output_dir=str(mode_dir),
                                step_e_root=str(step_e_root),
                                step_e_daily_log_count=len(step_e_daily_logs),
                                traceback_path=str(traceback_path),
                                error=f"{type(exc).__name__}: {exc}",
                                files_present=files_present,
                            )
                        )

            multi_summary_path = out_root / "stepF" / resolved_mode / f"stepF_multi_mode_summary_{symbol}.json"
            multi_summary_path.parent.mkdir(parents=True, exist_ok=True)
            success_modes = [r.mode for r in mode_records if r.status == "SUCCESS"]
            failed_modes = [r.mode for r in mode_records if r.status == "FAIL"]
            missing_outputs = [
                f"reward_{r.mode}/stepF_equity_marl_{symbol}.csv"
                for r in mode_records
                if r.status != "SUCCESS"
            ]
            multi_summary = {
                "compare_enabled": compare_enabled,
                "reward_modes": reward_modes,
                "success_modes": success_modes,
                "failed_modes": failed_modes,
                "missing_outputs": missing_outputs,
                "records": [r.__dict__ for r in mode_records],
            }
            multi_summary_path.write_text(json.dumps(multi_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log_stepf_multi(f"[STEPF_MULTI] success_modes={','.join(success_modes) if success_modes else '(none)'}")
            self._log_stepf_multi(f"[STEPF_MULTI] failed_modes={','.join(failed_modes) if failed_modes else '(none)'}")
            self._log_stepf_multi(f"[STEPF_MULTI] missing_outputs={','.join(missing_outputs) if missing_outputs else '(none)'}")
            self._log_stepf_multi(f"[STEPF_MULTI] summary_path={multi_summary_path}")

            if primary_result is None:
                failed = [f"{r.mode}:{r.error}" for r in mode_records if r.status == "FAIL"]
                raise RuntimeError("StepF reward mode execution failed for all modes: " + " | ".join(failed))
            return primary_result
        except Exception as exc:
            tb_text = traceback.format_exc()
            cwd = Path(os.getcwd())
            stepf_exists = bool(stepf_dir and stepf_dir.exists())
            parent = stepf_dir.parent if stepf_dir is not None else (out_root.parent if out_root is not None else cwd)
            try:
                parent_listing = ",".join(sorted(p.name for p in parent.iterdir()))
            except Exception as list_exc:  # pragma: no cover
                parent_listing = f"<failed:{type(list_exc).__name__}:{list_exc}>"

            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_exception={type(exc).__name__}: {exc}")
            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_exception_repr={repr(exc)}")
            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_cwd={cwd}")
            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_output_root={out_root}")
            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_stepf_dir={stepf_dir}")
            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_stepf_dir_exists={str(stepf_exists).lower()}")
            self._log_stepf_entry(f"[STEPF_ENTRY] wrapper_parent_listing={parent_listing if parent_listing else '(empty)'}")
            self._log_stepf_entry("[STEPF_ENTRY] wrapper_traceback_begin")
            for line in tb_text.rstrip().splitlines():
                self._log_stepf_entry(line)
            self._log_stepf_entry("[STEPF_ENTRY] wrapper_traceback_end")
            raise

    def run_live(self, date_range, symbol: str, retrain: str = "off", branch_id: str = "default", data_cutoff: str = "") -> StepFResult:
        cfg: StepFRouterConfig = deepcopy(getattr(self.app_config, "stepF", None))
        if cfg is None:
            raise ValueError("app_config.stepF is missing")
        cfg.retrain = "on" if str(retrain).lower() == "on" else "off"
        cfg.branch_id = str(branch_id or "default")
        return self._run_router(cfg, date_range, symbol=symbol, mode="live", data_cutoff=data_cutoff)

    @classmethod
    def _resolve_reward_modes(cls, cfg: StepFRouterConfig) -> Tuple[bool, List[str]]:
        compare_enabled = bool(getattr(cfg, "stepf_compare_reward_modes", False))
        raw_modes = str(getattr(cfg, "stepf_reward_modes", "") or "").strip()
        parsed: List[str] = []
        if raw_modes:
            parsed = [m.strip().lower() for m in raw_modes.split(",") if m.strip()]
        if compare_enabled and not parsed:
            parsed = list(cls.DEFAULT_COMPARE_REWARD_MODES)
        if not parsed:
            parsed = [str(getattr(cfg, "reward_mode", "legacy") or "legacy").strip().lower()]
        normalized: List[str] = []
        for mode_name in parsed:
            if mode_name not in normalized:
                normalized.append(mode_name)
        return compare_enabled, normalized

    @staticmethod
    def _reward_dir(out_root: Path, mode: str, retrain: str, reward_mode: str) -> Path:
        out_dir = (out_root / "stepF" / mode / f"retrain_{retrain}") if mode == "live" else (out_root / "stepF" / mode)
        return out_dir / f"reward_{reward_mode}"

    @staticmethod
    def _normalize_agent_names(raw: object) -> List[str]:
        names = [a.strip() for a in str(raw or "").split(",") if a and a.strip()]
        out: List[str] = []
        for name in names:
            if name not in out:
                out.append(name)
        return out

    @staticmethod
    def _normalize_absolute_path(raw_path: object) -> Path:
        s = str(raw_path or "").strip()
        if re.match(r"^[A-Za-z]:[\\/]", s):
            return Path(s)
        p = Path(s).expanduser() if s else Path.cwd()
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve(strict=False)

    def _resolve_output_root(self, cfg_output_root: object) -> Path:
        cfg_raw = str(cfg_output_root or "").strip()
        app_raw = str(getattr(self.app_config, "output_root", "") or "").strip()
        effective_raw = ""
        for attr in ("effective_output_root", "_effective_output_root", "resolved_output_root", "canonical_output_root"):
            candidate = str(getattr(self.app_config, attr, "") or "").strip()
            if candidate:
                effective_raw = candidate
                break

        if cfg_raw and cfg_raw.lower() != "output":
            chosen = cfg_raw
        elif effective_raw:
            chosen = effective_raw
        elif app_raw and app_raw.lower() != "output":
            chosen = app_raw
        elif app_raw:
            chosen = app_raw
        else:
            chosen = cfg_raw or "output"
        return self._normalize_absolute_path(chosen)

    def _resolve_step_e_root_candidates(self, out_root: Path, input_mode: str) -> List[Path]:
        candidates: List[Path] = [self._normalize_absolute_path(out_root / "stepE" / input_mode)]
        if str(input_mode).strip().lower() == "sim":
            candidates.append(self._normalize_absolute_path(out_root / "stepE" / "sim"))

        for attr in ("effective_output_root", "_effective_output_root", "resolved_output_root", "canonical_output_root"):
            raw = str(getattr(self.app_config, attr, "") or "").strip()
            if raw:
                candidates.append(self._normalize_absolute_path(Path(raw) / "stepE" / input_mode))

        unique_candidates: List[Path] = []
        for c in candidates:
            if c not in unique_candidates:
                unique_candidates.append(c)
        return unique_candidates

    @staticmethod
    def _resolve_device(requested: str) -> tuple[str, List[str]]:
        req = str(requested or "auto").strip().lower()
        warnings: List[str] = []
        if req in ("", "none", "auto"):
            actual = "cuda" if torch.cuda.is_available() else "cpu"
            if actual == "cpu":
                warnings.append("auto_device_resolved_to_cpu")
            return actual, warnings
        if req.startswith("cuda") and not torch.cuda.is_available():
            warnings.append("cuda_requested_but_unavailable_fallback_cpu")
            return "cpu", warnings
        return req, warnings

    @staticmethod
    def _extract_agent_name_from_daily_log(path: Path, symbol: str) -> Optional[str]:
        file_name = path.name
        match = re.match(rf"^stepE_daily_log_(.+)_{re.escape(symbol)}\.csv$", file_name, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip("_") or None

    def _resolve_agents(self, out_root: Path, input_mode: str, symbol: str, requested_agents_raw: object) -> Tuple[List[str], List[Path], List[str], Path]:
        candidate_roots = self._resolve_step_e_root_candidates(out_root=out_root, input_mode=input_mode)
        selected_step_e_root = candidate_roots[0]
        discovered_daily_logs: List[Path] = []
        for candidate_root in candidate_roots:
            logs = sorted(p.resolve(strict=False) for p in candidate_root.glob("stepE_daily_log_*.csv"))
            if logs:
                selected_step_e_root = candidate_root
                break
            if candidate_root.exists():
                selected_step_e_root = candidate_root

        discovered_daily_logs = sorted(p.resolve(strict=False) for p in selected_step_e_root.glob("stepE_daily_log_*.csv"))

        requested_agents = self._normalize_agent_names(requested_agents_raw)

        discovered_agents: List[str] = []
        for daily_log_path in discovered_daily_logs:
            agent_name = self._extract_agent_name_from_daily_log(daily_log_path, symbol)
            if agent_name and agent_name not in discovered_agents:
                discovered_agents.append(agent_name)

        resolved_agents = list(requested_agents) if requested_agents else list(discovered_agents)

        discovered_logs_text = ",".join(str(p) for p in discovered_daily_logs) if discovered_daily_logs else "(none)"
        resolved_agents_text = ",".join(resolved_agents) if resolved_agents else "(none)"
        print("[STEPF_AGENTS] begin")
        print(f"[STEPF_AGENTS] candidate_stepE_roots={','.join(str(p) for p in candidate_roots)}")
        print(f"[STEPF_AGENTS] selected_stepE_root={selected_step_e_root}")
        print(f"[STEPF_AGENTS] selected_stepE_root_exists={str(selected_step_e_root.exists()).lower()}")
        print(f"[STEPF_AGENTS] input_stepE_root={selected_step_e_root}")
        print(f"[STEPF_AGENTS] requested_agents_raw={requested_agents_raw}")
        print(f"[STEPF_AGENTS] requested_agents_normalized={','.join(requested_agents) if requested_agents else '(none)'}")
        print(f"[STEPF_AGENTS] discovered_daily_logs_count={len(discovered_daily_logs)}")
        print(f"[STEPF_AGENTS] discovered_daily_logs={discovered_logs_text}")
        print(f"[STEPF_AGENTS] resolved_agents_count={len(resolved_agents)}")
        print(f"[STEPF_AGENTS] resolved_agents={resolved_agents_text}")

        return resolved_agents, discovered_daily_logs, requested_agents, selected_step_e_root

    def _run_router(self, cfg: StepFRouterConfig, date_range, symbol: str, mode: str, data_cutoff: str = "", persist_primary_outputs: bool = True) -> StepFResult:
        timing = self._timing()
        with timing.stage("stepF.total"):
            out_root = self._resolve_output_root(cfg.output_root)
            print(f"[STEPF_ROOT] wrapper_cfg_output_root={cfg.output_root}")
            print(f"[STEPF_ROOT] service_app_output_root={getattr(self.app_config, 'output_root', '')}")
            print(f"[STEPF_ROOT] effective_output_root={out_root}")
            retrain = "on" if str(getattr(cfg, "retrain", "off")).lower() == "on" else "off"
            reward_mode = str(getattr(cfg, "reward_mode", "legacy") or "legacy").strip().lower()
            out_dir = (out_root / "stepF" / mode / f"retrain_{retrain}") if mode == "live" else (out_root / "stepF" / mode)
            reward_dir = self._reward_dir(out_root=out_root, mode=mode, retrain=retrain, reward_mode=reward_mode)
            router_dir = out_dir / "router"
            phase2_dir = out_dir / "phase2"
            router_dir.mkdir(parents=True, exist_ok=True)
            phase2_dir.mkdir(parents=True, exist_ok=True)
            reward_dir.mkdir(parents=True, exist_ok=True)

            input_mode = str(getattr(cfg, "input_mode", "") or mode)
            if mode == "live" and not (out_root / "stepA" / input_mode).exists():
                input_mode = "sim"

            agents, discovered_daily_logs, requested_agents, selected_step_e_root = self._resolve_agents(
                out_root=out_root,
                input_mode=input_mode,
                symbol=symbol,
                requested_agents_raw=getattr(cfg, "agents", ""),
            )
            if not agents:
                raise ValueError(
                    "StepF agents is empty "
                    f"(input_stepE_root={selected_step_e_root}, "
                    f"discovered_daily_logs_count={len(discovered_daily_logs)}, "
                    f"requested_agents={','.join(requested_agents) if requested_agents else '(none)'})"
                )

            safe_set = [a.strip() for a in str(cfg.safe_set or "").split(",") if a.strip()]
            safe_set = [a for a in safe_set if a in agents]
            if "dprime_mix_3scale" in agents and "dprime_mix_3scale" not in safe_set:
                safe_set.append("dprime_mix_3scale")
            if len(safe_set) < 2:
                for a in agents:
                    if a not in safe_set:
                        safe_set.append(a)
                    if len(safe_set) >= min(2, len(agents)):
                        break

            print(
                f"[STEPF_INPUT] symbol={symbol} mode={mode} input_mode={input_mode} "
                f"output_root={out_root} agents={','.join(agents)}"
            )
            with timing.stage("stepF.load_inputs"):
                prices_soxl = self._load_stepa_price_tech(out_root, input_mode, "SOXL")
                prices_soxs = self._load_stepa_price_tech(out_root, input_mode, "SOXS")
                logs_map = self._load_stepe_logs(selected_step_e_root, symbol, agents)
            if not logs_map:
                raise RuntimeError("StepF dependency missing: no StepE daily logs available for any configured agent")
            agents = [a for a in agents if a in logs_map]
            soxl_px = prices_soxl[["Date", "price_exec"]].rename(columns={"price_exec": "price_soxl"})
            soxs_px = prices_soxs[["Date", "price_exec"]].rename(columns={"price_exec": "price_soxs"})
            price_pair = soxl_px.merge(soxs_px, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
            price_pair["r_soxl"] = (price_pair["price_soxl"].shift(-1) / price_pair["price_soxl"] - 1.0).fillna(0.0)
            price_pair["r_soxs"] = (price_pair["price_soxs"].shift(-1) / price_pair["price_soxs"] - 1.0).fillna(0.0)

            if cfg.phase2_state_path:
                phase2 = pd.read_csv(cfg.phase2_state_path)
                phase2["Date"] = pd.to_datetime(phase2["Date"], errors="coerce")
            elif mode == "live" and retrain == "off":
                source_dir = Path(getattr(cfg, "model_source_dir", "") or (out_root / "stepF" / "live" / "retrain_on"))
                source_p = source_dir / "phase2" / f"phase2_state_{symbol}.csv"
                if not source_p.exists():
                    source_p = out_root / "stepF" / "sim" / "phase2" / f"phase2_state_{symbol}.csv"
                phase2 = pd.read_csv(source_p)
                phase2["Date"] = pd.to_datetime(phase2["Date"], errors="coerce")
            else:
                with timing.stage("stepF.phase2", meta={"fallback_used": bool((getattr(self, "_last_cluster_diag", {}) or {}).get("fallback_used", False)), "regime_id": "phase2", "cluster_id_stable": "phase2"}):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=PerformanceWarning)
                        phase2 = self._build_phase2_state(
                            cfg=cfg,
                            date_range=date_range,
                            symbol=symbol,
                            mode=input_mode,
                            out_root=out_root,
                            price_tech=prices_soxl,
                        )

            phase2_path = phase2_dir / f"phase2_state_{symbol}.csv"
            phase2.to_csv(phase2_path, index=False)

            phase2_cols = [
                c for c in (
                    "Date",
                    "regime_id",
                    "cluster_id_stable",
                    "cluster_id_raw20",
                    "rare_flag_raw20",
                    "confidence_stable",
                    "confidence_raw20",
                    "state_gap",
                    "state_atr_norm",
                    "state_ret_1",
                    "state_ret_5",
                    "state_ret_20",
                    "state_range_atr",
                    "state_body_ratio",
                    "state_bnf_score",
                    "state_context_norm",
                )
                if c in phase2.columns
            ]
            merged = phase2[phase2_cols].merge(price_pair[["Date", "r_soxl", "r_soxs"]], on="Date", how="inner")
            for agent in agents:
                adf = logs_map[agent][["Date", "Split", "ratio", "stepE_ret_for_stats"]].copy()
                adf = adf.rename(columns={"ratio": f"ratio_{agent}", "stepE_ret_for_stats": f"ret_{agent}", "Split": f"Split_{agent}"})
                merged = merged.merge(adf, on="Date", how="left")

            merged = merged.sort_values("Date").reset_index(drop=True)
            merged["Split"] = self._assign_split_by_date(merged["Date"], date_range)

            if mode == "live" and retrain == "off":
                source_dir = Path(getattr(cfg, "model_source_dir", "") or (out_root / "stepF" / "live" / "retrain_on"))
                edge_path_src = source_dir / "router" / f"regime_edge_table_{symbol}.csv"
                allow_path_src = source_dir / "router" / f"router_allowlist_{symbol}.csv"
                if not edge_path_src.exists() or not allow_path_src.exists():
                    edge_path_src = out_root / "stepF" / "sim" / "router" / f"regime_edge_table_{symbol}.csv"
                    allow_path_src = out_root / "stepF" / "sim" / "router" / f"router_allowlist_{symbol}.csv"
                edge_table = pd.read_csv(edge_path_src)
                allowlist = pd.read_csv(allow_path_src)
            else:
                with timing.stage("stepF.build_edge_table"):
                    edge_table = self._build_regime_edge_table(merged, agents, cfg)
                with timing.stage("stepF.allowlist"):
                    allowlist = self._build_allowlist(edge_table=edge_table, agents=agents, safe_set=safe_set, cfg=cfg)
            edge_path = router_dir / f"regime_edge_table_{symbol}.csv"
            edge_table.to_csv(edge_path, index=False)

            safe_set = self._stabilize_safe_set(safe_set=safe_set, edge_table=edge_table, agents=agents)

            allow_path = router_dir / f"router_allowlist_{symbol}.csv"
            allowlist.to_csv(allow_path, index=False)

            context_profiles = self._build_context_profiles(merged=merged, agents=agents)
            actual_device_name, device_warnings = self._resolve_device(getattr(cfg, "device", "auto"))
            with timing.stage("stepF.router_sim", agent_id=str(reward_mode), meta={"reward_mode": str(reward_mode), "compare_mode": str(mode), "agent_kind": "reward_mode", "fallback_used": bool((getattr(self, "_last_cluster_diag", {}) or {}).get("fallback_used", False))}) :
                daily = self._run_router_sim(
                    merged=merged,
                    agents=agents,
                    edge_table=edge_table,
                    allowlist=allowlist,
                    safe_set=safe_set,
                    cfg=cfg,
                    context_profiles=context_profiles,
                    device_name=actual_device_name,
                )

            if data_cutoff:
                cutoff_dt = pd.to_datetime(data_cutoff, errors="coerce")
                if pd.notna(cutoff_dt):
                    daily = daily[pd.to_datetime(daily["Date"], errors="coerce") <= cutoff_dt].copy()

            log_router_path = out_dir / f"stepF_daily_log_router_{symbol}.csv"
            summary_router_path = out_dir / f"stepF_summary_router_{symbol}.json"
            log_marl_path = out_dir / f"stepF_daily_log_marl_{symbol}.csv"
            eq_marl_path = out_dir / f"stepF_equity_marl_{symbol}.csv"
            ratio_live_path = out_dir / f"stepF_ratio_live_retrain_{retrain}_{symbol}.csv"

            with timing.stage("stepF.write_outputs", agent_id=str(reward_mode), meta={"reward_mode": str(reward_mode), "agent_kind": "reward_mode", "fallback_used": bool((getattr(self, "_last_cluster_diag", {}) or {}).get("fallback_used", False))}):
                cluster_diag = getattr(self, "_last_cluster_diag", {}) or {}
                daily = daily.copy()
                daily["selected_expert_definition"] = "argmax mixture weight for the day after EMA-smoothed routing weights are normalized"
                daily["device_requested"] = str(getattr(cfg, "device", "auto"))
                daily["device_execution"] = actual_device_name
                daily["clusterer_type_requested"] = cluster_diag.get("clusterer_type_requested", "")
                daily["clusterer_type_used"] = cluster_diag.get("clusterer_type_used", "")
                daily["fallback_type"] = cluster_diag.get("fallback_type", "")
                daily["fallback_used"] = bool(cluster_diag.get("fallback_used", False))
                daily["fallback_reason"] = cluster_diag.get("fallback_reason", "")
                daily["cluster_main_source"] = cluster_diag.get("cluster_main_source", "stable")
                eq_df = daily[daily["Split"] == "test"][["Date", "Split", "ratio", "ret", "equity"]].copy()
                if persist_primary_outputs:
                    daily.to_csv(log_router_path, index=False)
                    daily.to_csv(log_marl_path, index=False)
                    eq_df.to_csv(eq_marl_path, index=False)
                    daily[["Date", "ratio"]].to_csv(ratio_live_path, index=False)

                reward_router_path = reward_dir / f"stepF_daily_log_router_{symbol}.csv"
                reward_marl_path = reward_dir / f"stepF_daily_log_marl_{symbol}.csv"
                reward_eq_path = reward_dir / f"stepF_equity_marl_{symbol}.csv"
                reward_summary_path = reward_dir / f"stepF_summary_router_{symbol}.json"
                daily.to_csv(reward_router_path, index=False)
                daily.to_csv(reward_marl_path, index=False)
                eq_df.to_csv(reward_eq_path, index=False)

                daily_for_summary = daily.copy()
                if "Split" not in daily_for_summary.columns:
                    daily_for_summary["Split"] = self._assign_split_by_date(daily_for_summary["Date"], date_range)
                metrics = compute_split_metrics(daily_for_summary, split="test", equity_col="equity", ret_col="ret")
                test_df = daily_for_summary[daily_for_summary["Split"].astype(str).str.lower() == "test"].copy()
                summary = {
                    **metrics,
                    "test_return_pct": metrics["total_return_pct"],
                    "test_sharpe": metrics["sharpe"],
                    "test_max_dd": metrics["max_dd_pct"],
                    "total_return": metrics["total_return_pct"] / 100.0 if np.isfinite(metrics["total_return_pct"]) else float("nan"),
                    "max_drawdown": metrics["max_dd_pct"],
                    "equity_end": metrics["equity_end"],
                    "num_trades": int(np.sum(np.abs(np.diff(test_df["ratio"].astype(float).to_numpy())) > 1e-9)) if not test_df.empty else 0,
                    "turnover_sum": float(test_df["turnover"].astype(float).sum()) if "turnover" in test_df.columns and not test_df.empty else float("nan"),
                    "turnover_mean": float(test_df["turnover"].astype(float).mean()) if "turnover" in test_df.columns and not test_df.empty else float("nan"),
                }
                summary.update({"mode": mode, "symbol": symbol, "agents": agents})
                summary.update(
                    {
                        "selected_expert_definition": "argmax mixture weight for the day after EMA-smoothed routing weights are normalized",
                        "device_requested": str(getattr(cfg, "device", "auto")),
                        "device_execution": actual_device_name,
                        "device_warnings": list(device_warnings),
                        "clusterer_type_requested": cluster_diag.get("clusterer_type_requested", ""),
                        "clusterer_type_used": cluster_diag.get("clusterer_type_used", ""),
                        "fallback_type": cluster_diag.get("fallback_type", ""),
                        "fallback_used": bool(cluster_diag.get("fallback_used", False)),
                        "fallback_reason": cluster_diag.get("fallback_reason", ""),
                        "backend_resolved_name": cluster_diag.get("backend_resolved_name", ""),
                        "backend_predict_methods": list(cluster_diag.get("backend_predict_methods", []) or []),
                        "backend_methods": list(cluster_diag.get("backend_methods", []) or []),
                        "diagnostic_stage": cluster_diag.get("diagnostic_stage", ""),
                        "raw20_num_clusters_requested": int(cluster_diag.get("raw20_num_clusters_requested", 0) or 0),
                        "raw20_regime_count": int(cluster_diag.get("raw20_regime_count", 0) or 0),
                        "k_raw": int(cluster_diag.get("k_raw", cluster_diag.get("raw20_num_clusters_requested", 0)) or 0),
                        "k_valid": int(cluster_diag.get("k_valid", 0) or 0),
                        "k_eff": int(cluster_diag.get("k_eff", 0) or 0),
                        "k_eff_min": int(cluster_diag.get("k_eff_min", 0) or 0),
                        "stable_regime_count": int(cluster_diag.get("stable_regime_count", 0) or 0),
                        "cluster_main_source": cluster_diag.get("cluster_main_source", "stable"),
                        "rare_flag_raw20_count": int(cluster_diag.get("rare_flag_raw20_count", 0) or 0),
                    }
                )
                policy_compare = self._build_policy_compare_audit(daily=daily, merged=merged, agents=agents)
                summary["policy_compare"] = policy_compare["summary"]
                if persist_primary_outputs:
                    summary_router_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                summary_with_reward = {**summary, "reward_mode": reward_mode}
                reward_summary_path.write_text(json.dumps(summary_with_reward, ensure_ascii=False, indent=2), encoding="utf-8")
                stepf_audit_path = out_dir / f"stepF_audit_router_{symbol}.json"
                audit_dir = out_root / "audit" / mode
                audit_dir.mkdir(parents=True, exist_ok=True)
                compare_audit_path = audit_dir / f"stepF_policy_compare_{symbol}.json"
                audit_payload = {
                    "mode": mode,
                    "symbol": symbol,
                    "reward_mode": reward_mode,
                    "selected_expert_definition": "argmax mixture weight for the day after EMA-smoothed routing weights are normalized",
                    "device_requested": str(getattr(cfg, "device", "auto")),
                    "device_execution": actual_device_name,
                    "device_warnings": list(device_warnings),
                    "ratio_distribution": {
                        "mean": float(daily["ratio"].mean()) if not daily.empty else 0.0,
                        "std": float(daily["ratio"].std(ddof=0)) if not daily.empty else 0.0,
                        "min": float(daily["ratio"].min()) if not daily.empty else 0.0,
                        "max": float(daily["ratio"].max()) if not daily.empty else 0.0,
                    },
                    "expert_selection_frequency": policy_compare["expert_frequency"],
                    "cluster_regime_branch_stats": policy_compare["branch_stats"],
                    "reward_change_count": int((daily["reward"].diff().abs() > 1e-12).sum()) if "reward" in daily.columns else 0,
                    "equity_change_count": int((daily["equity"].diff().abs() > 1e-12).sum()) if "equity" in daily.columns else 0,
                    "all_zero_input_detected": bool(policy_compare["all_zero_input_detected"]),
                    "constant_output_detected": bool(policy_compare["constant_output_detected"]),
                    "policy_compare": policy_compare["summary"],
                }
                if persist_primary_outputs:
                    stepf_audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                compare_audit_path.write_text(json.dumps(policy_compare, ensure_ascii=False, indent=2), encoding="utf-8")

            if cfg.verbose:
                print(f"[StepF-router] wrote phase2={phase2_path}")
                print(f"[StepF-router] wrote edge={edge_path}")
                print(f"[StepF-router] wrote allowlist={allow_path}")
                print(f"[StepF-router] wrote daily={log_router_path}")
                print(f"[StepF-router] wrote summary={summary_router_path}")
            if persist_primary_outputs:
                final_eval = self.evaluate_final_outputs(output_root=out_root, mode=mode, symbol=symbol)
                if int(final_eval.get("return_code", 1)) != 0:
                    raise RuntimeError(f"StepF final artifacts invalid: {final_eval.get('errors', [])}")

            return StepFResult(daily_log_path=str(log_router_path), summary_path=str(summary_router_path), ratio_path=str(ratio_live_path))

    def _candidate_data_roots(self, out_root: Path) -> List[Path]:
        roots: List[Path] = []
        for attr in ("data_dir", "data_root"):
            raw = getattr(self.app_config, attr, None)
            if raw:
                roots.append(Path(str(raw)))
        nested = getattr(self.app_config, "data", None)
        if nested is not None:
            for attr in ("data_dir", "data_root"):
                raw = getattr(nested, attr, None)
                if raw:
                    roots.append(Path(str(raw)))
        roots.append(out_root.parent / "data")
        roots.append(Path("data"))

        out: List[Path] = []
        seen = set()
        for r in roots:
            rr = r.expanduser().resolve()
            key = str(rr)
            if key in seen:
                continue
            seen.add(key)
            out.append(rr)
        return out

    def _log_price_probe(self, symbol: str, source: str, path: Path) -> None:
        print(f"[STEPF_PRICE] symbol={symbol} source={source} path={path} exists={path.exists()}")

    def _load_stepa_price_tech(self, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        base = out_root / "stepA" / mode
        stepa_prices_train = base / f"stepA_prices_train_{symbol}.csv"
        stepa_prices_test = base / f"stepA_prices_test_{symbol}.csv"
        stepa_tech_train = base / f"stepA_tech_train_{symbol}.csv"
        stepa_tech_test = base / f"stepA_tech_test_{symbol}.csv"

        self._log_price_probe(symbol, "stepA", stepa_prices_train)
        self._log_price_probe(symbol, "stepA", stepa_prices_test)

        if stepa_prices_train.exists() and stepa_prices_test.exists():
            p_tr = pd.read_csv(stepa_prices_train)
            p_te = pd.read_csv(stepa_prices_test)
            prices = pd.concat([p_tr, p_te], ignore_index=True)
            source = "stepA"
            if stepa_tech_train.exists() and stepa_tech_test.exists():
                t_tr = pd.read_csv(stepa_tech_train)
                t_te = pd.read_csv(stepa_tech_test)
                tech = pd.concat([t_tr, t_te], ignore_index=True)
            else:
                self._log_price_probe(symbol, "stepA", stepa_tech_train)
                self._log_price_probe(symbol, "stepA", stepa_tech_test)
                tech = pd.DataFrame(columns=["Date"])
        else:
            raw_candidates = [r / f"prices_{symbol}.csv" for r in self._candidate_data_roots(out_root)]
            raw_hit = None
            for raw_path in raw_candidates:
                self._log_price_probe(symbol, "raw", raw_path)
                if raw_path.exists() and raw_hit is None:
                    raw_hit = raw_path
            if raw_hit is None:
                raise FileNotFoundError(
                    "StepF price load failed for "
                    f"{symbol}: stepA_path_train={stepa_prices_train} missing, "
                    f"stepA_path_test={stepa_prices_test} missing, "
                    f"raw_paths={','.join(str(p) for p in raw_candidates)} missing"
                )
            prices = pd.read_csv(raw_hit)
            source = "raw"
            tech = pd.DataFrame(columns=["Date"])

        print(f"[STEPF_PRICE] symbol={symbol} selected_source={source}")
        for df in (prices, tech):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        price_col = next((c for c in ["P_eff", "Close_eff", "price_eff", "close_eff", "Close"] if c in prices.columns), None)
        if price_col is None:
            raise KeyError(f"no usable price column for {symbol}")
        prices["price_exec"] = pd.to_numeric(prices[price_col], errors="coerce")
        merged = prices.merge(tech, on="Date", how="left", suffixes=("", "_tech"))
        merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return merged

    def _load_stepe_logs(self, step_e_root: Path, symbol: str, agents: List[str]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        selected_root = Path(step_e_root).resolve(strict=False)
        audit_root = selected_root.parent.parent / "audit" / selected_root.name
        for agent in agents:
            p = selected_root / f"stepE_daily_log_{agent}_{symbol}.csv"
            summary_path = selected_root / f"stepE_summary_{agent}_{symbol}.json"
            audit_path = audit_root / f"stepE_audit_{agent}_{symbol}.json"
            if not p.exists():
                print(f"[StepF-router] WARN: missing StepE log (skip agent): {p}")
                continue
            quality_payload = {}
            for meta_path in (summary_path, audit_path):
                if meta_path.exists():
                    try:
                        quality_payload.update(json.loads(meta_path.read_text(encoding="utf-8")))
                    except Exception as exc:
                        print(f"[StepF-router] WARN: unreadable StepE meta (skip agent) path={meta_path} exc={type(exc).__name__}:{exc}")
                        quality_payload = {"stepdprime_join_status": "FAIL", "failure_reason": f"meta_unreadable:{meta_path.name}"}
                        break
            if str(quality_payload.get("stepdprime_join_status", "PASS")).upper() != "PASS":
                print(f"[StepF-router] WARN: StepE agent filtered by join_status agent={agent} reason={quality_payload.get('failure_reason', '')}")
                continue
            if str(quality_payload.get("status", "PASS")).upper() == "FAIL" or quality_payload.get("reuse_eligible") is False:
                print(f"[StepF-router] WARN: StepE agent filtered by audit/status agent={agent} reason={quality_payload.get('failure_reason', '')}")
                continue
            df = pd.read_csv(p)
            if "Date" not in df.columns:
                raise KeyError(f"{p} missing Date")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            ratio_col = "ratio" if "ratio" in df.columns else ("pos" if "pos" in df.columns else ("Position" if "Position" in df.columns else None))
            if ratio_col is None:
                raise KeyError(f"{p} missing ratio/pos/Position")
            ret_col = "reward_next" if "reward_next" in df.columns else ("ret" if "ret" in df.columns else None)
            if ret_col is None:
                raise KeyError(f"{p} missing reward_next/ret")
            if "Split" not in df.columns:
                df["Split"] = "test"
            out_df = df[["Date", "Split", ratio_col, ret_col]].copy()
            out_df = out_df.rename(columns={ratio_col: "ratio", ret_col: "stepE_ret_for_stats"})
            out_df["ratio"] = pd.to_numeric(out_df["ratio"], errors="coerce").fillna(0.0)
            out_df["stepE_ret_for_stats"] = pd.to_numeric(out_df["stepE_ret_for_stats"], errors="coerce").fillna(0.0)
            out[agent] = out_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return out

    def _build_phase2_state(self, cfg: StepFRouterConfig, date_range, symbol: str, mode: str, out_root: Path, price_tech: pd.DataFrame) -> pd.DataFrame:
        tech_cols = [c for c in price_tech.columns if c.endswith("_tech")]
        tech = price_tech[["Date"] + tech_cols].copy()
        tech.columns = ["Date"] + [c.replace("_tech", "") for c in tech_cols]
        base_feat = _compute_base_features(price_tech, tech)
        features = [c for c in [
            "Gap", "ATR_norm", "gap_atr", "vol_log_ratio_20", "bnf_score", "ret_1", "ret_5", "ret_20", "range_atr",
            "body_ratio", "lower_wick_ratio", "upper_wick_ratio",
        ] if c in base_feat.columns]
        base_feat = base_feat[["Date"] + features].copy().sort_values("Date").reset_index(drop=True)

        z_past_rows = []
        z_dates = []
        win = int(cfg.past_window_days)
        tgt = int(cfg.past_resample_len)
        target_idx = np.linspace(0, win - 1, tgt)
        for i in range(len(base_feat)):
            if i + 1 < win:
                continue
            chunk = base_feat.iloc[i + 1 - win : i + 1]
            vec = []
            for c in features:
                s = chunk[c].to_numpy(dtype=float)
                src_idx = np.arange(win)
                vec.extend(np.interp(target_idx, src_idx, s).tolist())
            z_past_rows.append(vec)
            z_dates.append(base_feat.loc[i, "Date"])
        z_past = pd.DataFrame({"Date": z_dates})
        applied_copy_defragment = False
        if z_past_rows:
            z_arr = np.asarray(z_past_rows, dtype=float)
            print(f"[STEPF_FRAME] before_feature_concat base_cols={len(z_past.columns)}")
            zp_cols = [f"zp_{i:04d}" for i in range(z_arr.shape[1])]
            print(f"[STEPF_FRAME] new_feature_cols={len(zp_cols)}")
            zp_df = pd.DataFrame(z_arr, columns=zp_cols)
            z_past = pd.concat([z_past, zp_df], axis=1)
            print(f"[STEPF_FRAME] after_feature_concat total_cols={len(z_past.columns)}")
            z_past = z_past.copy()
            applied_copy_defragment = True
        print(f"[STEPF_FRAME] applied_copy_defragment={str(applied_copy_defragment).lower()}")

        X_df = z_past.merge(base_feat, on="Date", how="left")
        if cfg.use_z_pred:
            zpred = self._load_z_pred(out_root, mode, symbol, cfg.z_pred_source)
            X_df = X_df.merge(zpred, on="Date", how="left")
        X_df = X_df.sort_values("Date").reset_index(drop=True)

        tr_s = pd.to_datetime(getattr(date_range, "train_start"))
        tr_e = pd.to_datetime(getattr(date_range, "train_end"))
        te_s = pd.to_datetime(getattr(date_range, "test_start"))
        te_e = pd.to_datetime(getattr(date_range, "test_end"))
        X_df = X_df[(X_df["Date"] >= tr_s) & (X_df["Date"] <= te_e)].copy()
        feat_cols = [c for c in X_df.columns if c != "Date"]
        X_df[feat_cols] = X_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        tr_mask = (X_df["Date"] >= tr_s) & (X_df["Date"] <= tr_e)
        te_mask = (X_df["Date"] >= te_s) & (X_df["Date"] <= te_e)
        x_train = X_df.loc[tr_mask, feat_cols].to_numpy(dtype=float)
        x_test = X_df.loc[te_mask, feat_cols].to_numpy(dtype=float)

        scaler = RobustScaler() if cfg.robust_scaler else StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test) if len(x_test) else np.zeros((0, x_train_s.shape[1]))

        pca_n = max(1, min(int(cfg.pca_n_components), x_train_s.shape[0], x_train_s.shape[1]))
        pca = PCA(n_components=pca_n, random_state=42)
        x_train_p = pca.fit_transform(x_train_s)
        x_test_p = pca.transform(x_test_s) if len(x_test_s) else np.zeros((0, pca_n))

        train_dates = X_df.loc[tr_mask, "Date"].to_numpy()
        test_dates = X_df.loc[te_mask, "Date"].to_numpy()
        cluster_out, diag = self._cluster_phase2(
            cfg=cfg,
            x_train=x_train_p,
            x_test=x_test_p,
        )
        setattr(self, "_last_cluster_diag", dict(diag))

        phase2_train = pd.DataFrame(
            {
                "Date": train_dates,
                "regime_id": cluster_out["train_main"].astype(int),
                "cluster_id_stable": cluster_out["train_stable"].astype(int),
                "cluster_id_raw20": cluster_out["train_raw20"].astype(int),
                "rare_flag_raw20": cluster_out["train_rare_raw20"].astype(int),
                "confidence_stable": cluster_out["train_conf_stable"].astype(float),
                "confidence_raw20": cluster_out["train_conf_raw20"].astype(float),
            }
        )
        phase2_test = pd.DataFrame(
            {
                "Date": test_dates,
                "regime_id": cluster_out["test_main"].astype(int),
                "cluster_id_stable": cluster_out["test_stable"].astype(int),
                "cluster_id_raw20": cluster_out["test_raw20"].astype(int),
                "rare_flag_raw20": cluster_out["test_rare_raw20"].astype(int),
                "confidence_stable": cluster_out["test_conf_stable"].astype(float),
                "confidence_raw20": cluster_out["test_conf_raw20"].astype(float),
            }
        )
        phase2 = pd.concat([phase2_train, phase2_test], ignore_index=True).sort_values("Date").reset_index(drop=True)
        phase2["cluster_id_main"] = phase2["regime_id"].astype(int)
        phase2["confidence"] = phase2["confidence_stable"].astype(float)
        state_source = X_df[["Date"] + [c for c in ["Gap", "ATR_norm", "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "bnf_score"] if c in X_df.columns]].copy()
        rename_map = {
            "Gap": "state_gap",
            "ATR_norm": "state_atr_norm",
            "ret_1": "state_ret_1",
            "ret_5": "state_ret_5",
            "ret_20": "state_ret_20",
            "range_atr": "state_range_atr",
            "body_ratio": "state_body_ratio",
            "bnf_score": "state_bnf_score",
        }
        state_source = state_source.rename(columns=rename_map)
        state_cols = [c for c in state_source.columns if c != "Date"]
        if state_cols:
            state_source["state_context_norm"] = np.sqrt(np.square(state_source[state_cols].astype(float)).sum(axis=1))
            phase2 = phase2.merge(state_source, on="Date", how="left")
        regime_count = int(pd.Series(phase2["regime_id"].astype(int)).nunique()) if len(phase2) else 0
        print(f"[STEPF_CLUSTER] clusterer_type_requested={diag['clusterer_type_requested']}")
        print(f"[STEPF_CLUSTER] clusterer_type_used={diag['clusterer_type_used']}")
        print(f"[STEPF_CLUSTER] fallback_type={diag['fallback_type']}")
        print(f"[STEPF_CLUSTER] fallback_triggered={str(diag['fallback_used']).lower()}")
        print(f"[STEPF_CLUSTER] fallback_reason={diag.get('fallback_reason', '')}")
        print(f"[STEPF_CLUSTER] backend_resolved_name={diag.get('backend_resolved_name', '')}")
        print(f"[STEPF_CLUSTER] backend_predict_methods={','.join(diag.get('backend_predict_methods', []) or [])}")
        print(f"[STEPF_CLUSTER] diagnostic_stage={diag.get('diagnostic_stage', '')}")
        print(f"[STEPF_CLUSTER] raw20_num_clusters_requested={diag.get('raw20_num_clusters_requested', 0)}")
        print(f"[STEPF_CLUSTER] raw20_regime_count={diag.get('raw20_regime_count', 0)}")
        print(f"[STEPF_CLUSTER] k_valid={diag.get('k_valid', 0)}")
        print(f"[STEPF_CLUSTER] k_eff={diag.get('k_eff', 0)}")
        print(f"[STEPF_CLUSTER] k_eff_min={diag.get('k_eff_min', 0)}")
        print(f"[STEPF_CLUSTER] stable_regime_count={diag.get('stable_regime_count', 0)}")
        print(f"[STEPF_CLUSTER] main_cluster_source={diag.get('cluster_main_source', 'stable')}")
        print(f"[STEPF_CLUSTER] rare_flag_raw20_count={diag.get('rare_flag_raw20_count', 0)}")
        print(f"[STEPF_CLUSTER] train_rows={len(train_dates)}")
        print(f"[STEPF_CLUSTER] test_rows={len(test_dates)}")
        print(f"[STEPF_CLUSTER] regime_count={regime_count}")
        return phase2

    def _cluster_phase2(self, cfg: StepFRouterConfig, x_train: np.ndarray, x_test: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        requested = str(getattr(cfg, "clusterer_type", "ticc_raw20_stable") or "ticc_raw20_stable").strip().lower()
        fallback = "none"
        used = requested
        fallback_used = False
        fallback_reason = ""
        backend_resolved_name = ""
        backend_predict_methods: List[str] = []
        backend_methods: List[str] = []
        backend_entrypoint_name = ""
        backend_entrypoint_kind = ""
        backend_api_candidates: List[str] = []
        diagnostic_stage = "cluster_init"
        exception_type = ""
        exception_message = ""
        traceback_path = ""

        if requested in {"ticc_raw20_stable", "ticc"}:
            preflight_diag = self._preflight_ticc_backend()
            backend_resolved_name = str(preflight_diag.get("backend_resolved_name", "") or backend_resolved_name)
            backend_predict_methods = list(preflight_diag.get("backend_predict_methods", []) or backend_predict_methods)
            backend_methods = list(preflight_diag.get("backend_methods", []) or backend_methods)
            backend_entrypoint_name = str(preflight_diag.get("backend_entrypoint_name", "") or backend_entrypoint_name)
            backend_entrypoint_kind = str(preflight_diag.get("backend_entrypoint_kind", "") or backend_entrypoint_kind)
            backend_api_candidates = list(preflight_diag.get("backend_api_candidates", []) or backend_api_candidates)
            if not backend_resolved_name:
                diagnostic_stage = "preflight_failed"
                exception_type = "TICCUnavailableError"
                exception_message = str(preflight_diag.get("backend_resolve_error", "") or "TICC backend unresolved during preflight")
                fallback_reason = self._classify_ticc_failure_reason(TICCUnavailableError(exception_message))
                raise RuntimeError(
                    "StepF requested TICC clustering but preflight failed. "
                    f"fallback_reason={fallback_reason} backend_error={exception_message}"
                )
            else:
                try:
                    diagnostic_stage = "ticc_train_raw20"
                    train_raw20, test_raw20, conf_raw20 = self._run_ticc_clusterer_with_k(
                        cfg=cfg,
                        x_train=x_train,
                        x_test=x_test,
                        num_clusters=int(getattr(cfg, "raw20_num_clusters", 20)),
                    )
                    backend_resolved_name = str(getattr(self, "_last_ticc_backend_name", "") or backend_resolved_name)
                    backend_predict_methods = list(getattr(self, "_last_ticc_backend_predict_methods", []) or backend_predict_methods)
                    backend_methods = list(getattr(self, "_last_ticc_backend_methods", []) or backend_methods)
                    backend_entrypoint_name = str(getattr(self, "_last_ticc_backend_entrypoint_name", "") or backend_entrypoint_name)
                    backend_entrypoint_kind = str(getattr(self, "_last_ticc_backend_entrypoint_kind", "") or backend_entrypoint_kind)
                    backend_api_candidates = list(getattr(self, "_last_ticc_backend_api_candidates", []) or backend_api_candidates)
                    raw_stats = compute_cluster_stats(train_raw20.astype(int).tolist())
                    valid_clusters, rare_clusters = derive_valid_and_rare_clusters(
                        raw_stats,
                        share_min=float(getattr(cfg, "cluster_share_min", 0.01)),
                        mean_run_min=float(getattr(cfg, "cluster_mean_run_min", 3.0)),
                    )
                    k_valid = int(len(valid_clusters))
                    k_eff_min = int(getattr(cfg, "k_eff_min", 12))
                    k_eff = int(max(k_eff_min, k_valid))

                    diagnostic_stage = "ticc_train_stable"
                    train_stable, test_stable, conf_stable = self._run_ticc_clusterer_with_k(
                        cfg=cfg,
                        x_train=x_train,
                        x_test=x_test,
                        num_clusters=k_eff,
                    )
                    backend_resolved_name = str(getattr(self, "_last_ticc_backend_name", "") or backend_resolved_name)
                    backend_predict_methods = list(getattr(self, "_last_ticc_backend_predict_methods", []) or backend_predict_methods)
                    backend_methods = list(getattr(self, "_last_ticc_backend_methods", []) or backend_methods)
                    backend_entrypoint_name = str(getattr(self, "_last_ticc_backend_entrypoint_name", "") or backend_entrypoint_name)
                    backend_entrypoint_kind = str(getattr(self, "_last_ticc_backend_entrypoint_kind", "") or backend_entrypoint_kind)
                    backend_api_candidates = list(getattr(self, "_last_ticc_backend_api_candidates", []) or backend_api_candidates)
                    used = "ticc_raw20_stable"
                    diagnostic_stage = "ticc_success"
                    main_source = str(getattr(cfg, "cluster_main_source", "stable") or "stable").strip().lower()
                    train_main = train_stable if main_source == "stable" else train_raw20
                    test_main = test_stable if main_source == "stable" else test_raw20
                    train_rare = np.isin(train_raw20.astype(int), list(set(rare_clusters))).astype(int)
                    test_rare = np.isin(test_raw20.astype(int), list(set(rare_clusters))).astype(int)
                    if not bool(getattr(cfg, "enable_raw20_monitor", True)):
                        train_rare = np.zeros((x_train.shape[0],), dtype=int)
                        test_rare = np.zeros((x_test.shape[0],), dtype=int)
                    out = {
                        "train_main": train_main.astype(int),
                        "test_main": test_main.astype(int),
                        "train_stable": train_stable.astype(int),
                        "test_stable": test_stable.astype(int),
                        "train_raw20": train_raw20.astype(int),
                        "test_raw20": test_raw20.astype(int),
                        "train_conf_stable": np.ones((x_train.shape[0],), dtype=float),
                        "test_conf_stable": conf_stable.astype(float),
                        "train_conf_raw20": np.ones((x_train.shape[0],), dtype=float),
                        "test_conf_raw20": conf_raw20.astype(float),
                        "train_rare_raw20": train_rare,
                        "test_rare_raw20": test_rare,
                    }
                except Exception as exc:
                    print(f"[STEPF_CLUSTER] ticc_raw20_stable_unavailable={type(exc).__name__}: {exc}")
                    diagnostic_stage = "ticc_exception"
                    exception_type = type(exc).__name__
                    exception_message = str(exc)
                    backend_resolved_name = str(getattr(self, "_last_ticc_backend_name", "") or backend_resolved_name)
                    backend_predict_methods = list(getattr(self, "_last_ticc_backend_predict_methods", []) or backend_predict_methods)
                    backend_methods = list(getattr(self, "_last_ticc_backend_methods", []) or backend_methods)
                    backend_entrypoint_name = str(getattr(self, "_last_ticc_backend_entrypoint_name", "") or backend_entrypoint_name)
                    backend_entrypoint_kind = str(getattr(self, "_last_ticc_backend_entrypoint_kind", "") or backend_entrypoint_kind)
                    backend_api_candidates = list(getattr(self, "_last_ticc_backend_api_candidates", []) or backend_api_candidates)
                    fallback_reason = self._classify_ticc_failure_reason(exc)
                    tb_text = traceback.format_exc()
                    traceback_path = self._write_ticc_traceback(tb_text)
                    print(f"[STEPF_CLUSTER] backend_resolved={backend_resolved_name or 'unknown'}")
                    print(f"[STEPF_CLUSTER] backend_predict_methods={','.join(backend_predict_methods)}")
                    print(f"[STEPF_CLUSTER][ERROR] requested=ticc_raw20_stable failed reason={fallback_reason}")
                    if traceback_path:
                        print(f"[STEPF_CLUSTER][ERROR] traceback_path={traceback_path}")
                    raise RuntimeError(
                        "StepF requested TICC clustering but execution failed. "
                        f"fallback_reason={fallback_reason} exception={exception_type}: {exception_message} "
                        f"traceback_path={traceback_path}"
                    ) from exc
        elif requested == "none":
            diagnostic_stage = "fallback_none_requested"
            out, used, fallback_reason = self._run_cluster_fallback_dual(x_train=x_train, x_test=x_test)
            raw_stats = pd.DataFrame(columns=["cluster_id", "count", "share", "mean_run"])
            valid_clusters = []
            rare_clusters = []
            k_valid = 0
            k_eff_min = int(getattr(cfg, "k_eff_min", 12))
            k_eff = k_eff_min
        else:
            fallback_used = True
            diagnostic_stage = "fallback_unknown_requested"
            out, used, fallback_reason = self._run_cluster_fallback_dual(x_train=x_train, x_test=x_test, fallback_reason="ticc_failure:UnknownClustererType")
            raw_stats = pd.DataFrame(columns=["cluster_id", "count", "share", "mean_run"])
            valid_clusters = []
            rare_clusters = []
            k_valid = 0
            k_eff_min = int(getattr(cfg, "k_eff_min", 12))
            k_eff = k_eff_min

        if fallback_used and not fallback_reason:
            fallback_reason = "fallback_requested"

        diag = {
            "clusterer_type_requested": requested,
            "clusterer_type_used": used,
            "backend_resolved_name": backend_resolved_name,
            "backend_predict_methods": backend_predict_methods,
            "backend_methods": backend_methods,
            "backend_entrypoint_name": backend_entrypoint_name,
            "backend_entrypoint_kind": backend_entrypoint_kind,
            "backend_api_candidates": backend_api_candidates,
            "diagnostic_stage": diagnostic_stage,
            "fallback_type": fallback,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "traceback_path": traceback_path,
            "raw20_num_clusters_requested": int(getattr(cfg, "raw20_num_clusters", 20)),
            "k_raw": int(getattr(cfg, "raw20_num_clusters", 20)),
            "raw20_regime_count": int(len(pd.unique(out["train_raw20"]))),
            "k_valid": int(k_valid),
            "k_eff": int(k_eff),
            "k_eff_min": int(k_eff_min),
            "stable_regime_count": int(len(pd.unique(out["train_stable"]))),
            "cluster_main_source": str(getattr(cfg, "cluster_main_source", "stable") or "stable").strip().lower(),
            "rare_flag_raw20_count": int(np.sum(out["test_rare_raw20"].astype(int)) + np.sum(out["train_rare_raw20"].astype(int))),
            "rare_clusters_raw20": [int(x) for x in sorted(set(rare_clusters))],
            "raw20_cluster_stats": raw_stats.to_dict(orient="records"),
        }
        return out, diag

    def _run_cluster_fallback(self, x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._run_none_clusterer(x_train=x_train, x_test=x_test)

    def _run_cluster_fallback_dual(self, x_train: np.ndarray, x_test: np.ndarray, fallback_reason: str = "") -> Tuple[Dict[str, np.ndarray], str, str]:
        used = "none"
        train, test, conf = self._run_none_clusterer(x_train=x_train, x_test=x_test)

        out = {
            "train_main": train.astype(int),
            "test_main": test.astype(int),
            "train_stable": train.astype(int),
            "test_stable": test.astype(int),
            "train_raw20": train.astype(int),
            "test_raw20": test.astype(int),
            "train_conf_stable": np.ones((x_train.shape[0],), dtype=float),
            "test_conf_stable": conf.astype(float),
            "train_conf_raw20": np.ones((x_train.shape[0],), dtype=float),
            "test_conf_raw20": conf.astype(float),
            "train_rare_raw20": np.zeros((x_train.shape[0],), dtype=int),
            "test_rare_raw20": np.zeros((x_test.shape[0],), dtype=int),
        }
        return out, used, fallback_reason

    def _run_ticc_clusterer_with_k(self, cfg: StepFRouterConfig, x_train: np.ndarray, x_test: np.ndarray, num_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        clusterer = TICCClusterer(
            num_clusters=int(num_clusters),
            window_size=int(cfg.ticc_window_size),
            lambda_parameter=float(cfg.ticc_lambda),
            beta=float(cfg.ticc_beta),
            max_iter=int(cfg.ticc_max_iter),
            threshold=float(cfg.ticc_threshold),
        )
        diag = clusterer.get_diagnostics() if hasattr(clusterer, "get_diagnostics") else {}
        self._last_ticc_backend_name = str(diag.get("backend_resolved_name", "") or "")
        self._last_ticc_backend_predict_methods = list(diag.get("backend_predict_methods", []) or [])
        self._last_ticc_backend_methods = list(diag.get("backend_methods", []) or [])
        self._last_ticc_backend_entrypoint_name = str(diag.get("backend_entrypoint_name", "") or "")
        self._last_ticc_backend_entrypoint_kind = str(diag.get("backend_entrypoint_kind", "") or "")
        self._last_ticc_backend_api_candidates = list(diag.get("backend_api_candidates", []) or [])
        if self._last_ticc_backend_name:
            print(f"[STEPF_CLUSTER] backend_resolved={self._last_ticc_backend_name}")
        if self._last_ticc_backend_predict_methods:
            print(f"[STEPF_CLUSTER] backend_predict_methods={','.join(self._last_ticc_backend_predict_methods)}")
        train_labels = clusterer.fit_predict_train(x_train)
        diag = clusterer.get_diagnostics() if hasattr(clusterer, "get_diagnostics") else {}
        self._last_ticc_backend_name = str(diag.get("backend_resolved_name", "") or self._last_ticc_backend_name)
        self._last_ticc_backend_predict_methods = list(diag.get("backend_predict_methods", []) or self._last_ticc_backend_predict_methods)
        self._last_ticc_backend_methods = list(diag.get("backend_methods", []) or self._last_ticc_backend_methods)
        self._last_ticc_backend_entrypoint_name = str(diag.get("backend_entrypoint_name", "") or getattr(self, "_last_ticc_backend_entrypoint_name", ""))
        self._last_ticc_backend_entrypoint_kind = str(diag.get("backend_entrypoint_kind", "") or getattr(self, "_last_ticc_backend_entrypoint_kind", ""))
        self._last_ticc_backend_api_candidates = list(diag.get("backend_api_candidates", []) or getattr(self, "_last_ticc_backend_api_candidates", []))
        if self._last_ticc_backend_name:
            print(f"[STEPF_CLUSTER] backend_resolved={self._last_ticc_backend_name}")
        if self._last_ticc_backend_predict_methods:
            print(f"[STEPF_CLUSTER] backend_predict_methods={','.join(self._last_ticc_backend_predict_methods)}")
        test_labels, strengths = clusterer.predict_test(x_test)
        return train_labels.astype(int), test_labels.astype(int), strengths.astype(float)

    def _preflight_ticc_backend(self) -> Dict[str, object]:
        clusterer = TICCClusterer(
            num_clusters=2,
            window_size=2,
            lambda_parameter=0.11,
            beta=600.0,
            max_iter=1,
            threshold=2e-5,
        )
        diag = clusterer.get_diagnostics() if hasattr(clusterer, "get_diagnostics") else {}
        backend_name = str(diag.get("backend_resolved_name", "") or "")
        backend_error = str(diag.get("backend_resolve_error", "") or "")
        if not backend_name and not backend_error and not hasattr(clusterer, "get_diagnostics"):
            backend_name = type(clusterer).__name__
        backend_methods = list(diag.get("backend_methods", []) or [])
        backend_predict_methods = list(diag.get("backend_predict_methods", []) or [])
        backend_entrypoint_name = str(diag.get("backend_entrypoint_name", "") or "")
        backend_entrypoint_kind = str(diag.get("backend_entrypoint_kind", "") or "")
        backend_api_candidates = list(diag.get("backend_api_candidates", []) or [])
        self._last_ticc_backend_name = backend_name
        self._last_ticc_backend_methods = backend_methods
        self._last_ticc_backend_predict_methods = backend_predict_methods
        self._last_ticc_backend_entrypoint_name = backend_entrypoint_name
        self._last_ticc_backend_entrypoint_kind = backend_entrypoint_kind
        self._last_ticc_backend_api_candidates = backend_api_candidates
        return {
            "backend_resolved_name": backend_name,
            "backend_resolve_error": backend_error,
            "backend_methods": backend_methods,
            "backend_predict_methods": backend_predict_methods,
            "backend_entrypoint_name": backend_entrypoint_name,
            "backend_entrypoint_kind": backend_entrypoint_kind,
            "backend_api_candidates": backend_api_candidates,
        }

    @staticmethod
    def _classify_ticc_failure_reason(exc: Exception) -> str:
        name = type(exc).__name__
        message = str(exc).lower()
        stage = "unknown"
        if "stage=" in message:
            try:
                stage = message.split("stage=", 1)[1].split()[0].split(",")[0]
            except Exception:
                stage = "unknown"
        if isinstance(exc, TICCUnavailableError):
            if "placeholder" in message:
                return f"ticc_failure:PlaceholderArtifact:stage={stage}"
            if "not_wired" in message or "not yet wired" in message:
                return f"ticc_failure:NotWired:stage={stage}"
            if "missing_method" in message:
                return f"ticc_failure:PredictMethodMissing:stage={stage}"
            if "api mismatch" in message or "entrypoint" in message:
                return f"ticc_failure:BackendAPIMismatch:stage={stage}"
            if "import fast_ticc failed" in message or "ticc_solver" in message:
                return f"ticc_failure:BackendUnavailable:stage={stage}"
            return f"ticc_failure:TICCUnavailableError:stage={stage}"
        if name in {"ImportError", "ModuleNotFoundError"}:
            return f"ticc_failure:ImportError:stage={stage}"
        if name == "ValueError":
            return f"ticc_failure:ValueError:stage={stage}"
        return f"ticc_failure:{name}:stage={stage}"

    @staticmethod
    def _write_ticc_traceback(tb_text: str) -> str:
        try:
            root = Path(tempfile.gettempdir()) / "apex_traderai" / "stepf"
            root.mkdir(parents=True, exist_ok=True)
            path = root / "stepf_ticc_failure_traceback.log"
            path.write_text(tb_text, encoding="utf-8")
            return str(path)
        except Exception:
            return ""

    def _run_ticc_clusterer(self, cfg: StepFRouterConfig, x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        clusterer = TICCClusterer(
            num_clusters=int(cfg.ticc_num_clusters),
            window_size=int(cfg.ticc_window_size),
            lambda_parameter=float(cfg.ticc_lambda),
            beta=float(cfg.ticc_beta),
            max_iter=int(cfg.ticc_max_iter),
            threshold=float(cfg.ticc_threshold),
        )
        train_labels = clusterer.fit_predict_train(x_train)
        test_labels, strengths = clusterer.predict_test(x_test)
        return train_labels.astype(int), test_labels.astype(int), strengths.astype(float)

    @staticmethod
    def _run_none_clusterer(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((x_train.shape[0],), dtype=int),
            np.zeros((x_test.shape[0],), dtype=int),
            np.ones((x_test.shape[0],), dtype=float),
        )

    def _load_z_pred(self, out_root: Path, mode: str, symbol: str, profile: str) -> pd.DataFrame:
        stepd_root, warnings, _legacy_read = resolve_stepdprime_root(out_root, mode)
        for warning in warnings:
            print(f"[StepF][STEPDPRIME_WARN] {warning}")
        p = stepd_root / "embeddings" / f"stepDprime_{profile}_{symbol}_embeddings_all.csv"
        if not p.exists():
            return pd.DataFrame({"Date": []})
        df = pd.read_csv(p)
        if "Date" not in df.columns:
            return pd.DataFrame({"Date": []})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        return df[["Date"] + emb_cols].copy()

    def _build_regime_edge_table(self, merged: pd.DataFrame, agents: List[str], cfg: StepFRouterConfig) -> pd.DataFrame:
        rows = []
        train_df = merged[merged["Split"].astype(str).str.lower() == "train"].copy()
        global_stats: Dict[str, Dict[str, float]] = {}
        for agent in agents:
            sub_g = pd.to_numeric(train_df.get(f"ret_{agent}"), errors="coerce").dropna().astype(float)
            if len(sub_g) == 0:
                global_stats[agent] = {"n_global": 0, "EV_global": float("nan"), "IR_global": float("nan")}
                continue
            ev_g = float(sub_g.mean())
            sd_g = float(sub_g.std(ddof=0))
            global_stats[agent] = {
                "n_global": int(len(sub_g)),
                "EV_global": ev_g,
                "IR_global": float(ev_g / (sd_g + float(cfg.eps_ir))),
            }

        for rid in sorted(train_df["regime_id"].dropna().astype(int).unique().tolist()):
            for agent in agents:
                col = f"ret_{agent}"
                sub = train_df[train_df["regime_id"].astype(int) == rid][col].astype(float)
                if len(sub) == 0:
                    continue
                ev = float(sub.mean())
                sd = float(sub.std(ddof=0))
                ir = float(ev / (sd + float(cfg.eps_ir)))
                p_win = float((sub > 0).mean())
                q05 = float(sub.quantile(0.05))
                q95 = float(sub.quantile(0.95))
                eq = np.cumprod(1.0 + sub.to_numpy(dtype=float))
                peak = np.maximum.accumulate(eq)
                dd_proxy = float(np.min(eq / np.where(peak == 0, 1.0, peak) - 1.0))
                g = global_stats.get(agent, {})
                n_days = int(len(sub))
                w = float(n_days) / float(n_days + max(1, int(cfg.shrink_k)))
                ir_g = float(g.get("IR_global", np.nan))
                ev_g = float(g.get("EV_global", np.nan))
                ir_shrink = float(w * ir + (1.0 - w) * ir_g) if np.isfinite(ir_g) else ir
                ev_shrink = float(w * ev + (1.0 - w) * ev_g) if np.isfinite(ev_g) else ev
                rows.append(
                    {
                        "regime_id": int(rid),
                        "agent": agent,
                        "n_days": n_days,
                        "EV": ev,
                        "IR": ir,
                        "EV_shrink": ev_shrink,
                        "IR_shrink": ir_shrink,
                        "w_shrink": w,
                        "n_global": int(g.get("n_global", 0)),
                        "EV_global": ev_g,
                        "IR_global": ir_g,
                        "p_win": p_win,
                        "q05": q05,
                        "q95": q95,
                        "dd_proxy": dd_proxy,
                    }
                )
        return pd.DataFrame(rows)

    def _build_allowlist(self, edge_table: pd.DataFrame, agents: List[str], safe_set: List[str], cfg: StepFRouterConfig) -> pd.DataFrame:
        topk_global = self._topk_global_agents(edge_table=edge_table, topk=max(1, int(cfg.topK)))
        if edge_table.empty:
            base = list(dict.fromkeys(safe_set + topk_global))
            return pd.DataFrame([{"regime_id": -1, "allowed_agents": "|".join(base)}])
        out_rows = [{"regime_id": -1, "allowed_agents": "|".join(list(dict.fromkeys(safe_set + topk_global)))}]
        for rid, df_r in edge_table.groupby("regime_id"):
            rid = int(rid)
            if rid == -1:
                continue
            if int(df_r["n_days"].max()) < int(cfg.min_samples_regime):
                allowed = list(dict.fromkeys(safe_set + topk_global))
            else:
                cands = df_r.copy()
                if cfg.topk_filter_ev_positive:
                    ev_col = "EV_shrink" if "EV_shrink" in cands.columns else "EV"
                    pos = cands[cands[ev_col] > 0].copy()
                    if not pos.empty:
                        cands = pos
                ir_col = "IR_shrink" if "IR_shrink" in cands.columns else "IR"
                ev_col = "EV_shrink" if "EV_shrink" in cands.columns else "EV"
                cands = cands.sort_values([ir_col, ev_col, "dd_proxy", "n_days"], ascending=[False, False, True, False])
                picked = cands["agent"].tolist()[: int(cfg.topK)]
                allowed = list(dict.fromkeys(safe_set + picked))
            out_rows.append({"regime_id": rid, "allowed_agents": "|".join(allowed)})
        return pd.DataFrame(out_rows)

    def _topk_global_agents(self, edge_table: pd.DataFrame, topk: int) -> List[str]:
        if edge_table.empty:
            return []
        cols = [c for c in ["IR_global", "EV_global", "n_global"] if c in edge_table.columns]
        if len(cols) < 3:
            return []
        gdf = edge_table[["agent", "IR_global", "EV_global", "n_global"]].drop_duplicates(subset=["agent"])
        gdf = gdf.sort_values(["IR_global", "EV_global", "n_global"], ascending=[False, False, False])
        return gdf["agent"].tolist()[: max(1, int(topk))]

    def _stabilize_safe_set(self, safe_set: List[str], edge_table: pd.DataFrame, agents: List[str]) -> List[str]:
        base = [a for a in safe_set if a in agents]
        if not base:
            base = self._topk_global_agents(edge_table=edge_table, topk=min(3, len(agents)))
        if not base:
            base = agents[: min(3, len(agents))]
        return list(dict.fromkeys(base))

    def _build_context_profiles(self, merged: pd.DataFrame, agents: List[str]) -> Dict[Tuple[int, int, str], Dict[str, float]]:
        train_df = merged[merged["Split"].astype(str).str.lower() == "train"].copy()
        context_cols = [c for c in merged.columns if c.startswith("state_")]
        profiles: Dict[Tuple[int, int, str], Dict[str, float]] = {}
        if train_df.empty or not context_cols:
            return profiles
        for rid, cid, agent in train_df[["regime_id", "cluster_id_stable"]].dropna().assign(agent_key=1).merge(
            pd.DataFrame({"agent": agents, "agent_key": 1}), on="agent_key", how="left"
        )[["regime_id", "cluster_id_stable", "agent"]].itertuples(index=False):
            sub = train_df[(train_df["regime_id"].astype(int) == int(rid)) & (train_df["cluster_id_stable"].astype(int) == int(cid))].copy()
            if sub.empty:
                continue
            profiles[(int(rid), int(cid), str(agent))] = {
                c: float(pd.to_numeric(sub[c], errors="coerce").fillna(0.0).mean()) for c in context_cols
            }
        return profiles

    def _build_policy_compare_audit(self, daily: pd.DataFrame, merged: pd.DataFrame, agents: List[str]) -> Dict[str, object]:
        train_df = merged[merged["Split"].astype(str).str.lower() == "train"].copy()
        fixed_best = ""
        if not train_df.empty:
            means = {a: float(pd.to_numeric(train_df.get(f"ret_{a}"), errors="coerce").mean()) for a in agents}
            fixed_best = max(means, key=means.get)
        oracle_reward = []
        fixed_reward = []
        for row in merged.itertuples(index=False):
            agent_rewards = [float(getattr(row, f"ret_{a}", np.nan)) for a in agents if np.isfinite(float(getattr(row, f"ret_{a}", np.nan)))]
            oracle_reward.append(max(agent_rewards) if agent_rewards else 0.0)
            fixed_reward.append(float(getattr(row, f"ret_{fixed_best}", 0.0) or 0.0) if fixed_best else 0.0)
        selected_series = daily.get("selected_expert")
        expert_frequency = selected_series.value_counts(dropna=False).to_dict() if selected_series is not None else {}
        branch_stats = (
            daily.groupby(["regime_id", "cluster_id_stable"]).agg(
                rows=("Date", "size"),
                ratio_mean=("ratio", "mean"),
                reward_mean=("reward", "mean"),
            ).reset_index().to_dict(orient="records")
            if {"regime_id", "cluster_id_stable", "ratio", "reward"}.issubset(daily.columns)
            else []
        )
        return {
            "summary": {
                "fixed_best_agent": fixed_best,
                "current_policy_test_equity_end": float(daily.loc[daily["Split"].astype(str).str.lower() == "test", "equity"].iloc[-1]) if not daily.empty else 1.0,
                "fixed_best_test_equity_end": float(np.cumprod(1.0 + np.asarray(fixed_reward, dtype=float))[-1]) if fixed_reward else 1.0,
                "oracle_test_equity_end": float(np.cumprod(1.0 + np.asarray(oracle_reward, dtype=float))[-1]) if oracle_reward else 1.0,
            },
            "expert_frequency": expert_frequency,
            "branch_stats": branch_stats,
            "all_zero_input_detected": bool(any(c.startswith("state_") and np.all(np.abs(pd.to_numeric(merged[c], errors="coerce").fillna(0.0).to_numpy()) < 1e-12) for c in merged.columns)),
            "constant_output_detected": bool(float(pd.to_numeric(daily["ratio"], errors="coerce").std(ddof=0)) < 1e-9) if "ratio" in daily.columns and not daily.empty else True,
        }

    def _run_router_sim(self, merged: pd.DataFrame, agents: List[str], edge_table: pd.DataFrame, allowlist: pd.DataFrame, safe_set: List[str], cfg: StepFRouterConfig, context_profiles: Dict[Tuple[int, int, str], Dict[str, float]], device_name: str) -> pd.DataFrame:
        allow_map = {int(r.regime_id): [a for a in str(r.allowed_agents).split("|") if a] for r in allowlist.itertuples(index=False)}
        ir_map: Dict[Tuple[int, str], float] = {}
        score_col = "IR_shrink" if "IR_shrink" in edge_table.columns else "IR"
        for r in edge_table.itertuples(index=False):
            ir_map[(int(r.regime_id), str(r.agent))] = float(getattr(r, score_col, np.nan))

        w_prev = {a: 0.0 for a in agents}
        out = []
        ratio_prev = 0.0
        eq = 1.0
        peak_eq = 1.0
        reward_mode = str(getattr(cfg, "reward_mode", "legacy") or "legacy").strip().lower()
        compute_device = torch.device(device_name if str(device_name).startswith("cuda") and torch.cuda.is_available() else "cpu")
        context_cols = [c for c in merged.columns if c.startswith("state_")]
        for row in merged.itertuples(index=False):
            rid = int(getattr(row, "regime_id"))
            cid = int(getattr(row, "cluster_id_stable", getattr(row, "regime_id", 0)) or 0)
            allowed = allow_map.get(rid, safe_set if rid == -1 else agents)
            if rid == -1:
                allowed = list(safe_set)
            allowed = [a for a in allowed if a in agents]
            if not allowed:
                allowed = list(safe_set or agents)

            scores = np.array([ir_map.get((rid, a), np.nan) for a in allowed], dtype=float)
            context_bonus = []
            for a in allowed:
                profile = context_profiles.get((rid, cid, a), {})
                if profile and context_cols:
                    deltas = [abs(float(getattr(row, c, 0.0) or 0.0) - float(profile.get(c, 0.0))) for c in context_cols]
                    context_bonus.append(float(np.exp(-np.mean(deltas))))
                else:
                    context_bonus.append(0.0)
            scores = scores + np.asarray(context_bonus, dtype=float)
            if np.any(np.isnan(scores)):
                w_raw_allowed = np.ones(len(allowed), dtype=float) / max(1, len(allowed))
            else:
                z = torch.tensor(float(cfg.softmax_beta) * scores, dtype=torch.float32, device=compute_device)
                z = z - torch.max(z)
                e = torch.softmax(z, dim=0)
                w_raw_allowed = e.detach().cpu().numpy().astype(float)

            w_raw_full = {a: 0.0 for a in agents}
            for i, a in enumerate(allowed):
                w_raw_full[a] = float(w_raw_allowed[i])

            w_full = {}
            for a in agents:
                w_full[a] = float(cfg.ema_alpha) * w_raw_full[a] + (1.0 - float(cfg.ema_alpha)) * w_prev[a]
                if a not in allowed:
                    w_full[a] = 0.0
            s = sum(w_full.values())
            if s <= 0:
                for a in allowed:
                    w_full[a] = 1.0 / len(allowed)
            else:
                for a in agents:
                    w_full[a] /= s

            ratio = 0.0
            for a in agents:
                ratio += w_full[a] * float(getattr(row, f"ratio_{a}", 0.0) or 0.0)
            ratio = float(np.clip(ratio, -float(cfg.pos_limit), float(cfg.pos_limit)))

            cost = float(cfg.trade_cost_bps) * 1e-4 * abs(ratio - ratio_prev)
            turnover = float(abs(ratio - ratio_prev))
            penalty = float(cfg.pos_l2_lambda) * (ratio ** 2)
            switch_cost = turnover
            churn_cost = float(np.sum([abs(w_full[a] - w_prev[a]) for a in agents]))
            pos_plus = max(ratio, 0.0)
            pos_minus = max(-ratio, 0.0)
            r_soxl = float(getattr(row, "r_soxl", 0.0) or 0.0)
            r_soxs = float(getattr(row, "r_soxs", 0.0) or 0.0)
            ret_selected = pos_plus * r_soxl + pos_minus * r_soxs
            ret_best_expert = max([float(getattr(row, f"ret_{a}", np.nan)) for a in agents if np.isfinite(float(getattr(row, f"ret_{a}", np.nan)))] or [ret_selected])

            if reward_mode == "profit_basic":
                reward = ret_selected - cost - penalty - float(cfg.lambda_switch) * switch_cost - float(cfg.lambda_churn) * churn_cost
            elif reward_mode == "profit_regret":
                regret = max(0.0, ret_best_expert - ret_selected)
                reward = ret_selected - cost - penalty - float(cfg.lambda_regret) * regret - float(cfg.lambda_switch) * switch_cost
            elif reward_mode == "profit_light_risk":
                eq_tmp = eq * (1.0 + ret_selected - cost - penalty)
                peak_next = max(peak_eq, eq_tmp)
                dd_penalty = max(0.0, 1.0 - eq_tmp / max(1e-12, peak_next))
                reward = ret_selected - cost - penalty - float(cfg.lambda_switch) * switch_cost - float(cfg.lambda_dd) * dd_penalty
            else:
                reward = ret_selected - cost - penalty

            split = str(getattr(row, "Split", "test"))
            selected_expert = max(w_full, key=w_full.get) if w_full else ""
            input_feature_summary = {
                "regime_id": rid,
                "cluster_id_stable": cid,
                "confidence_stable": float(getattr(row, "confidence_stable", 0.0) or 0.0),
                "state_context_norm": float(getattr(row, "state_context_norm", 0.0) or 0.0),
            }
            input_nonzero_feature_count = int(sum(abs(float(v or 0.0)) > 1e-12 for v in input_feature_summary.values()))
            rec = {
                "Date": getattr(row, "Date"),
                "Split": split,
                "regime_id": rid,
                "cluster_id_stable": cid,
                "ratio": ratio,
                "final_ratio": ratio,
                "reward": reward,
                "cost": cost,
                "turnover": turnover,
                "ret_selected": ret_selected,
                "ret_best_expert": ret_best_expert,
                "switch_cost": switch_cost,
                "churn_cost": churn_cost,
                "reward_mode": reward_mode,
                "selected_expert": selected_expert,
                "selected_expert_weight": float(w_full.get(selected_expert, 0.0)),
                "mixture_weights_json": json.dumps({a: float(w_full[a]) for a in agents}, ensure_ascii=False),
                "allowed_agents": "|".join(allowed),
                "r_soxl": r_soxl,
                "r_soxs": r_soxs,
                "source_device": str(compute_device),
                "input_feature_count": int(len(input_feature_summary)),
                "input_nonzero_feature_count": input_nonzero_feature_count,
                "input_has_regime": 1,
                "input_has_cluster": 1,
                "input_has_pred_block": 0,
                "input_has_past_block": 1,
                "input_feature_summary": json.dumps(input_feature_summary, ensure_ascii=False),
            }
            for a in agents:
                rec[f"w_{a}"] = w_full[a]
            out.append(rec)
            w_prev = w_full
            ratio_prev = ratio
        out_df = pd.DataFrame(out)
        if out_df.empty:
            return out_df
        out_df["ret"] = pd.to_numeric(out_df["reward"], errors="coerce").shift(1).fillna(0.0)
        out_df["realized_ret"] = out_df["ret"]
        out_df["realized_ret_next"] = pd.to_numeric(out_df["reward"], errors="coerce").fillna(0.0)
        out_df["equity"] = (1.0 + out_df["ret"].astype(float)).cumprod()
        return out_df

    def _assign_split_by_date(self, dates: pd.Series, date_range) -> pd.Series:
        d = pd.to_datetime(dates, errors="coerce").dt.normalize()
        tr_s = pd.to_datetime(getattr(date_range, "train_start"), errors="coerce")
        tr_e = pd.to_datetime(getattr(date_range, "train_end"), errors="coerce")
        te_s = pd.to_datetime(getattr(date_range, "test_start"), errors="coerce")
        te_e = pd.to_datetime(getattr(date_range, "test_end"), errors="coerce")
        split = pd.Series(np.full(len(d), "other", dtype=object), index=dates.index)
        split[(d >= tr_s) & (d <= tr_e)] = "train"
        split[(d >= te_s) & (d <= te_e)] = "test"
        return split
