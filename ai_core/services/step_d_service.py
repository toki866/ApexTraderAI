from __future__ import annotations

"""ai_core/services/step_d_service.py

StepDService (Envelope + Events + Daily envelope features)

This service generates:
- output/stepD/<mode>/stepD_events_{agent}_{symbol}.csv
- output/stepD/<mode>/stepD_envelope_daily_{agent}_{symbol}.csv  <-- required by StepE

Inputs (mode-first):
- StepC: output/stepC/<mode>/stepC_pred_time_all_<SYMBOL>.csv   (preferred)
- StepB: output/stepB/<mode>/stepB_pred_time_all_<SYMBOL>.csv   (fallback)
- StepA: output/stepA/<mode>/stepA_prices_train/test_<SYMBOL>.csv (timeline fallback)

Multi-horizon support:
- If prediction columns like *_h01/_h05/_h10/_h20 exist, StepD writes suffixed daily features:
  DeltaP_pct_h01, Theta_norm_h01, ... (11 cols per horizon)
- Also writes unsuffixed columns (backward compatibility) using a preferred horizon:
  - date_range.env_horizon_days if set and exists
  - otherwise h20 if available, else the largest available horizon

Leakage note:
Daily features are computed CAUSALLY using only a rolling window (rbw) ending at each day.
No realized future prices are used. Only prediction series values are used.
"""

from dataclasses import dataclass
from pathlib import Path

from ai_core.utils.paths import resolve_repo_path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re


@dataclass
class StepDEvent:
    direction: str  # "UP" or "DOWN"
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration: int
    delta_p: float
    delta_p_pct: float
    theta_norm: float
    top_pct: float
    bottom_pct: float
    d_norm: float
    d_abs: float
    l_abs: float
    theta_deg: float
    top_abs: float
    bottom_abs: float


class StepDService:
    def __init__(self, app_config: Any, symbol: str, date_range: Any):
        self.app_config = app_config
        self.symbol = str(symbol)
        self.date_range = date_range

    def run(self) -> Dict[str, Any]:
        output_root = self._get_output_root()

        hint = (
            getattr(self.date_range, "stepD_mode", None)
            or getattr(self.date_range, "stepC_mode", None)
            or getattr(self.date_range, "stepB_mode", None)
            or getattr(self.date_range, "mamba_mode", None)
            or getattr(self.date_range, "op_mode", None)
            or getattr(self.date_range, "mode", None)
            or "sim"
        )
        mode = self._normalize_mode(hint)

        stepd_dir = output_root / "stepD" / mode
        stepd_dir.mkdir(parents=True, exist_ok=True)

        pred_path = self._resolve_pred_time_all_path(output_root, step="stepC", mode=mode)
        used_source = "stepC"
        if pred_path is None or not pred_path.exists():
            pred_path = self._resolve_stepb_pred_close_path(output_root, mode=mode)
            if pred_path is None or not pred_path.exists():
                pred_path = self._resolve_pred_time_all_path(output_root, step="stepB", mode=mode)
            used_source = "stepB"

        df = pd.DataFrame()
        if pred_path is not None and pred_path.exists():
            try:
                df = pd.read_csv(pred_path)
            except Exception:
                df = pd.DataFrame()

        df = self._normalize_date_column(df)
        start_ts, end_ts = self._get_date_bounds(self.date_range, df)

        if len(df) > 0:
            df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].reset_index(drop=True)

        if len(df) == 0:
            df_price = self._load_stepa_prices_timeline(output_root, mode=mode)
            df_price = self._normalize_date_column(df_price)
            df = df_price[(df_price["Date"] >= start_ts) & (df_price["Date"] <= end_ts)].reset_index(drop=True)
            used_source = "stepA_prices" if len(df) > 0 else used_source

        rbw_default = self._get_env_rbw_default20()

        agents = self._detect_agents(df)
        if "xsr" not in agents:
            agents = ["xsr"] + agents

        print(f"[StepD] mode={mode} source={used_source} rows={len(df)} cols={len(df.columns)} rbw={rbw_default}")

        results: Dict[str, Any] = {
            "symbol": self.symbol,
            "mode": mode,
            "source": used_source,
            "output_dir": str(stepd_dir),
            "agents": agents,
            "rbw": int(rbw_default),
            "files": {},
        }

        env_h = getattr(self.date_range, "env_horizon_days", None)
        env_horizons_req = self._parse_horizon_list(
            getattr(self.date_range, "env_horizons", None)
            or getattr(self.app_config, "env_horizons", None)
            or env_h
        )
        try:
            env_h = int(env_h) if env_h is not None else None
        except Exception:
            env_h = None

        for agent in agents:
            pred_cols_map = self._pick_pred_cols(df, agent)  # {h:int -> colname}

            # Skip agents that have no real prediction values (e.g. disabled models left as all-NaN columns)
            if pred_cols_map:
                nn_total = 0
                for _c in pred_cols_map.values():
                    nn_total += int(pd.to_numeric(df[_c], errors="coerce").notna().sum())
                if nn_total < 20:
                    print(f"[StepD] skip agent={agent} not_enough_nonnull_preds nn_total={nn_total} cols={list(pred_cols_map.values())}")
                    pred_cols_map = {}

            if len(df) == 0 or not pred_cols_map:
                print(f"[StepD] skip agent={agent} (no usable prediction columns)")
                continue

            else:
                daily_df = pd.DataFrame({"Date": df["Date"].dt.strftime("%Y-%m-%d")})
                horizons_available = sorted(pred_cols_map.keys())
                horizons_sorted = [h for h in (env_horizons_req or horizons_available) if h in pred_cols_map]
                if not horizons_sorted:
                    horizons_sorted = horizons_available

                preferred_h = self._pick_base_horizon(env_h=env_h, horizons=horizons_available)
                used_cols_list: List[str] = []

                for h_days in horizons_sorted:
                    col = pred_cols_map[h_days]
                    used_cols_list.append(col)
                    series = pd.to_numeric(df[col], errors="coerce").astype(float).ffill().fillna(0.0)

                    rbw_agent = self._get_env_rbw_for_agent(agent, rbw_default)
                    daily_h = self._envelope_daily_from_series(series=series, dates=df["Date"], rbw=rbw_agent)

                    suffix = f"_h{int(h_days):02d}"
                    rename_map = {c: f"{c}{suffix}" for c in daily_h.columns if c != "Date"}
                    daily_h = daily_h.rename(columns=rename_map)

                    daily_df = daily_df.merge(daily_h, on="Date", how="left")

                pref_suffix = f"_h{int(preferred_h):02d}"
                for base_col in self._daily_feature_cols():
                    c_suf = f"{base_col}{pref_suffix}"
                    if c_suf in daily_df.columns:
                        daily_df[base_col] = daily_df[c_suf]

                for c in daily_df.columns:
                    if c == "Date":
                        continue
                    daily_df[c] = pd.to_numeric(daily_df[c], errors="coerce").fillna(0.0)

                try:
                    events_df = self._events_from_daily(daily_df, suffix="")
                except Exception:
                    events_df = pd.DataFrame([])

                used_cols = ",".join(used_cols_list)

            events_path = stepd_dir / f"stepD_events_{agent}_{self.symbol}.csv"
            daily_path = stepd_dir / f"stepD_envelope_daily_{agent}_{self.symbol}.csv"
            events_df.to_csv(events_path, index=False, encoding="utf-8-sig")
            daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
            # Optional: write 1-row-per-day snapshots for the test dates (if StepA daily exists)
            try:
                self._write_daily_snapshots_from_stepA(daily_df=daily_df, symbol=self.symbol, agent=agent, mode=mode, step_d_dir=stepd_dir)
            except Exception as e:
                print(f"[StepD] daily snapshots skipped: {e}")

            results["files"][agent] = {
                "events_csv": str(events_path),
                "daily_csv": str(daily_path),
                "used_cols": used_cols,
                "num_events": int(len(events_df)),
                "num_days": int(len(daily_df)),
            }
            print(f"[StepD] wrote agent={agent} daily={daily_path.name} used_cols={used_cols} days={len(daily_df)}")

        return results

    # ---------- Path / mode ----------
    def _normalize_mode(self, mode: str) -> str:
        m = (str(mode or "")).strip().lower()
        if m in {"live", "ops", "op", "prod", "production", "real"}:
            return "live"
        return "sim"

    def _get_output_root(self) -> Path:
        try:
            data = getattr(self.app_config, "data")
            out = getattr(data, "output_root", None)
            if out:
                return Path(out)
        except Exception:
            pass
        for attr in ("output_root", "output_dir", "out_dir"):
            try:
                out = getattr(self.app_config, attr)
                if out:
                    return Path(out)
            except Exception:
                continue
        return resolve_repo_path("output")

    def _resolve_pred_time_all_path(self, output_root: Path, step: str, mode: str) -> Optional[Path]:
        step = str(step).strip()
        mode = self._normalize_mode(mode)
        if step.lower() == "stepc":
            name = f"stepC_pred_time_all_{self.symbol}.csv"
        elif step.lower() == "stepb":
            name = f"stepB_pred_time_all_{self.symbol}.csv"
        else:
            return None

        cand = [
            output_root / step / mode / name,
            output_root / step / "ops" / name,
            output_root / step / name,
            output_root / name,
        ]
        for p in cand:
            if p.exists():
                return p
        step_dir = output_root / step
        if step_dir.exists():
            hits = list(step_dir.glob(f"**/{name}"))
            if hits:
                return hits[0]
        return None

    def _resolve_stepb_pred_close_path(self, output_root: Path, mode: str) -> Optional[Path]:
        mode = self._normalize_mode(mode)
        cand = [
            output_root / "stepB" / mode / f"stepB_pred_close_mamba_{self.symbol}.csv",
            output_root / "stepB" / mode / f"stepB_pred_close_mamba_periodic_{self.symbol}.csv",
            output_root / "stepB" / f"stepB_pred_close_mamba_{self.symbol}.csv",
            output_root / "stepB" / f"stepB_pred_close_mamba_periodic_{self.symbol}.csv",
        ]
        for p in cand:
            if p.exists():
                return p
        stepb_dir = output_root / "stepB"
        if stepb_dir.exists():
            hits = sorted(stepb_dir.glob(f"**/stepB_pred_close_*_{self.symbol}.csv"))
            if hits:
                return hits[0]
        return None

    def _load_stepa_prices_timeline(self, output_root: Path, mode: str) -> pd.DataFrame:
        stepa_dir = output_root / "stepA"
        mode = self._normalize_mode(mode)

        p_tr = stepa_dir / mode / f"stepA_prices_train_{self.symbol}.csv"
        p_te = stepa_dir / mode / f"stepA_prices_test_{self.symbol}.csv"
        if p_tr.exists() and p_te.exists():
            dtr = pd.read_csv(p_tr)
            dte = pd.read_csv(p_te)
            return pd.concat([dtr, dte], ignore_index=True)

        cand = [
            stepa_dir / mode / f"stepA_prices_{self.symbol}.csv",
            stepa_dir / f"stepA_prices_{self.symbol}.csv",
            output_root / f"stepA_prices_{self.symbol}.csv",
        ]
        for p in cand:
            if p.exists():
                return pd.read_csv(p)
        return pd.DataFrame({"Date": []})

    # ---------- Date helpers ----------
    def _normalize_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame({"Date": pd.to_datetime([], errors="coerce")})
        if "Date" not in df.columns:
            if "date" in df.columns:
                df = df.rename(columns={"date": "Date"})
            elif df.columns.size > 0:
                df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df

    def _get_date_bounds(self, date_range: Any, df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        def _pick(*names: str):
            for n in names:
                v = getattr(date_range, n, None)
                if v is not None:
                    return v
            return None

        start = _pick("start", "train_start", "date_from")
        end = _pick("end", "test_end", "date_to")
        if start is None:
            start = df["Date"].min() if df is not None and len(df) else pd.Timestamp("2014-01-02")
        if end is None:
            end = df["Date"].max() if df is not None and len(df) else pd.Timestamp("2099-12-31")
        return pd.to_datetime(start), pd.to_datetime(end)

    # ---------- Env params ----------
    def _get_env_rbw_default20(self) -> int:
        rbw = None
        try:
            rbw = getattr(self.app_config, "env_rbw", None)
        except Exception:
            rbw = None
        if rbw is None:
            try:
                env = getattr(self.app_config, "env", None)
                rbw = getattr(env, "rbw", None) if env is not None else None
            except Exception:
                rbw = None
        try:
            rbw = int(rbw) if rbw is not None else 20
        except Exception:
            rbw = 20
        return rbw if rbw > 1 else 20

    def _get_env_rbw_for_agent(self, agent: str, default_rbw: int) -> int:
        a = (agent or "").strip().lower()
        if not a:
            return int(default_rbw)
        key = f"env_rbw_{a}"
        try:
            v = getattr(self.app_config, key, None)
            if v is not None:
                v = int(v)
                return v if v > 1 else int(default_rbw)
        except Exception:
            pass
        try:
            env = getattr(self.app_config, "env", None)
            if env is not None:
                v = getattr(env, f"rbw_{a}", None)
                if v is not None:
                    v = int(v)
                    return v if v > 1 else int(default_rbw)
        except Exception:
            pass
        return int(default_rbw)

    def _parse_horizon_list(self, v: Any) -> List[int]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            parts = list(v)
        else:
            parts = str(v).replace(",", " ").split()
        out: List[int] = []
        for p in parts:
            try:
                h = int(p)
                if h >= 1:
                    out.append(h)
            except Exception:
                continue
        return sorted(set(out))

    def _pick_base_horizon(self, env_h: Optional[int], horizons: List[int]) -> int:
        hs = sorted(int(h) for h in horizons if int(h) >= 1)
        if not hs:
            return int(env_h or 1)
        if env_h in hs:
            return int(env_h)
        if 20 in hs:
            return 20
        return hs[-1]

    # ---------- Agent / column selection ----------
    def _detect_agents(self, df: pd.DataFrame) -> List[str]:
        if df is None or len(df) == 0:
            return ["xsr", "mamba", "mamba_periodic", "fed"]
        cols = [str(c).lower() for c in df.columns]
        agents: List[str] = []
        if any("xsr" in c for c in cols):
            agents.append("xsr")
        if any("mamba_periodic" in c for c in cols):
            agents.append("mamba_periodic")
        if any(("mamba" in c and "mamba_periodic" not in c) or "lstm" in c for c in cols):
            agents.append("mamba")
        if any("fed" in c for c in cols):
            agents.append("fed")
        return agents or ["xsr", "mamba", "mamba_periodic", "fed"]

    def _agent_suffix(self, agent: str) -> str:
        a = (agent or "").strip().lower()
        if a == "xsr":
            return "XSR"
        if a in {"mamba", "lstm"}:
            return "MAMBA"
        if a in {"mamba_periodic", "periodic", "mamba-periodic"}:
            return "MAMBA_PERIODIC"
        if a in {"fed", "fedformer"}:
            return "FED"
        return a.upper() if a else "MODEL"

    def _pick_pred_cols(self, df: pd.DataFrame, agent: str) -> Dict[int, str]:
        if df is None or len(df) == 0:
            return {}
        a = (agent or "").strip().lower()
        suf = self._agent_suffix(agent)
        cols = list(df.columns)

        out: Dict[int, Tuple[int, str]] = {}
        for c in cols:
            cl = str(c).lower()
            m = re.search(r"_h(\d{1,3})$", cl)
            if not m:
                continue
            h = int(m.group(1))
            if suf.lower() not in cl:
                continue
            if a in {"mamba", "lstm"} and "mamba_periodic" in cl:
                continue
            if "pred" not in cl or "close" not in cl:
                continue
            score = 0
            if "scaled" in cl:
                score += 10
            if "pred_close" in cl:
                score += 5
            score += 3
            prev = out.get(h)
            if prev is None or score > prev[0]:
                out[h] = (score, str(c))

        if out:
            return {h: col for h, (sc, col) in sorted(out.items(), key=lambda kv: kv[0])}

        for cand in (f"Pred_Close_scaled_{suf}", f"Pred_Close_{suf}", f"Close_pred_{suf}"):
            if cand in df.columns:
                return {1: cand}
            for c in cols:
                if str(c).lower() == cand.lower():
                    return {1: str(c)}

        for c in cols:
            cl = str(c).lower()
            if suf.lower() in cl and "pred" in cl and "close" in cl:
                if a in {"mamba", "lstm"} and "mamba_periodic" in cl:
                    continue
                return {1: str(c)}
        return {}

    # ---------- Envelope daily (causal) ----------
    def _daily_feature_cols(self) -> List[str]:
        return [
            "DeltaP_pct",
            "Theta_norm",
            "Top_pct",
            "Bottom_pct",
            "D_norm",
            "DeltaP",
            "D",
            "L",
            "Theta_deg",
            "Top_abs",
            "Bottom_abs",
        ]

    def _envelope_daily_from_series(self, series: pd.Series, dates: pd.Series, rbw: int = 20) -> pd.DataFrame:
        rbw = int(rbw) if rbw and int(rbw) > 1 else 20
        s = pd.to_numeric(series, errors="coerce").astype(float).ffill().fillna(0.0)
        dts = pd.to_datetime(dates, errors="coerce").ffill()

        top = s.rolling(rbw, min_periods=1).max()
        bot = s.rolling(rbw, min_periods=1).min()
        rng = (top - bot).replace(0, np.nan)

        pos = ((s - bot) / (rng + 1e-9)).fillna(0.5)
        top_pct = (pos * 100.0).clip(0.0, 100.0)
        bottom_pct = ((1.0 - pos) * 100.0).clip(0.0, 100.0)

        D = np.zeros(len(s), dtype=float)
        DeltaP = (top - bot).fillna(0.0).to_numpy(dtype=float)

        s_np = s.to_numpy(dtype=float)
        for i in range(len(s_np)):
            j0 = max(0, i - rbw + 1)
            win = s_np[j0 : i + 1]
            if win.size <= 1:
                D[i] = 0.0
            else:
                imax = int(np.argmax(win))
                imin = int(np.argmin(win))
                D[i] = float(abs(imax - imin))

        D_norm = D / float(rbw)
        bot_np = bot.to_numpy(dtype=float)
        DeltaP_pct = np.divide(DeltaP, bot_np, out=np.zeros_like(DeltaP, dtype=float), where=(bot_np != 0.0)) * 100.0

        Theta_deg = np.degrees(np.arctan2(DeltaP, np.maximum(D, 1.0)))
        Theta_norm = Theta_deg / 90.0
        L = np.hypot(D, DeltaP)

        return pd.DataFrame(
            {
                "Date": pd.to_datetime(dts).dt.strftime("%Y-%m-%d"),
                "DeltaP_pct": DeltaP_pct,
                "Theta_norm": Theta_norm,
                "Top_pct": top_pct.to_numpy(dtype=float),
                "Bottom_pct": bottom_pct.to_numpy(dtype=float),
                "D_norm": D_norm,
                "DeltaP": DeltaP,
                "D": D,
                "L": L,
                "Theta_deg": Theta_deg,
                "Top_abs": top.to_numpy(dtype=float),
                "Bottom_abs": bot.to_numpy(dtype=float),
            }
        )

    def _events_from_daily(self, daily_df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        if daily_df is None or len(daily_df) == 0:
            return pd.DataFrame([])
        col_dp = f"DeltaP_pct{suffix}"
        col_theta = f"Theta_norm{suffix}"
        if col_dp not in daily_df.columns or col_theta not in daily_df.columns:
            return pd.DataFrame([])

        dts = pd.to_datetime(daily_df["Date"], errors="coerce")
        dp = pd.to_numeric(daily_df[col_dp], errors="coerce").fillna(0.0).to_numpy()
        theta = pd.to_numeric(daily_df[col_theta], errors="coerce").fillna(0.0).to_numpy()

        top_col = f"Top_abs{suffix}"
        bot_col = f"Bottom_abs{suffix}"
        top = pd.to_numeric(daily_df.get(top_col, 0.0), errors="coerce").fillna(0.0).to_numpy()
        bot = pd.to_numeric(daily_df.get(bot_col, 0.0), errors="coerce").fillna(0.0).to_numpy()
        direction = np.where(top >= bot, "UP", "DOWN")

        rows = []
        start_i = 0
        for i in range(1, len(direction)):
            if direction[i] != direction[i - 1]:
                rows.append(
                    {
                        "Direction": str(direction[i - 1]),
                        "Start": dts.iloc[start_i].strftime("%Y-%m-%d") if pd.notna(dts.iloc[start_i]) else "",
                        "End": dts.iloc[i - 1].strftime("%Y-%m-%d") if pd.notna(dts.iloc[i - 1]) else "",
                        "Duration": int(i - 1 - start_i),
                        "DeltaP_pct_mean": float(np.mean(dp[start_i:i])),
                        "Theta_norm_mean": float(np.mean(theta[start_i:i])),
                    }
                )
                start_i = i

        if len(direction) > 0:
            rows.append(
                {
                    "Direction": str(direction[-1]),
                    "Start": dts.iloc[start_i].strftime("%Y-%m-%d") if pd.notna(dts.iloc[start_i]) else "",
                    "End": dts.iloc[len(direction) - 1].strftime("%Y-%m-%d") if pd.notna(dts.iloc[len(direction) - 1]) else "",
                    "Duration": int(len(direction) - 1 - start_i),
                    "DeltaP_pct_mean": float(np.mean(dp[start_i:len(direction)])),
                    "Theta_norm_mean": float(np.mean(theta[start_i:len(direction)])),
                }
            )
        return pd.DataFrame(rows)

    def _empty_daily(self, dates: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        if dates is None or len(dates) == 0:
            dts = pd.date_range(start_ts, end_ts, freq="B")
        else:
            dts = pd.to_datetime(dates, errors="coerce").dropna()
            if len(dts) == 0:
                dts = pd.date_range(start_ts, end_ts, freq="B")
        base = {"Date": pd.to_datetime(dts).strftime("%Y-%m-%d")}
        for c in self._daily_feature_cols():
            base[c] = 0.0
        return pd.DataFrame(base)

    def _write_daily_snapshots_from_stepA(self, daily_df: pd.DataFrame, symbol: str, agent: str, mode: str, step_d_dir: Path) -> None:
        """Create StepD daily snapshot files (1 row per day) for StepA's test dates, if StepA daily outputs exist."""
        if daily_df is None or len(daily_df) == 0:
            return
        output_root = self._get_output_root()
        step_a_dir = output_root / "stepA" / mode
        dates = self._collect_stepA_daily_dates(step_a_dir=step_a_dir, symbol=symbol)
        if not dates:
            return
        if "Date" not in daily_df.columns:
            return

        df = daily_df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        daily_dir = step_d_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            file_date = dt.strftime("%Y_%m_%d")
            one = df.loc[df["Date"] == date_str]
            if one.empty:
                continue
            out_path = daily_dir / f"stepD_daily_envelope_{agent}_{symbol}_{file_date}.csv"
            one.to_csv(out_path, index=False, encoding="utf-8")
            rows.append({"Date": date_str, "stepD_daily_path": str(out_path.as_posix())})

        if rows:
            # Per-agent manifest (keeps backwards compatibility)
            manifest_path_agent = step_d_dir / f"stepD_daily_manifest_{agent}_{symbol}.csv"
            pd.DataFrame(rows).to_csv(manifest_path_agent, index=False, encoding="utf-8")
            print(f"[StepD] wrote daily snapshots: {len(rows)} agent={agent} -> {manifest_path_agent}")

            # Global manifest (expected by workflow): output/stepD/<mode>/stepD_daily_manifest_<SYMBOL>.csv
            rows_g = []
            for r in rows:
                rr = dict(r)
                rr["agent"] = agent
                rows_g.append(rr)
            manifest_path_global = step_d_dir / f"stepD_daily_manifest_{symbol}.csv"
            try:
                if manifest_path_global.exists():
                    old = pd.read_csv(manifest_path_global, encoding="utf-8")
                    new = pd.DataFrame(rows_g)
                    merged = pd.concat([old, new], ignore_index=True)
                    # Keep latest per (Date, agent)
                    if "Date" in merged.columns and "agent" in merged.columns:
                        merged = merged.drop_duplicates(subset=["Date", "agent"], keep="last")
                    merged.to_csv(manifest_path_global, index=False, encoding="utf-8")
                else:
                    pd.DataFrame(rows_g).to_csv(manifest_path_global, index=False, encoding="utf-8")
                # Only print once per agent write (path shown for clarity)
                print(f"[StepD] ensured global daily manifest -> {manifest_path_global}")
            except Exception as e:
                print(f"[StepD] WARN: failed to write global daily manifest: {e}")

    def _collect_stepA_daily_dates(self, step_a_dir: Path, symbol: str) -> List[pd.Timestamp]:
        manifest = step_a_dir / f"stepA_daily_manifest_{symbol}.csv"
        if manifest.exists():
            try:
                mdf = pd.read_csv(manifest)
                for col in ("Date", "date", "DATE"):
                    if col in mdf.columns:
                        dts = pd.to_datetime(mdf[col], errors="coerce").dropna().drop_duplicates().sort_values()
                        return [pd.Timestamp(x).normalize() for x in dts.tolist()]
            except Exception:
                pass

        daily_dir = step_a_dir / "daily"
        if not daily_dir.exists():
            return []

        import re as _re
        pat = _re.compile(rf"{_re.escape(symbol)}_(\d{{4}})_(\d{{2}})_(\d{{2}})\.csv$")
        dates: List[pd.Timestamp] = []
        for p in daily_dir.glob(f"*{symbol}_*.csv"):
            m = pat.search(p.name)
            if not m:
                continue
            y, mo, d = map(int, m.groups())
            try:
                dates.append(pd.Timestamp(year=y, month=mo, day=d).normalize())
            except Exception:
                continue
        return sorted(set(dates))
