from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_PROFILES: Tuple[str, ...] = (
    "dprime_bnf_h01",
    "dprime_bnf_h02",
    "dprime_bnf_3scale",
    "dprime_mix_h01",
    "dprime_mix_h02",
    "dprime_mix_3scale",
    "dprime_all_features_h01",
    "dprime_all_features_h02",
    "dprime_all_features_h03",
    "dprime_all_features_3scale",
)


@dataclass
class StepDPrimeConfig:
    symbol: str
    mode: str = "sim"
    output_root: str = "output"
    stepA_root: Optional[str] = None
    stepB_root: Optional[str] = None
    stepC_root: Optional[str] = None
    stepDprime_root: Optional[str] = None
    legacy_stepDprime_root: Optional[str] = None
    profiles: Tuple[str, ...] = _PROFILES
    l_past: int = 63
    pred_k: int = 20
    z_past_dim: int = 32
    z_pred_dim: int = 32
    verbose: bool = True


def _normalize_mode(mode: str) -> str:
    m = str(mode or "sim").strip().lower()
    if m in {"ops", "op", "prod", "production", "real"}:
        return "live"
    if m not in {"sim", "live", "display"}:
        raise ValueError(f"StepDPrime invalid mode={mode}. expected sim/live/display")
    return m


def _read_split_summary(stepa_dir: Path, symbol: str) -> Dict[str, str]:
    p = stepa_dir / f"stepA_split_summary_{symbol}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing StepA split summary: {p}")
    df = pd.read_csv(p)
    if {"key", "value"}.issubset(df.columns):
        kv = {str(k): str(v) for k, v in zip(df["key"], df["value"])}
    else:
        kv = {c: str(df.iloc[0][c]) for c in df.columns}
    req = ["train_start", "train_end", "test_start", "test_end"]
    out = {}
    for k in req:
        if k not in kv or kv[k] in {"None", "nan", "NaT"}:
            raise ValueError(f"split summary missing {k}")
        out[k] = str(pd.to_datetime(kv[k]).date())
    return out


def _read_pair(base: Path, stem: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p_tr = base / f"{stem}_train_{symbol}.csv"
    p_te = base / f"{stem}_test_{symbol}.csv"
    if not (p_tr.exists() and p_te.exists()):
        raise FileNotFoundError(f"missing {stem} pair: {p_tr} / {p_te}")
    tr, te = pd.read_csv(p_tr), pd.read_csv(p_te)
    for df in (tr, te):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return tr, te


def _safe(df: pd.DataFrame, c: str, default: float = 0.0) -> pd.Series:
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _compute_base_features(prices: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values("Date").reset_index(drop=True).copy()
    close = _safe(df, "Close")
    open_ = _safe(df, "Open")
    high = _safe(df, "High")
    low = _safe(df, "Low")
    vol = _safe(df, "Volume")
    prev_close = close.shift(1)
    prev_vol = vol.shift(1)

    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    rng = (high - low).replace(0, np.nan)
    macd = _safe(tech, "MACD")
    macd_sig = _safe(tech, "MACD_signal")
    macd_hist = _safe(tech, "MACD_hist") if "MACD_hist" in tech.columns else (macd - macd_sig)

    out = pd.DataFrame({"Date": df["Date"]})
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["range_atr"] = (high - low) / atr14.replace(0, np.nan)
    out["body_ratio"] = (close - open_) / rng
    out["body_atr"] = (close - open_).abs() / atr14.replace(0, np.nan)
    out["upper_wick_ratio"] = (high - np.maximum(open_, close)).clip(lower=0) / rng
    out["lower_wick_ratio"] = (np.minimum(open_, close) - low).clip(lower=0) / rng
    out["Gap"] = _safe(tech, "Gap") if "Gap" in tech.columns else (open_ / prev_close.replace(0, np.nan) - 1.0)
    out["ATR_norm"] = _safe(tech, "ATR_norm") if "ATR_norm" in tech.columns else (atr14 / prev_close.replace(0, np.nan))
    out["gap_atr"] = out["Gap"] / out["ATR_norm"].replace(0, np.nan)
    out["vol_log_ratio_20"] = np.log(vol.replace(0, np.nan)) - np.log(vol.rolling(20, min_periods=20).mean().replace(0, np.nan))
    out["vol_chg"] = vol.pct_change(1)
    z_mu = close.rolling(25, min_periods=25).mean()
    z_sd = close.rolling(25, min_periods=25).std(ddof=0).replace(0, np.nan)
    out["dev_z_25"] = (close - z_mu) / z_sd
    out["bnf_score"] = out["dev_z_25"] * out["vol_log_ratio_20"]
    out["RSI"] = _safe(tech, "RSI") if "RSI" in tech.columns else _safe(tech, "RSI_14")
    out["MACD_hist"] = macd_hist
    out["macd_hist_delta"] = out["MACD_hist"].diff()
    mh = out["MACD_hist"].fillna(0.0)
    out["macd_hist_cross_up"] = ((mh > 0.0) & (mh.shift(1) <= 0.0)).astype(float)
    out["clv"] = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    out["distribution_day"] = ((close < prev_close) & (vol > prev_vol)).astype(float)
    out["dist_count_25"] = out["distribution_day"].rolling(25, min_periods=1).sum()
    out["absorption_day"] = ((close > prev_close) & (vol < prev_vol)).astype(float)
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mf_mult * vol
    out["cmf_20"] = mfv.rolling(20, min_periods=20).sum() / vol.rolling(20, min_periods=20).sum().replace(0, np.nan)

    for c in out.columns:
        if c != "Date":
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_scaler(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _fit_pca(train: np.ndarray, dim: int) -> Tuple[np.ndarray, int]:
    if train.shape[0] < 2:
        raise ValueError("too few train rows for PCA fit")
    x = train - train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    d = int(min(dim, vt.shape[0], vt.shape[1]))
    if d <= 0:
        raise ValueError("invalid PCA dim")
    return vt[:d].T.astype(float), d


def _project(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, comp: np.ndarray) -> np.ndarray:
    xn = (x - mu) / sd
    return xn @ comp


def _build_pred_from_stepb(stepb_dir: Path, symbol: str, pred_k: int) -> Tuple[pd.DataFrame, str]:
    cands = sorted(stepb_dir.glob(f"stepB_pred_pathseq_*_h{pred_k:02d}_{symbol}.csv"))
    if not cands:
        raise FileNotFoundError(f"missing pathseq file under {stepb_dir}")
    p = cands[0]
    df = pd.read_csv(p)
    if "Date_anchor" not in df.columns:
        raise ValueError(f"{p} missing Date_anchor")
    df["Date"] = pd.to_datetime(df["Date_anchor"], errors="coerce").dt.normalize()
    return df, str(p.as_posix())




def _load_pred_time_priority(stepc_dir: Path, stepb_dir: Path, symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
    for root, label in ((stepc_dir, "stepC"), (stepb_dir, "stepB")):
        if root is None or not root.exists():
            continue
        cands = sorted(root.glob(f"stepC_pred_time*_{symbol}.csv")) if label == "stepC" else sorted(root.glob(f"stepB_pred_time*_{symbol}.csv"))
        if not cands:
            continue
        p = cands[0]
        df = pd.read_csv(p)
        dcol = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
        if dcol is None:
            continue
        df["Date"] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
        return df, str(p.as_posix())
    return None, ""
def _infer_pred_type(profile: str) -> str:
    if profile.endswith("_h01"):
        return "h01"
    if profile.endswith("_h02"):
        return "h02"
    if profile.endswith("_h03"):
        return "h03"
    return "3scale"


class StepDPrimeService:
    def run(self, cfg: StepDPrimeConfig) -> Dict[str, object]:
        mode = _normalize_mode(cfg.mode)
        out_root = Path(cfg.output_root)
        stepa_dir = Path(cfg.stepA_root) if cfg.stepA_root else out_root / "stepA" / mode
        stepb_dir = Path(cfg.stepB_root) if cfg.stepB_root else out_root / "stepB" / mode
        stepd_dir = Path(cfg.stepDprime_root) if cfg.stepDprime_root else out_root / "stepDprime" / mode
        legacy_dir = Path(cfg.legacy_stepDprime_root) if cfg.legacy_stepDprime_root else out_root / "stepD_prime" / mode
        emb_dir = legacy_dir / "embeddings"
        stepd_dir.mkdir(parents=True, exist_ok=True)
        legacy_dir.mkdir(parents=True, exist_ok=True)
        emb_dir.mkdir(parents=True, exist_ok=True)

        split = _read_split_summary(stepa_dir, cfg.symbol)
        tr_s, tr_e = pd.to_datetime(split["train_start"]), pd.to_datetime(split["train_end"])
        te_s, te_e = pd.to_datetime(split["test_start"]), pd.to_datetime(split["test_end"])

        pr_tr, pr_te = _read_pair(stepa_dir, "stepA_prices", cfg.symbol)
        tc_tr, tc_te = _read_pair(stepa_dir, "stepA_tech", cfg.symbol)
        pe_tr, pe_te = _read_pair(stepa_dir, "stepA_periodic", cfg.symbol)

        prices = pd.concat([pr_tr, pr_te], ignore_index=True).sort_values("Date").reset_index(drop=True)
        tech = pd.concat([tc_tr, tc_te], ignore_index=True).sort_values("Date").reset_index(drop=True)
        periodic = pd.concat([pe_tr, pe_te], ignore_index=True).sort_values("Date").reset_index(drop=True)
        base = _compute_base_features(prices, tech)
        all_df = base.merge(tech, on="Date", how="left", suffixes=("", "_tech")).merge(periodic, on="Date", how="left")

        num_cols = [c for c in all_df.columns if c != "Date" and pd.api.types.is_numeric_dtype(all_df[c])]
        for c in num_cols:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        stepc_dir = Path(cfg.stepC_root) if cfg.stepC_root else out_root / "stepC" / mode
        pred_df, pred_src = _build_pred_from_stepb(stepb_dir, cfg.symbol, cfg.pred_k)
        pred_time_df, pred_time_src = _load_pred_time_priority(stepc_dir, stepb_dir, cfg.symbol)
        pred_cols = [f"Pred_Close_t_plus_{i:02d}" for i in range(1, cfg.pred_k + 1)]
        for c in pred_cols:
            if c not in pred_df.columns:
                pred_df[c] = np.nan
        data = all_df.merge(pred_df[["Date"] + pred_cols], on="Date", how="left")
        data["Close_anchor"] = _safe(prices, "Close")

        if pred_time_df is not None:
            for h in (1, 5, 10, 20):
                cands = [f"Pred_Close_MAMBA_h{h:02d}", f"Close_pred_h{h}", f"Pred_Close_t_plus_{h:02d}"]
                col = next((c for c in cands if c in pred_time_df.columns), None)
                if col is not None:
                    sub = pred_time_df[["Date", col]].rename(columns={col: f"Pred_Close_t_plus_{h:02d}"})
                    data = data.drop(columns=[f"Pred_Close_t_plus_{h:02d}"], errors="ignore").merge(sub, on="Date", how="left")
            if pred_time_src:
                pred_src = pred_time_src + "|" + pred_src
        for i in range(1, cfg.pred_k + 1):
            c = f"Pred_Close_t_plus_{i:02d}"
            data[f"pred_ret_{i:02d}"] = pd.to_numeric(data[c], errors="coerce") / data["Close_anchor"].replace(0, np.nan) - 1.0
            data[f"pred_ret_{i:02d}"] = data[f"pred_ret_{i:02d}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        bnf_cols = [
            "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
            "Gap", "ATR_norm", "gap_atr", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score",
        ]
        mix_cols = bnf_cols + ["RSI", "MACD_hist", "macd_hist_delta", "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20"]
        all_cols = [c for c in num_cols if c not in {"Open", "High", "Low", "Close", "Volume", "Close_anchor"}]

        results: Dict[str, object] = {"mode": mode, "symbol": cfg.symbol, "profiles": {}, "output_dir": str(stepd_dir)}
        date_list = data["Date"].tolist()
        idx_train = [i for i, d in enumerate(date_list) if tr_s <= d <= tr_e]
        idx_test = [i for i, d in enumerate(date_list) if te_s <= d <= te_e]
        if min(idx_test) < cfg.l_past - 1:
            raise RuntimeError("insufficient history before test_start for L_past window")

        for profile in cfg.profiles:
            fam = "all_features" if "all_features" in profile else ("mix" if "mix" in profile else "bnf")
            pred_type = _infer_pred_type(profile)
            past_cols = all_cols if fam == "all_features" else (mix_cols if fam == "mix" else bnf_cols)
            past_cols = [c for c in past_cols if c in data.columns]

            pred_steps = [1] if pred_type == "h01" else ([1, 5, 10, 20] if pred_type == "h02" else list(range(1, cfg.pred_k + 1)))
            pred_use_cols = [f"pred_ret_{s:02d}" for s in pred_steps]

            def _build_rows(indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
                xp, xf, ds = [], [], []
                for i in indices:
                    if i < cfg.l_past - 1:
                        continue
                    hist = data.iloc[i - cfg.l_past + 1 : i + 1]
                    xp.append(hist[past_cols].to_numpy(dtype=float).reshape(-1))
                    ds.append(data.iloc[i]["Date"])
                    if pred_type == "3scale":
                        p = data.iloc[i][[f"pred_ret_{k:02d}" for k in range(1, cfg.pred_k + 1)]].to_numpy(dtype=float)
                        xf.append(np.concatenate([p[:5], p[:10], p[:20]], axis=0))
                    else:
                        xf.append(data.iloc[i][pred_use_cols].to_numpy(dtype=float))
                return np.asarray(xp, float), np.asarray(xf, float), ds

            Xp_tr, Xf_tr, Dtr = _build_rows(idx_train)
            Xp_te, Xf_te, Dte = _build_rows(idx_test)
            if len(Dtr) == 0 or len(Dte) == 0:
                raise RuntimeError(f"profile={profile}: no rows for train/test")

            mu_p, sd_p = _fit_scaler(Xp_tr)
            mu_f, sd_f = _fit_scaler(Xf_tr)
            comp_p, d_p = _fit_pca((Xp_tr - mu_p) / sd_p, cfg.z_past_dim)
            pred_dim = cfg.z_pred_dim * 3 if pred_type == "3scale" else cfg.z_pred_dim
            comp_f, d_f = _fit_pca((Xf_tr - mu_f) / sd_f, pred_dim)
            Zp_tr, Zp_te = _project(Xp_tr, mu_p, sd_p, comp_p), _project(Xp_te, mu_p, sd_p, comp_p)
            Zf_tr, Zf_te = _project(Xf_tr, mu_f, sd_f, comp_f), _project(Xf_te, mu_f, sd_f, comp_f)

            def _to_df(ds: List[pd.Timestamp], zp: np.ndarray, zf: np.ndarray) -> pd.DataFrame:
                out = pd.DataFrame({"Date": pd.to_datetime(ds).strftime("%Y-%m-%d")})
                for j in range(zp.shape[1]):
                    out[f"zp_{j:03d}"] = zp[:, j]
                for j in range(zf.shape[1]):
                    out[f"zf_{j:03d}"] = zf[:, j]
                ref = data.set_index("Date")
                out["gap_atr"] = [float(ref.loc[pd.to_datetime(d), "gap_atr"]) for d in pd.to_datetime(ds)]
                out["ATR_norm"] = [float(ref.loc[pd.to_datetime(d), "ATR_norm"]) for d in pd.to_datetime(ds)]
                out["pos_prev"] = 0.0
                out["action_prev"] = 0.0
                out["time_in_trade"] = 0.0
                return out

            df_tr, df_te = _to_df(Dtr, Zp_tr, Zf_tr), _to_df(Dte, Zp_te, Zf_te)
            p_tr = stepd_dir / f"stepDprime_state_train_{profile}_{cfg.symbol}.csv"
            p_te = stepd_dir / f"stepDprime_state_test_{profile}_{cfg.symbol}.csv"
            s_path = stepd_dir / f"stepDprime_split_summary_{profile}_{cfg.symbol}.csv"
            df_tr.to_csv(p_tr, index=False)
            df_te.to_csv(p_te, index=False)
            pd.DataFrame([
                {"key": "mode", "value": mode}, {"key": "symbol", "value": cfg.symbol}, {"key": "profile", "value": profile},
                {"key": "train_start", "value": str(tr_s.date())}, {"key": "train_end", "value": str(tr_e.date())},
                {"key": "test_start", "value": str(te_s.date())}, {"key": "test_end", "value": str(te_e.date())},
                {"key": "L_past", "value": cfg.l_past}, {"key": "pred_type", "value": pred_type}, {"key": "pred_k", "value": cfg.pred_k},
                {"key": "z_past_dim", "value": int(d_p)}, {"key": "z_pred_dim", "value": int(d_f)},
                {"key": "rows_train_written", "value": int(len(df_tr))}, {"key": "rows_test_written", "value": int(len(df_te))},
                {"key": "past_feature_channels", "value": "|".join(past_cols)},
                {"key": "pred_source_file", "value": pred_src},
                {"key": "fit_stats", "value": f"train_only:{tr_s.date()}..{tr_e.date()}"},
                {"key": "pca_components_shape", "value": f"past={comp_p.shape},pred={comp_f.shape}"},
            ]).to_csv(s_path, index=False)

            # legacy dual output
            lp_tr = legacy_dir / f"stepDprime_state_{profile}_{cfg.symbol}_train.csv"
            lp_te = legacy_dir / f"stepDprime_state_{profile}_{cfg.symbol}_test.csv"
            df_tr.to_csv(lp_tr, index=False)
            df_te.to_csv(lp_te, index=False)

            emb_cols = [c for c in df_tr.columns if c.startswith("zp_") or c.startswith("zf_")]

            def _to_emb_df(df_state: pd.DataFrame) -> pd.DataFrame:
                out = pd.DataFrame({"Date": pd.to_datetime(df_state["Date"], errors="coerce").dt.strftime("%Y-%m-%d")})
                for i, c in enumerate(emb_cols):
                    out[f"emb_{i:03d}"] = pd.to_numeric(df_state[c], errors="coerce").astype(float)
                return out

            df_emb_tr = _to_emb_df(df_tr)
            df_emb_te = _to_emb_df(df_te)
            df_emb_all = pd.concat([df_emb_tr, df_emb_te], axis=0, ignore_index=True)
            df_emb_all["Date"] = pd.to_datetime(df_emb_all["Date"], errors="coerce")
            df_emb_all = (
                df_emb_all.dropna(subset=["Date"])
                .sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="last")
                .reset_index(drop=True)
            )
            df_emb_all["Date"] = df_emb_all["Date"].dt.strftime("%Y-%m-%d")

            ep_tr = emb_dir / f"stepDprime_{profile}_{cfg.symbol}_embeddings_train.csv"
            ep_te = emb_dir / f"stepDprime_{profile}_{cfg.symbol}_embeddings_test.csv"
            ep_all = emb_dir / f"stepDprime_{profile}_{cfg.symbol}_embeddings_all.csv"
            df_emb_tr.to_csv(ep_tr, index=False)
            df_emb_te.to_csv(ep_te, index=False)
            df_emb_all.to_csv(ep_all, index=False)

            results["profiles"][profile] = {"train": str(p_tr), "test": str(p_te), "summary": str(s_path)}

        (stepd_dir / f"stepDprime_summary_{cfg.symbol}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
