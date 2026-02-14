# -*- coding: utf-8 -*-
r"""tools/run_router_auto_decision_one_day.py

Auto Router (Contextual Bandit / LinUCB) 1日分の推論ツール。

目的
  - run_router_bandit_backtest.py が出力した policy/meta/router_auto を読み、
    指定日1日の bandit ルーター意思決定（chosen_agent / ratio）を再現する。
  - OPS/LIVE で「今日の推奨エージェント / ratio」を出す。

想定入力
  (A) router_auto_{SYMBOL}.yaml (または .json)
      - phase2_state CSV のパス
      - agent_csv (name->csv) マップ
      - policy_train / policy_testB_final のパス（無い場合は自動探索）
  (B) router_bandit 出力ディレクトリ
      - router_meta_bandit_{SYMBOL}.csv
      - router_policy_train.pkl
      - router_policy_testB_final.pkl (任意)

注意
  - policyファイルは「LinUCBPolicy.dumps() の bytes」をそのまま保存している前提。
  - ルーターの action は 0=NO_TRADE, 1..N=agents（agent_names の順序）
  - agent_names の順序は router_auto の agent_names を優先。
    無い場合は router-log の (action_chosen -> chosen_agent) から推定を試みる。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    # tools/ 配下から呼ばれても router_bandit を import できるようにする
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from router_bandit.context_builder import RouterContextBuilder  # noqa: E402
from router_bandit.io_utils import load_agent_ratio_series, load_phase2_state  # noqa: E402
from router_bandit.linucb import LinUCBPolicy  # noqa: E402
from router_bandit.routers import BanditRouter  # noqa: E402
from router_bandit.size_scaler import SizeScaler  # noqa: E402


def _coerce_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"missing '{col}' column")
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[col].isna().any():
        bad = out.loc[out[col].isna(), col].head(5).tolist()
        raise ValueError(f"Date parse failed. examples={bad}")
    out[col] = out[col].dt.tz_localize(None)
    return out


def _read_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "PyYAML が未インストールです。pip install pyyaml するか router_auto を json で出力してください。"
            ) from e
        if not isinstance(obj, dict):
            raise ValueError("router_auto yaml must be a dict")
        return obj

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("router_auto json must be a dict")
    return obj


def _discover_router_auto_config(symbol: str, mode: str, output_root: str, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    base = Path(output_root) / "stepF" / mode
    cands = [
        base / f"router_auto_{symbol}.yaml",
        base / f"router_auto_{symbol}.yml",
        base / f"router_auto_{symbol}.json",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "router_auto config が見つかりません。"
        "run_router_bandit_backtest.py を実行して router_auto を生成するか、--router-auto-config を指定してください。"
    )


def _resolve_path(p: str, base_dir: Path) -> Path:
    # 空文字や '.' は無効扱い
    if p is None:
        return Path(".")
    ps = str(p).strip()
    if ps in ("", "."):
        return Path(".")
    pp = Path(ps)
    if pp.is_absolute():
        return pp
    # router_auto の相対パスは「実行カレント or repo root」依存になりがちなので、
    # まず base_dir（output_root/stepF/mode）基準も試す
    cand = base_dir / pp
    if cand.exists():
        return cand
    # 次に repo root 基準
    cand2 = REPO_ROOT / pp
    return cand2


def _find_first_existing(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        if p.exists() and p.is_file():
            return p
    return None


def _load_policy_bytes(policy_path: Path) -> LinUCBPolicy:
    if (not policy_path.exists()) or policy_path.is_dir():
        raise FileNotFoundError(str(policy_path))
    b = policy_path.read_bytes()
    # policy.dumps() の bytes をそのまま保存している想定
    return LinUCBPolicy.loads(b)


def _load_meta(meta_csv: Path) -> Dict[str, float]:
    if (not meta_csv.exists()) or meta_csv.is_dir():
        raise FileNotFoundError(str(meta_csv))
    df = pd.read_csv(meta_csv)
    if len(df) == 0:
        raise ValueError(f"meta csv is empty: {meta_csv}")
    row = df.iloc[0].to_dict()
    out: Dict[str, float] = {}
    for k in ("best_T0_quantile", "best_T0", "best_T1", "context_dim", "n_actions"):
        if k in row:
            out[k] = float(row[k])
    return out


def _seed_hold_state_from_router_log(
    router: BanditRouter,
    router_log_path: Path,
    target_date: pd.Timestamp,
    min_hold_days: int,
) -> Tuple[float, int]:
    """router_log から前日までの action/pos を seed する。

    返り値: (pos_prev, action_prev)
    """

    df = pd.read_csv(router_log_path)
    df = _coerce_date_col(df, "Date")
    df = df.sort_values("Date").reset_index(drop=True)
    df_prev = df[df["Date"] < target_date]
    if len(df_prev) == 0:
        router._last_action = 0
        router._hold_counter = 0
        return 0.0, 0

    last = df_prev.iloc[-1]
    action_prev = int(last.get("action_chosen", 0))
    pos_prev = float(last.get("pos", 0.0))

    act_series = df_prev["action_chosen"].astype(int).to_numpy()
    k = 1
    for i in range(len(act_series) - 2, -1, -1):
        if int(act_series[i]) == action_prev:
            k += 1
        else:
            break
    hold_counter = max(int(min_hold_days) - int(k), 0)

    router._last_action = int(action_prev)
    router._hold_counter = int(hold_counter)
    return pos_prev, action_prev


def _row_for_date(
    *,
    date_str: str,
    df_router_log: Optional[pd.DataFrame],
    phase2_by_date: Optional[Dict[str, pd.Series]],
) -> pd.Series:
    """
    Choose the row that will be fed into router.decide().

    Priority (when available):
      1) router-log row (ground truth for replay / exact match)
      2) phase2_state row (for ops/live usage where router-log doesn't exist)

    Note:
      - Some router-log files may not contain all Phase2 fields (regime/trend/phase/agreement_*).
        In that case, we will *fill missing fields only* from phase2_state when available.
    """
    required_phase2 = ["regime_cluster", "trend_cluster", "phase", "agreement_dist", "agreement_label"]

    if df_router_log is not None:
        sub = df_router_log[df_router_log["Date"].astype(str) == date_str]
        if len(sub) >= 1:
            row = sub.iloc[0].copy()
            if phase2_by_date is not None and date_str in phase2_by_date:
                row_p2 = phase2_by_date[date_str]
                for k in required_phase2:
                    if (k not in row.index) or pd.isna(row.get(k, np.nan)):
                        if k in row_p2.index:
                            row[k] = row_p2[k]
            return row

    if phase2_by_date is not None and date_str in phase2_by_date:
        return phase2_by_date[date_str]

    raise KeyError(f"Date not found: {date_str}")

def _replay_until_date(
    *,
    router: BanditRouter,
    phase2_by_date: Dict[str, pd.Series],
    df_router_log: pd.DataFrame,
    agent_ratio_by_agent: Dict[str, pd.Series],
    target_date: str,
) -> Tuple[np.ndarray, RouterDecision, float, int]:
    """Replay router decisions from the first date in router_log up to target_date.

    This reproduces BanditRouter's internal state (_last_action/_hold_counter) exactly,
    so it should match backtest logs (testA/testB) when policy/scaler/inputs match.

    Returns: (x, decision, pos_prev_before_decision, action_prev_before_decision)
    """
    df = df_router_log.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    if target_date not in set(df["Date"].tolist()):
        raise KeyError(f"target_date {target_date} not found in router_log")

    dates = sorted(df["Date"].unique().tolist())
    pos_prev = 0.0
    action_prev = 0

    last_x: Optional[np.ndarray] = None
    last_dec: Optional[RouterDecision] = None
    last_seed_pos = 0.0
    last_seed_action = 0

    for d in dates:
        last_seed_pos = float(pos_prev)
        last_seed_action = int(action_prev)

        row = _row_for_date(date_str=d, df_router_log=df, phase2_by_date=phase2_by_date)
        agent_map: Dict[str, float] = {}
        for a, s in agent_ratio_by_agent.items():
            v = float(s.get(d, 0.0))
            if not np.isfinite(v):
                v = 0.0
            agent_map[a] = v

        x, dec = router.decide(row=row, pos_prev=pos_prev, action_prev=action_prev, agent_ratio_map=agent_map)
        last_x, last_dec = x, dec

        # advance external state like backtest_runner
        pos_prev = float(dec.ratio)
        action_prev = int(dec.action)

        if d == target_date:
            break

    assert last_x is not None and last_dec is not None
    return last_x, last_dec, last_seed_pos, last_seed_action


def _infer_agent_names_from_router_log(router_log_path: Path) -> Dict[int, str]:
    """router_log から action_chosen -> chosen_agent を推定（観測できた action のみ）。"""
    df = pd.read_csv(router_log_path)
    if "action_chosen" not in df.columns or "chosen_agent" not in df.columns:
        return {}
    m: Dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            a = int(r["action_chosen"])
        except Exception:
            continue
        if a <= 0:
            continue
        name = str(r["chosen_agent"]) if pd.notna(r["chosen_agent"]) else ""
        if name and name != "NO_TRADE":
            m[a] = name
    return m


def _build_agent_ratio_cache(agent_csv: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    for name, path in agent_csv.items():
        df = load_agent_ratio_series(path)
        df = _coerce_date_col(df, "Date")
        cache[name] = df[["Date", "agent_ratio"]].copy()
    return cache


def _agent_ratio_map_for_date(
    cache: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, df in cache.items():
        m = df[df["Date"] == date]
        if len(m) == 0:
            out[name] = 0.0
        else:
            out[name] = float(m["agent_ratio"].iloc[0])
    return out


@dataclass
class RouterAutoConfig:
    symbol: str
    mode: str
    output_root: str
    phase2_state: str
    agent_csv: Dict[str, str]
    agent_names: List[str]
    policy_train: str
    policy_testB_final: Optional[str]
    meta_csv: str
    trade_cost_bps: float
    min_hold_days: int
    delta_threshold: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RouterAutoConfig":
        # agent_csv
        agent_csv = d.get("agent_csv") or d.get("agents") or d.get("agent_paths") or {}
        if isinstance(agent_csv, list):
            agent_csv = {it["name"]: it["path"] for it in agent_csv}
        agent_csv = {str(k): str(v) for k, v in dict(agent_csv).items()}

        # agent_names
        agent_names = d.get("agent_names")
        if agent_names is None:
            agent_names = list(agent_csv.keys())
        else:
            agent_names = [str(x) for x in list(agent_names)]

        return RouterAutoConfig(
            symbol=str(d.get("symbol", "")),
            mode=str(d.get("mode", "")),
            output_root=str(d.get("output_root", "")),
            phase2_state=str(
                d.get("phase2_state")
                or d.get("phase2_state_path")
                or d.get("phase2_state_csv")
                or d.get("phase2_state_csv_path")
                or ""
            ),
            agent_csv=agent_csv,
            agent_names=agent_names,
            policy_train=str(
                d.get("policy_train")
                or d.get("policy_train_path")
                or d.get("router_policy_train")
                or d.get("policy_path")
                or ""
            ),
            policy_testB_final=(
                str(d.get("policy_testB_final") or d.get("router_policy_testB_final") or d.get("policy_testB_final_path"))
                if (d.get("policy_testB_final") or d.get("router_policy_testB_final") or d.get("policy_testB_final_path"))
                else None
            ),
            meta_csv=str(d.get("meta_csv") or d.get("meta_csv_path") or d.get("router_meta_csv") or ""),
            trade_cost_bps=float(d.get("trade_cost_bps", 10.0)),
            min_hold_days=int(d.get("min_hold_days", 0)),
            delta_threshold=float(d.get("delta_threshold", 0.0)),
        )


def _select_policy_path(
    cfg: RouterAutoConfig,
    base_dir: Path,
    output_root: str,
    mode: str,
    symbol: str,
    which: str,
    override: Optional[str],
) -> Path:
    if override:
        p = Path(override)
        if p.exists() and p.is_file():
            return p
        # relative fallback
        p2 = _resolve_path(override, base_dir)
        if p2.exists() and p2.is_file():
            return p2
        raise FileNotFoundError(str(p2))

    # config first
    raw = cfg.policy_testB_final if which == "testB" else cfg.policy_train
    p = _resolve_path(raw, base_dir)
    if p.exists() and p.is_file():
        return p

    # auto fallback to router_bandit outputs
    out_dir = Path(output_root) / "stepF" / mode / "router_bandit"
    if which == "testB":
        cands = [
            out_dir / "router_policy_testB_final.pkl",
            out_dir / f"router_policy_testB_final_{symbol}.pkl",
        ]
    else:
        cands = [
            out_dir / "router_policy_train.pkl",
            out_dir / f"router_policy_train_{symbol}.pkl",
        ]
    found = _find_first_existing(cands)
    if found is not None:
        return found
    raise FileNotFoundError(f"policy file not found. tried: {[str(x) for x in cands]}")


def _select_meta_csv(cfg: RouterAutoConfig, base_dir: Path, output_root: str, mode: str, symbol: str) -> Path:
    p = _resolve_path(cfg.meta_csv, base_dir)
    if p.exists() and p.is_file():
        return p
    out_dir = Path(output_root) / "stepF" / mode / "router_bandit"
    cands = [
        out_dir / f"router_meta_bandit_{symbol}.csv",
        out_dir / "router_meta_bandit.csv",
        out_dir / f"router_meta_{symbol}.csv",
        out_dir / "router_meta.csv",
    ]
    found = _find_first_existing(cands)
    if found is not None:
        return found
    raise FileNotFoundError(f"meta csv not found. tried: {[str(x) for x in cands]}")


def _select_phase2_state(cfg: RouterAutoConfig, base_dir: Path, output_root: str, mode: str, symbol: str, override: Optional[str]) -> Path:
    if override:
        p = Path(override)
        if p.exists() and p.is_file():
            return p
        p2 = _resolve_path(override, base_dir)
        if p2.exists() and p2.is_file():
            return p2
        raise FileNotFoundError(str(p2))

    p = _resolve_path(cfg.phase2_state, base_dir)
    if p.exists() and p.is_file():
        return p

    # fallback: phase2_state_* in stepF/mode
    d = Path(output_root) / "stepF" / mode
    cands = [
        d / f"phase2_state_{symbol}.csv",
        d / f"phase2_state_{symbol}_auto.csv",
        d / f"phase2_state_{symbol}_auto.csv".replace("_auto", "_auto"),
        d / f"phase2_state_{symbol}_auto.csv",  # keep
        d / f"phase2_state_{symbol}_auto.csv",  # keep
        d / f"phase2_state_{symbol}_auto.csv",  # keep
    ]
    # also glob
    globbed = sorted(d.glob(f"phase2_state*{symbol}*.csv"))
    cands = cands + globbed
    found = _find_first_existing(cands)
    if found is not None:
        return found
    raise FileNotFoundError(f"phase2_state not found. tried in: {d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--router-auto-config", default=None)
    ap.add_argument("--router-log", default=None, help="seed 用: router_daily_log_*.csv")
    ap.add_argument(
        "--no-replay",
        action="store_true",
        help="If set, do NOT replay from router-log. Default behavior: when --router-log is provided, this tool replays from the first date up to --date to reproduce internal hold-state exactly as in backtest.",
    )
    ap.add_argument("--policy", default="train", choices=["train", "testB"], help="どの policy を使うか")
    ap.add_argument("--policy-path", default=None, help="policy を明示指定（router_auto/config より優先）")
    ap.add_argument("--phase2-state", default=None, help="phase2_state を明示指定（router_auto/config より優先）")
    ap.add_argument(
        "--agent-csv",
        action="append",
        default=[],
        help="(router_auto に agent_csv が無い場合のみ) repeatable: name=path",
    )
    args = ap.parse_args()

    cfg_path = _discover_router_auto_config(args.symbol, args.mode, args.output_root, args.router_auto_config)
    raw = _read_yaml_or_json(cfg_path)
    cfg = RouterAutoConfig.from_dict(raw)

    base_dir = Path(args.output_root) / "stepF" / args.mode

    # agent csv mapping fallback
    if (not cfg.agent_csv) and args.agent_csv:
        # parse name=path
        m: Dict[str, str] = {}
        for kv in args.agent_csv:
            if "=" not in kv:
                raise SystemExit(f"--agent-csv must be name=path, got: {kv}")
            k, v = kv.split("=", 1)
            m[str(k).strip()] = str(v).strip()
        cfg.agent_csv = m
        cfg.agent_names = list(m.keys())

    if not cfg.agent_csv:
        raise SystemExit("router_auto に agent_csv がありません。--agent-csv を指定するか router_auto を再生成してください")

    # resolve & validate important files
    policy_path = _select_policy_path(
        cfg,
        base_dir=base_dir,
        output_root=args.output_root,
        mode=args.mode,
        symbol=args.symbol,
        which=("testB" if args.policy == "testB" else "train"),
        override=args.policy_path,
    )
    meta_csv = _select_meta_csv(cfg, base_dir=base_dir, output_root=args.output_root, mode=args.mode, symbol=args.symbol)
    phase2_csv = _select_phase2_state(cfg, base_dir=base_dir, output_root=args.output_root, mode=args.mode, symbol=args.symbol, override=args.phase2_state)

    meta = _load_meta(meta_csv)
    best_T0 = float(meta.get("best_T0", 0.0))
    best_T1 = float(meta.get("best_T1", 1.0))

    # phase2
    phase2 = load_phase2_state(str(phase2_csv))
    phase2 = _coerce_date_col(phase2, "Date")
    regime_cats = np.sort(phase2["regime_cluster"].astype(int).unique())
    trend_cats = np.sort(phase2["trend_cluster"].astype(int).unique())
    phase_cats = np.sort(phase2["phase"].astype(int).unique())
    cb = RouterContextBuilder(regime_cats, trend_cats, phase_cats)
    scaler = SizeScaler(T0=best_T0, T1=best_T1)
    pol = _load_policy_bytes(policy_path)

    # agent_names order
    agent_names: List[str]
    if cfg.agent_names and all(isinstance(x, str) for x in cfg.agent_names):
        agent_names = list(cfg.agent_names)
    else:
        agent_names = list(cfg.agent_csv.keys())

    # If router-log exists and agent_names not explicit, try to infer mapping (action->name)
    if args.router_log and ("agent_names" not in raw):
        inferred = _infer_agent_names_from_router_log(Path(args.router_log))
        if inferred:
            na = int(meta.get("n_actions", len(agent_names) + 1))
            N = max(na - 1, max(inferred.keys(), default=0))
            tmp = [None] * N
            for a, nm in inferred.items():
                if 1 <= a <= N:
                    tmp[a - 1] = nm
            # fill remaining from agent_csv
            leftover = [k for k in cfg.agent_csv.keys() if k not in tmp]
            for i in range(N):
                if tmp[i] is None:
                    tmp[i] = leftover.pop(0) if leftover else list(cfg.agent_csv.keys())[0]
            agent_names = [str(x) for x in tmp]

    router = BanditRouter(
        policy=pol,
        scaler=scaler,
        context_builder=cb,
        agent_names=agent_names,
        min_hold_days=int(cfg.min_hold_days),
        delta_threshold=float(cfg.delta_threshold),
    )

    # prepare ratios
    ratio_cache = _build_agent_ratio_cache(cfg.agent_csv)

    target_date = pd.to_datetime(args.date).tz_localize(None)
    target_str = target_date.strftime("%Y-%m-%d")

    # Build quick lookups for replay
    phase2_tmp = phase2.copy()
    phase2_tmp["Date"] = pd.to_datetime(phase2_tmp["Date"]).dt.strftime("%Y-%m-%d")
    phase2_by_date: Dict[str, pd.Series] = {d: r for d, r in phase2_tmp.set_index("Date").iterrows()}

    # agent_ratio_by_agent: agent -> Series(date_str -> ratio)
    agent_ratio_by_agent: Dict[str, pd.Series] = {}
    for nm, df in ratio_cache.items():
        dstr = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        agent_ratio_by_agent[nm] = pd.Series(df["agent_ratio"].astype(float).values, index=dstr)

    df_router_log: Optional[pd.DataFrame] = None
    if args.router_log:
        rp = Path(args.router_log)
        if rp.exists():
            df_router_log = pd.read_csv(rp)

    # Default: if router-log is provided, REPLAY from the first date up to --date to reproduce internal hold state.
    if (df_router_log is not None) and (not args.no_replay):
        x, dec, pos_prev, action_prev = _replay_until_date(
            router=router,
            phase2_by_date=phase2_by_date,
            df_router_log=df_router_log,
            agent_ratio_by_agent=agent_ratio_by_agent,
            target_date=target_str,
        )
        row_s = _row_for_date(date_str=target_str, df_router_log=df_router_log, phase2_by_date=phase2_by_date)

        # If both sources exist, sanity-check Phase2 fields and warn if they differ.
        if df_router_log is not None and phase2_by_date is not None and target_str in phase2_by_date:
            try:
                row_log_sub = df_router_log[df_router_log["Date"].astype(str) == target_str]
                if len(row_log_sub) >= 1:
                    row_log = row_log_sub.iloc[0]
                    row_p2 = phase2_by_date[target_str]
                    keys = ["regime_cluster", "trend_cluster", "phase", "agreement_dist", "agreement_label"]
                    diffs = []
                    for k in keys:
                        if k in row_log.index and k in row_p2.index:
                            v1 = float(row_log[k]) if pd.notna(row_log[k]) else float("nan")
                            v2 = float(row_p2[k]) if pd.notna(row_p2[k]) else float("nan")
                            if not (np.isclose(v1, v2, equal_nan=True)):
                                diffs.append((k, v1, v2))
                    if diffs:
                        print("  [WARN] Phase2 fields differ between router-log row and phase2_state row:")
                        for k, v1, v2 in diffs:
                            print(f"         {k}: router-log={v1}  phase2_state={v2}")
            except Exception:
                pass
        agent_map = {a: float(agent_ratio_by_agent.get(a, pd.Series(dtype=float)).get(target_str, 0.0)) for a in agent_names}
    else:
        # one-day mode (best-effort seed)
        if target_str not in phase2_by_date:
            raise SystemExit(f"phase2_state に指定日がありません: {target_str}")
        row_s = _row_for_date(date_str=target_str, df_router_log=df_router_log, phase2_by_date=phase2_by_date)
        agent_map = _agent_ratio_map_for_date(ratio_cache, target_date)
        if args.router_log and df_router_log is not None:
            pos_prev, action_prev = _seed_hold_state_from_router_log(
                router=router,
                router_log_path=Path(args.router_log),
                target_date=target_date,
                min_hold_days=int(cfg.min_hold_days),
            )
        else:
            pos_prev, action_prev = 0.0, 0

        x, dec = router.decide(
            row=row_s,
            pos_prev=float(pos_prev),
            action_prev=int(action_prev),
            agent_ratio_map=agent_map,
        )

    agreement_dist = float(row_s.get("agreement_dist", 0.0))
    agreement_label = float(row_s.get("agreement_label", agreement_dist))

    print("[router_auto_decision]")
    print(f"  config: {cfg_path}")
    print(f"  date:   {target_date.date()}")
    print(f"  policy: {policy_path}")
    print(f"  meta:   T0={best_T0}  T1={best_T1}  ctx_dim={meta.get('context_dim')}  n_actions={meta.get('n_actions')}")
    print(f"  seed:   action_prev={action_prev}  pos_prev={pos_prev:.6f}  hold_counter={getattr(router, '_hold_counter', None)}")
    print(
        f"  phase2: regime={int(row_s['regime_cluster'])} trend={int(row_s['trend_cluster'])} phase={int(row_s['phase'])} "
        f"agreement_dist={agreement_dist:.6f} agreement_label={agreement_label:.6f} size={float(dec.size):.6f}"
    )
    print(
        f"  decide: action={int(dec.action)} chosen_agent={dec.chosen_agent} agent_ratio={float(dec.agent_ratio):.6f} ratio={float(dec.ratio):.6f}"
    )
    print(
        f"  debug:  best_score={float(dec.debug_best_score):.6f} current_score={float(dec.debug_current_score):.6f} gap={float(dec.debug_gap):.6f}"
    )

    if df_router_log is not None and (df_router_log["Date"] == target_str).any():
        exp = df_router_log[df_router_log["Date"] == target_str].iloc[0]
        try:
            exp_action = int(exp.get("action_chosen"))
        except Exception:
            exp_action = None
        exp_agent = str(exp.get("chosen_agent", ""))
        try:
            exp_ratio = float(exp.get("ratio"))
        except Exception:
            exp_ratio = None
        print(f"  expected(from router-log): action={exp_action} chosen_agent={exp_agent} ratio={exp_ratio}")
        if (exp_action is not None) and (exp_action != int(dec.action)):
            print("  [WARN] action mismatch vs router-log")
        if exp_agent and (exp_agent != str(dec.chosen_agent)):
            print("  [WARN] chosen_agent mismatch vs router-log")
        if (exp_ratio is not None) and (abs(exp_ratio - float(dec.ratio)) > 1e-9):
            print("  [WARN] ratio mismatch vs router-log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
