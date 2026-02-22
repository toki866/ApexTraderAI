from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ai_core.utils.paths import resolve_repo_path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class StateBuilderConfig:
    """
    StateBuilder の設定クラス。

    Parameters
    ----------
    output_root : Path
        StepA〜F の出力CSVが置かれているルートディレクトリ。
        例: resolve_repo_path("output")
    envelope_agent : str
        EnvelopeイベントをどのAIエージェントのものから取るか。
        例: "mamba"
    daily_log_pattern : str
        日次ログCSVのパスパターン。
        "{symbol}" を銘柄コードで置き換える。
        例: "output/daily_log_{symbol}.csv"
    mode : str | None
        出力フォルダが output/stepX/<mode>/ で分離されている場合に指定する。
        例: "sim", "ops", "live", "display"
        None の場合は、存在するファイルを探索して見つかったものを使う（優先順: live→ops→sim→display→legacy）。
    """
    output_root: Path = resolve_repo_path("output")
    envelope_agent: str = "mamba"
    daily_log_pattern: str = "output/daily_log_{symbol}.csv"
    mode: Optional[str] = None


class StateBuilder:
    """
    StepE で定義した 24次元観測ベクトル仕様に基づき、指定日の観測ベクトルを構築する。

    観測ベクトル s_t（24次元）
    --------------------------
    1) 3AI の ΔClose 予測（3）
        - dPred_MAMBA
    2) Envelope 幾何特徴（11）
        - DeltaP_pct, theta_norm, Top_pct, Bottom_pct, D_norm,
          DeltaP, D, L, theta_deg, Top_abs, Bottom_abs
    3) テクニカル（3）
        - RSI, MACD, Gap_pct(or Gap)
    4) 価格（5）
        - Open, High, Low, Close, Volume
    5) 履歴（2）
        - PnL_pct(prev), Action(prev)

    注意:
    - Router は get_tech_row() で返す tech_row の BNF_* 列を参照する。
    """

    def __init__(self, config: Optional[StateBuilderConfig] = None) -> None:
        self.config = config or StateBuilderConfig()
        self.output_root: Path = Path(self.config.output_root)
        self.envelope_agent: str = self.config.envelope_agent
        self.daily_log_pattern: str = self.config.daily_log_pattern
        self.mode: Optional[str] = self.config.mode

        # cache
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._features_cache: Dict[str, pd.DataFrame] = {}
        self._pred_cache: Dict[str, pd.DataFrame] = {}
        self._env_cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # (agent,symbol)
        self._log_cache: Dict[str, pd.DataFrame] = {}

    # =====================================================
    # Public API
    # =====================================================
    def get_tech_row(self, symbol: str, trading_date: Union[date, datetime, str]) -> pd.Series:
        """
        指定日の StepA(tech) 行（pandas.Series）を返す。

        Regime Router はこの行の BNF 列（例: BNF_DivDownVolUp）を参照して局面判定します。
        """
        d = self._to_date(trading_date)
        df = self._load_features(symbol)
        return self._get_row_by_date(df, d, "features")

    def build_morning_state(self, symbol: str, trading_date: date) -> np.ndarray:
        """
        朝（オープン直後）に使用する観測ベクトルを構築する（現状は日足ベース）。
        """
        d = self._to_date(trading_date)

        prices = self._load_prices(symbol)
        features = self._load_features(symbol)
        pred = self._load_predictions(symbol)
        env = self._load_envelope(self.envelope_agent, symbol)
        daily_log = self._load_daily_log(symbol)

        row_prices = self._get_row_by_date(prices, d, "prices")
        row_features = self._get_row_by_date(features, d, "features")
        row_pred = self._get_row_by_date(pred, d, "predictions")

        d_pred_mamba = self._extract_dpred(row_pred)
        env_vec = self._extract_envelope_features(env, d)
        rsi, macd, gap_pct = self._extract_technical_features(row_features)
        open_, high, low, close, volume = self._extract_price_features(row_prices)
        pnl_prev, action_prev = self._extract_prev_history(daily_log, d)

        vec = np.array(
            [
                # 3AI ΔClose
                d_pred_mamba,
                0.0,
                0.0,
                # Envelope 11
                *env_vec.tolist(),
                # Technical 3
                rsi,
                macd,
                gap_pct,
                # Price 5
                open_,
                high,
                low,
                close,
                volume,
                # History 2
                pnl_prev,
                action_prev,
            ],
            dtype=np.float64,
        )

        if vec.shape != (24,):
            raise RuntimeError(f"State vector must be 24-dim, got {vec.shape}")

        return vec

    def build_close_state(self, symbol: str, trading_date: date) -> np.ndarray:
        """
        引け前に使用する観測ベクトルを構築する（v1は朝と同じ）。
        """
        return self.build_morning_state(symbol, trading_date)

    # =====================================================
    # Loading CSVs
    # =====================================================
    def _mode_candidates(self) -> List[str]:
        if self.mode:
            return [self.mode]
        return ["live", "ops", "sim", "display"]

    def _load_prices(self, symbol: str) -> pd.DataFrame:
        """
        StepA prices を読み込む。

        優先:
        - output_root/stepA/**/stepA_prices_*_{symbol}.csv（train/test/live など）を全結合
        - legacy: output_root/stepA_prices_{symbol}.csv
        """
        if symbol in self._prices_cache:
            return self._prices_cache[symbol]

        root = self.output_root
        stepA_dir = root / "stepA"

        candidates: List[Path] = []
        if stepA_dir.exists():
            candidates.extend(sorted(stepA_dir.rglob(f"stepA_prices_*_{symbol}.csv")))

        legacy = root / f"stepA_prices_{symbol}.csv"
        if legacy.exists():
            candidates.append(legacy)

        if not candidates:
            raise FileNotFoundError(
                f"StepA prices CSV not found for symbol={symbol}. "
                f"Searched: {stepA_dir}/**/stepA_prices_*_{symbol}.csv and legacy={legacy}"
            )

        frames: List[pd.DataFrame] = []
        for p in candidates:
            df = pd.read_csv(p)
            df = self._parse_date_col(df, "Date")
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        self._prices_cache[symbol] = df_all
        return df_all

    def _load_features(self, symbol: str) -> pd.DataFrame:
        """
        StepA tech を読み込む。Regime Router が BNF_* 列を参照する。

        優先:
        - output_root/stepA/**/stepA_tech_*_{symbol}.csv（train/test/live など）を全結合
        - legacy: output_root/stepA_features_{symbol}.csv（旧仕様）
        """
        if symbol in self._features_cache:
            return self._features_cache[symbol]

        root = self.output_root
        stepA_dir = root / "stepA"

        candidates: List[Path] = []
        if stepA_dir.exists():
            candidates.extend(sorted(stepA_dir.rglob(f"stepA_tech_*_{symbol}.csv")))

        legacy = root / f"stepA_features_{symbol}.csv"
        if legacy.exists():
            candidates.append(legacy)

        if not candidates:
            raise FileNotFoundError(
                f"StepA tech CSV not found for symbol={symbol}. "
                f"Searched: {stepA_dir}/**/stepA_tech_*_{symbol}.csv and legacy={legacy}"
            )

        frames: List[pd.DataFrame] = []
        for p in candidates:
            df = pd.read_csv(p)
            df = self._parse_date_col(df, "Date")
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        self._features_cache[symbol] = df_all
        return df_all

    def _resolve_stepC_pred_path(self, symbol: str) -> Path:
        root = self.output_root

        # legacy
        legacy = root / f"stepC_pred_time_all_{symbol}.csv"
        if legacy.exists():
            return legacy

        # mode-separated
        for m in self._mode_candidates():
            p = root / "stepC" / m / f"stepC_pred_time_all_{symbol}.csv"
            if p.exists():
                return p

        # fallback: search anywhere under stepC
        stepC_dir = root / "stepC"
        if stepC_dir.exists():
            hits = sorted(stepC_dir.rglob(f"stepC_pred_time_all_{symbol}.csv"))
            if hits:
                return hits[0]

        raise FileNotFoundError(
            f"Prediction CSV not found for symbol={symbol}. "
            f"Tried legacy={legacy} and {root}/stepC/<mode>/stepC_pred_time_all_{symbol}.csv"
        )

    def _load_predictions(self, symbol: str) -> pd.DataFrame:
        """
        StepC の予測CSVを読み込む。必要列:
        - Date
        - Pred_Close_MAMBA（無ければ 0 として扱う）
        さらに dPred_* を diff で生成する。
        """
        if symbol in self._pred_cache:
            return self._pred_cache[symbol]

        path = self._resolve_stepC_pred_path(symbol)
        df = pd.read_csv(path)
        df = self._parse_date_col(df, "Date")
        df = df.sort_values("Date").reset_index(drop=True)

        if "Pred_Close_MAMBA" in df.columns:
            df["dPred_MAMBA"] = df["Pred_Close_MAMBA"].diff()
        else:
            df["dPred_MAMBA"] = 0.0

        self._pred_cache[symbol] = df
        return df

    def _resolve_stepD_events_path(self, agent: str, symbol: str) -> Path:
        root = self.output_root

        # legacy
        legacy = root / f"stepD_events_{agent}_{symbol}.csv"
        if legacy.exists():
            return legacy

        for m in self._mode_candidates():
            p = root / "stepD" / m / f"stepD_events_{agent}_{symbol}.csv"
            if p.exists():
                return p

        stepD_dir = root / "stepD"
        if stepD_dir.exists():
            hits = sorted(stepD_dir.rglob(f"stepD_events_{agent}_{symbol}.csv"))
            if hits:
                return hits[0]

        raise FileNotFoundError(
            f"StepD events CSV not found for agent={agent}, symbol={symbol}. "
            f"Tried legacy={legacy} and {root}/stepD/<mode>/stepD_events_{agent}_{symbol}.csv"
        )

    def _load_envelope(self, agent: str, symbol: str) -> pd.DataFrame:
        """
        Envelope のイベントCSVを読み込む。

        想定:
        - start_date / end_date があり、d がその範囲に入るイベントを選ぶ
        - 11特徴量のカラムが存在すれば使用（無い列は 0）
        """
        key = (agent, symbol)
        if key in self._env_cache:
            return self._env_cache[key]

        path = self._resolve_stepD_events_path(agent, symbol)
        df = pd.read_csv(path)

        # start_date / end_date を date に
        if "start_date" not in df.columns or "end_date" not in df.columns:
            raise ValueError(f"Envelope CSV missing start_date/end_date: {path}")

        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.date
        df = df.sort_values(["start_date", "end_date"]).reset_index(drop=True)

        self._env_cache[key] = df
        return df

    def _load_daily_log(self, symbol: str) -> pd.DataFrame:
        """
        日次ログ（存在すれば）を読み込む。無ければ空DFを返す。
        """
        if symbol in self._log_cache:
            return self._log_cache[symbol]

        path = Path(self.daily_log_pattern.format(symbol=symbol))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if not path.exists():
            df = pd.DataFrame(columns=["Date", "PnL_pct", "Action"])
            self._log_cache[symbol] = df
            return df

        df = pd.read_csv(path)
        if "Date" in df.columns:
            df = self._parse_date_col(df, "Date")
        else:
            raise ValueError(f"Daily log missing Date column: {path}")

        df = df.sort_values("Date").reset_index(drop=True)
        self._log_cache[symbol] = df
        return df

    # =====================================================
    # Extractors
    # =====================================================
    @staticmethod
    def _extract_dpred(row_pred: pd.Series) -> float:
        def _g(col: str) -> float:
            v = row_pred.get(col, 0.0)
            return float(v) if pd.notna(v) else 0.0

        return _g("dPred_MAMBA")

    @staticmethod
    def _extract_envelope_features(df_env: pd.DataFrame, d: date) -> np.ndarray:
        """
        trading_date が属する Envelope イベントを探して 11次元を返す。
        見つからない場合は 0 ベクトル。
        """
        if df_env.empty:
            return np.zeros(11, dtype=np.float64)

        mask = (df_env["start_date"] <= d) & (df_env["end_date"] >= d)
        if not mask.any():
            return np.zeros(11, dtype=np.float64)

        row = df_env.loc[mask].iloc[0]
        cols = [
            "DeltaP_pct",
            "theta_norm",
            "Top_pct",
            "Bottom_pct",
            "D_norm",
            "DeltaP",
            "D",
            "L",
            "theta_deg",
            "Top_abs",
            "Bottom_abs",
        ]
        vec: List[float] = []
        for c in cols:
            v = row.get(c, 0.0)
            vec.append(float(v) if pd.notna(v) else 0.0)
        return np.asarray(vec, dtype=np.float64)

    @staticmethod
    def _extract_technical_features(row_features: pd.Series) -> Tuple[float, float, float]:
        """
        RSI / MACD / Gap% を取り出す。無ければ 0。
        """
        def _g(col: str) -> float:
            v = row_features.get(col, 0.0)
            return float(v) if pd.notna(v) else 0.0

        rsi = _g("RSI")
        macd = _g("MACD")
        gap_pct = _g("Gap_pct") if "Gap_pct" in row_features.index else _g("Gap")
        return rsi, macd, gap_pct

    @staticmethod
    def _extract_price_features(row_prices: pd.Series) -> Tuple[float, float, float, float, float]:
        """
        Open / High / Low / Close / Volume を取り出す。無ければ 0。
        """
        def _g(col: str) -> float:
            v = row_prices.get(col, 0.0)
            return float(v) if pd.notna(v) else 0.0

        return _g("Open"), _g("High"), _g("Low"), _g("Close"), _g("Volume")

    @staticmethod
    def _extract_prev_history(df_log: pd.DataFrame, d: date) -> Tuple[float, float]:
        """
        日次ログから trading_date より前の最新行を取り、PnL% と Action を返す。
        """
        if df_log.empty:
            return 0.0, 0.0

        mask = df_log["Date"].dt.date < d
        if not mask.any():
            return 0.0, 0.0

        row = df_log.loc[mask].iloc[-1]
        pnl_prev = row.get("PnL_pct", 0.0)
        act_prev = row.get("Action", 0.0)
        pnl_prev = float(pnl_prev) if pd.notna(pnl_prev) else 0.0
        act_prev = float(act_prev) if pd.notna(act_prev) else 0.0
        return pnl_prev, act_prev

    # =====================================================
    # Utilities
    # =====================================================
    @staticmethod
    def _to_date(d: Union[date, datetime, str]) -> date:
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        raise TypeError(f"Unsupported date type: {type(d)}")

    @staticmethod
    def _parse_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f"Missing date column '{col}'")
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df[df[col].isna()].head(3)
            raise ValueError(f"Failed to parse some dates in '{col}':\n{bad}")
        return df

    @staticmethod
    def _get_row_by_date(df: pd.DataFrame, d: date, kind: str) -> pd.Series:
        if df.empty:
            raise ValueError(f"{kind} DataFrame is empty")

        if "Date" not in df.columns:
            raise ValueError(f"{kind} DataFrame missing Date column")

        mask = df["Date"].dt.date == d
        if not mask.any():
            raise KeyError(f"{kind} row for date {d} not found")
        return df.loc[mask].iloc[0]
