# ai_core/rl/types.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RLRunResult:
    """
    RL / MARL 実行の結果をまとめた共通構造体。

    StepE（単体RL） / StepF（MARL）の両方で同じ構造を使う想定。

    各アルゴリズム側（RLSingleAlgo / MARLAlgo）は、
    学習＋評価の完了時に RLRunResult を返す。

    Attributes
    ----------
    success : bool
        学習・評価が正常終了したかどうか。
        True : 少なくとも1エピソード分の評価が行えた / ログが生成された
        False: 何らかの理由で失敗（例: ログが空、例外発生など）

    message : str
        実行結果メッセージ（ログ／GUI表示用）。
        例: "StepE single training finished for agent=xsr"

    final_return : float
        最終リターン（初期資産を1とした場合の (Equity_final / Equity_initial - 1)）。

    max_drawdown : float
        最大ドローダウン（Equityベース、負の値）。

    sharpe_ratio : float
        シャープレシオ（年率換算を想定、実装側の定義に依存）。

    bh_final_return : Optional[float]
        Buy & Hold ベンチマークの最終リターン。
        単体RL（StepE）では None または 0.0 が入り得る。
        MARL（StepF）では Buy & Hold の最終リターンが格納される想定。

    policy_path : Path
        学習済みポリシーファイルのパス。
        例: output/policies/policy_stepE_XSR_SOXL.npz

    equity_curve_path : Path
        Equity カーブ CSV のパス。
        例: output/stepE_equity_XSR_SOXL.csv

    daily_log_path : Path
        日次ログ CSV のパス。
        例: output/daily_log_XSR_SOXL.csv

    metrics_csv_path : Path
        主要メトリクス（final_return, max_drawdown, sharpe_ratio, ...）をまとめた
        CSV のパス。
        例: output/stepE_rl_metrics_XSR_SOXL.csv
    """

    # 実行ステータス
    success: bool
    message: str

    # 指標
    final_return: float
    max_drawdown: float
    sharpe_ratio: float
    bh_final_return: Optional[float]

    # ファイルパス
    policy_path: Path
    equity_curve_path: Path
    daily_log_path: Path
    metrics_csv_path: Path
