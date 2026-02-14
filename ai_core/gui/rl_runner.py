# ai_core/gui/rl_runner.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, Tuple, Sequence

from ai_core.types.common import DateRange
from ai_core.config.dummy_app_config import create_dummy_app_config
from ai_core.services.step_e_service import StepEService, StepEConfig
from ai_core.services.step_f_service import StepFService, StepFConfig


@dataclass
class StepEResultView:
    success: bool
    message: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]


@dataclass
class StepFResultView:
    success: bool
    message: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]


@dataclass
class RLPipelineResult:
    """GUI から扱いやすいように StepE / StepF の結果をまとめたビュー"""
    step_e: StepEResultView
    step_f: StepFResultView


def run_step_e_and_step_f(
    symbol: str,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    agent_names: Sequence[str] = ("xsr", "lstm", "fed"),
    single_agent_for_step_e: str | None = None,
) -> RLPipelineResult:
    """
    GUI から呼び出すための統合パイプライン。

    - ダミー AppConfig を作成
    - StepE（単体 RL）を 1 エージェント分実行
    - StepF（MARL ダミー）を実行
    - 結果を RLPipelineResult にまとめて返す
    """

    if single_agent_for_step_e is None:
        # 特に指定がなければ先頭を使う（例: "xsr"）
        single_agent_for_step_e = agent_names[0]

    # 1) ダミー AppConfig を作成
    app_config = create_dummy_app_config(agent_names=tuple(agent_names))

    # 2) DateRange を構築
    dr = DateRange(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    # 3) StepE: 単体エージェント学習
    step_e = StepEService(app_config, StepEConfig(enabled_agents=list(agent_names)))
    res_single = step_e.train_single_model(symbol, dr, single_agent_for_step_e)

    step_e_view = StepEResultView(
        success=res_single.success,
        message=res_single.message,
        metrics=res_single.details.get("metrics", {}),
        artifacts=res_single.details.get("artifacts", {}),
    )

    # 4) StepF: MARL ダミー学習
    step_f = StepFService(app_config, StepFConfig(agent_names=tuple(agent_names)))
    res_marl = step_f.train_marl(symbol, dr)

    step_f_view = StepFResultView(
        success=res_marl.success,
        message=res_marl.message,
        metrics=res_marl.details.get("metrics", {}),
        artifacts=res_marl.details.get("artifacts", {}),
    )

    # 5) まとめて返す
    return RLPipelineResult(
        step_e=step_e_view,
        step_f=step_f_view,
    )
