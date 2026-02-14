# ai_core/gui/rl_onclick.py

from __future__ import annotations

from datetime import date
from typing import Sequence

from ai_core.gui.rl_runner import run_step_e_and_step_f


def on_click_run_rl(
    *,
    symbol: str = "SOXL",
    train_start: date | None = None,
    train_end: date | None = None,
    test_start: date | None = None,
    test_end: date | None = None,
    agent_names: Sequence[str] = ("xsr", "lstm", "fed"),
    single_agent_for_step_e: str | None = None,
):
    """
    GUI のボタンから呼び出す想定のラッパ関数。

    - DateRange の生成
    - ダミー AppConfig の生成
    - StepE（単体RL）実行
    - StepF（MARLダミー）実行

    をすべてまとめてやってくれる。
    戻り値は run_step_e_and_step_f() の結果オブジェクト。
    """

    # デフォルト日付（何も指定されなかった場合）
    if train_start is None:
        train_start = date(2016, 1, 4)
    if train_end is None:
        train_end = date(2021, 12, 30)
    if test_start is None:
        test_start = date(2022, 1, 3)
    if test_end is None:
        test_end = date(2022, 3, 31)

    # StepE でメインにする単体エージェント（未指定なら先頭）
    if single_agent_for_step_e is None:
        single_agent_for_step_e = agent_names[0]

    # 実際の RL パイプラインを実行
    result = run_step_e_and_step_f(
        symbol=symbol,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        agent_names=tuple(agent_names),
        single_agent_for_step_e=single_agent_for_step_e,
    )

    return result


if __name__ == "__main__":
    # 単体テスト用: python -m ai_core.gui.rl_onclick でも動くようにしておく
    res = on_click_run_rl()

    print("[StepE]", res.step_e.success, res.step_e.message)
    print("  metrics:", res.step_e.metrics)
    print("  artifacts:", res.step_e.artifacts)

    print("[StepF]", res.step_f.success, res.step_f.message)
    print("  metrics:", res.step_f.metrics)
    print("  artifacts:", res.step_f.artifacts)
