from __future__ import annotations

"""ai_core.gui._tab_utils

Tkinter ベースの TabA〜TabF に共通の小ヘルパ関数を 1 箇所に集約するモジュール。

目的
- build_global_symbols の重複検出で頻出する private helper の重複を、実害なく整理する。
- Tab ごとに微妙に異なる実装差（空文字/None など）を吸収して堅牢化する。

対象の重複関数
- _parse_ymd: 'YYYY-MM-DD' 文字列 → datetime へ変換
- _ensure_datetime: pandas Series を datetime 型に整形
- _numeric_columns: DataFrame から数値列名を抽出

注意
- ここで定義している関数名は既存 Tab の実装と合わせてあるため、
  Tab 側は「def を持たず import して使う」だけで移行できる。
"""

from datetime import datetime
from typing import Optional, List

import pandas as pd


def _parse_ymd(s: str) -> Optional[datetime]:
    """'YYYY-MM-DD' 形式の文字列を datetime に変換する。失敗時は None を返す。"""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Series を datetime64 系に揃える（変換不能は NaT）。"""
    if not pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return series


def _numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """DataFrame から numeric dtype の列名だけを返す。"""
    exclude = exclude or []
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
