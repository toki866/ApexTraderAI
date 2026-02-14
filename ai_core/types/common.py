from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DateRange:
    """学習期間・テスト期間をまとめて保持する構造体。

    Notes
    -----
    - 本プロジェクトでは「学習8年 + テスト3ヶ月」等の分割を頻繁に行うため、
      train/test の 4日付を 1つの構造体にまとめて扱う。
    - pandas.Timestamp ではなく datetime.date を採用（YAML/JSON との相性と、
      GUI入力（カレンダー）との相互変換を簡単にするため）。
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def to_dict(self) -> Dict[str, str]:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DateRange":
        def _parse(v: Any) -> date:
            if isinstance(v, date):
                return v
            if v is None:
                raise ValueError("DateRange.from_dict: date value is None")
            # "YYYY-MM-DD" を想定
            return date.fromisoformat(str(v))

        return cls(
            train_start=_parse(d["train_start"]),
            train_end=_parse(d["train_end"]),
            test_start=_parse(d["test_start"]),
            test_end=_parse(d["test_end"]),
        )


@dataclass
class StepResult:
    """各 StepA〜F の処理結果のベースクラス。

    このクラスは過去の実装で複数のバリエーションが存在しており、
    その差異が「同名だけど中身が違う」重複を生んでいました。

    本ファイルでは **1本化（正本）** として以下を標準化します。

    - success / message : 成否とメッセージ
    - metrics           : 数値指標（勝率、DD、Sharpe 等）
    - artifacts         : CSV/モデル/ログなどの出力パス
    - details           : 追加情報（任意の構造を許容）

    既存コードが `.metrics` / `.artifacts` を参照しても落ちないように、
    互換フィールドとして保持します。
    """

    success: bool
    message: str = ""

    # 互換フィールド（rl_benchmark 等が参照する）
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    # 任意詳細
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "message": str(self.message),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "details": dict(self.details),
        }

    def set_artifact(self, key: str, path: str) -> None:
        self.artifacts[key] = path

    def set_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def set_detail(self, key: str, value: Any) -> None:
        self.details[key] = value
