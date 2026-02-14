# ai_core/config/rl_config.py

from __future__ import annotations

from dataclasses import dataclass, asdict, fields, field
from typing import Any, Dict, Optional


def _from_dict_helper(cls, data: Dict[str, Any]) -> Any:
    """dataclass 用の共通 from_dict ヘルパー。

    - 未指定フィールドにはデフォルト値を入れる
    - 余分なキーは無視する（YAML 互換・将来拡張のため）
    """
    default = cls()
    field_names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for name in field_names:
        if name in data:
            kwargs[name] = data[name]
        else:
            kwargs[name] = getattr(default, name)
    return cls(**kwargs)


# ==============================
# 共通 Env 設定
# ==============================


@dataclass
class EnvConfig:
    """RL 環境（単体 / MARL 共通）の設定。

    Notes
    -----
    以前の実装で EnvConfig が複数存在し、同名重複の原因になっていたため、
    本ファイルの EnvConfig を正本として統一します。

    Attributes
    ----------
    initial_cash : float
        初期資金（単位は任意。1.0 でも 1,000,000 でもOK）
    trading_cost_bp : float
        売買コスト（basis point）。例: 10 = 0.1%
    risk_penalty : float
        リスクペナルティ係数（reward から減算する係数）。0 なら無効。
        ※実損益ではないため、通常 equity には反映しない。
    max_leverage : float
        最大レバレッジ（ratio をクリップする閾値の目安）。とりあえず 1.0。
    reward_scale : float
        報酬スケール。reward = (pnl_rate - penalty)*reward_scale のように扱う。
    """

    initial_cash: float = 1_000_000.0
    trading_cost_bp: float = 10.0
    risk_penalty: float = 0.0
    max_leverage: float = 1.0
    reward_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
        return _from_dict_helper(cls, data)


# ==============================
# 単体 RL 用設定（StepE）
# ==============================


@dataclass
class RLSingleConfig:
    """単体 RL（StepE）用 PPO 系ハイパーパラメータ。

    Notes
    -----
    過去の AppConfig では `algo_name/max_epochs/batch_size(=2048)` のような
    別仕様が存在していました。

    「同名だけど中身が違う」重複を避けるため、この RLSingleConfig に
    互換フィールド（algo_name/max_epochs）を取り込み、
    既存 YAML / 既存 GUI がそのまま動くようにしています。

    主に SB3 PPO を想定:
    - total_timesteps, learning_rate, gamma, clip_range, n_steps, batch_size, gae_lambda, ...
    """

    # ---- 互換フィールド（旧YAML/旧GUI向け）----
    algo_name: str = "ppo"
    max_epochs: int = 50

    # ---- 推奨フィールド（新・正本）----
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2

    n_steps: int = 2048
    batch_size: int = 64
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLSingleConfig":
        return _from_dict_helper(cls, data)


# ==============================
# MARL / StepF 用設定
# ==============================


@dataclass
class RLMARLConfig(RLSingleConfig):
    """MARL（StepF）用設定。

    v1 では「本物の MARL 学習」よりも、
    3エージェント行動の統合・可視化や運用の器を先に固める用途が多い。

    互換のため、旧MARLConfigに存在した `initial_weights` も保持する。
    """

    algo_name: str = "marl_ppo"
    max_epochs: int = 50

    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "xsr": 1.0,
        "lstm": 0.0,
        "fed": 0.0,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLMARLConfig":
        return _from_dict_helper(cls, data)
