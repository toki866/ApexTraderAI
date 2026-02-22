from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ai_core.utils.paths import resolve_repo_path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import numpy as np
import yaml

ActionType = Literal["ratio", "discrete"]
LivePolicyMode = Literal["single", "marl", "router"]

logger = logging.getLogger(__name__)


# =====================================
# LivePolicyConfig（単体RL / MARL / Router 共通）
# =====================================

@dataclass
class LivePolicyConfig:
    """
    ライブ運用用の RL ポリシー設定（単体RL / MARL / Router 共通）。

    1) 単体RL（single）の最小構成:
        LivePolicyConfig(
            symbol="SOXL",
            mode="single",
            policy_path=str(resolve_repo_path("output/policies/policy_stepE_MAMBA_SOXL.npz")),
            agent_names=["mamba"],
        )

    2) MARL（marl）の例:
        LivePolicyConfig(
            symbol="SOXL",
            mode="marl",
            agent_names=["mamba"],
            policy_paths={
                "mamba": str(resolve_repo_path("output/policies/policy_stepE_MAMBA_SOXL.npz")),
            },
            marl_weights={"mamba": 1.0},
        )

    3) Regime Router（router）の例:
        LivePolicyConfig(
            symbol="SOXL",
            mode="router",
            agent_names=["dprime_bnf_h01", "dprime_mix_h01", "dprime_all_features_h01"],
            policy_paths={
                "dprime_bnf_h01": str(resolve_repo_path("output/policies/policy_stepE_dprime_bnf_h01_SOXL.npz")),
                "dprime_mix_h01": str(resolve_repo_path("output/policies/policy_stepE_dprime_mix_h01_SOXL.npz")),
                "dprime_all_features_h01": str(resolve_repo_path("output/policies/policy_stepE_dprime_all_features_h01_SOXL.npz")),
            },
            router_table_path=str(resolve_repo_path("output/stepF/sim/router_table_SOXL.yaml")),
            router_log_path=str(resolve_repo_path("output/engine/router_log_SOXL.csv")),
        )

    Router table（YAML）の確定仕様（version=1）
    -----------------------------------------
    version: 1
    symbol: SOXL
    hysteresis:
      min_hold_days: 2            # 任意（cfg.router_min_hold_days があればそちら優先）
    regimes:
      DivDownVolUp:
        primary_agent: dprime_bnf_h01
      EnergyFade:
        primary_agent: dprime_mix_h01
      Normal:
        primary_agent: dprime_all_features_h01

    注意:
    - エンジン側が参照するのは「regimes -> <regime> -> primary_agent」と
      「hysteresis -> min_hold_days（任意）」のみです。
    """

    symbol: str
    mode: LivePolicyMode = "single"

    policy_path: Optional[Union[str, Path]] = None
    policy_paths: Optional[Dict[str, Union[str, Path]]] = None
    agent_names: Optional[List[str]] = None
    marl_weights: Optional[Dict[str, float]] = None

    action_type: ActionType = "ratio"
    obs_dim: int = 24
    device: str = "cpu"

    # 既存の decision log
    log_decisions: bool = False
    decisions_log_path: Optional[Union[str, Path]] = None

    # 離散アクション用マップ（必要なら）
    discrete_action_map: Optional[Dict[int, float]] = None

    # Router mode（regime router）
    router_table_path: Optional[Union[str, Path]] = None
    router_min_hold_days: Optional[int] = None
    router_log_path: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
        # policy_path(s) を Path に正規化
        if self.policy_path is not None:
            self.policy_path = Path(self.policy_path).resolve()

        if self.policy_paths is not None:
            norm_paths: Dict[str, Path] = {}
            for name, p in self.policy_paths.items():
                norm_paths[name] = Path(p).resolve()
            self.policy_paths = norm_paths

        if self.decisions_log_path is not None:
            self.decisions_log_path = Path(self.decisions_log_path).resolve()

        if self.router_table_path is not None:
            self.router_table_path = Path(self.router_table_path).resolve()

        if self.router_log_path is not None:
            self.router_log_path = Path(self.router_log_path).resolve()

        # policy_paths が指定されていない場合は、単体RL用 policy_path を使って構築
        if self.policy_paths is None:
            if self.policy_path is None:
                raise ValueError("Either policy_path or policy_paths must be provided.")
            # agent_names が指定されていればその1つ目を使う。無ければ "agent" をデフォルトに。
            if self.agent_names and len(self.agent_names) > 0:
                name = self.agent_names[0]
            else:
                name = "agent"
                self.agent_names = [name]
            self.policy_paths = {name: Path(self.policy_path)}
        else:
            # policy_paths がある場合、agent_names が None ならキーから自動決定
            if self.agent_names is None:
                self.agent_names = sorted(self.policy_paths.keys())

        assert self.policy_paths is not None
        assert self.agent_names is not None

        # mode ごとの整形
        if self.mode == "single":
            if not self.agent_names:
                raise ValueError("agent_names must not be empty for single mode.")
            first = self.agent_names[0]
            if first not in self.policy_paths:
                raise ValueError(f"policy_paths must contain '{first}' for single mode.")
            self.policy_paths = {first: self.policy_paths[first]}
            self.agent_names = [first]

        elif self.mode == "marl":
            if len(self.agent_names) < 2:
                raise ValueError("MARL mode requires at least 2 agent_names.")
            # 重みが無ければ一様重み
            if self.marl_weights is None:
                w = 1.0 / float(len(self.agent_names))
                self.marl_weights = {n: w for n in self.agent_names}
            else:
                # 欠けていたらエラー
                for n in self.agent_names:
                    if n not in self.marl_weights:
                        raise ValueError(f"marl_weights missing key: {n}")

        elif self.mode == "router":
            if not self.router_table_path:
                raise ValueError("router mode requires router_table_path.")
            # router は weights 不要
            self.marl_weights = None

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =====================================
# Policy ラッパー
# =====================================

class BasePolicyWrapper:
    """
    観測ベクトル obs (np.ndarray, shape=(obs_dim,)) を入力し、action を返すインタフェース。
    戻り値の action の意味は LivePolicyRunner 側の action_type に依存。
    """

    def predict(self, obs: np.ndarray) -> float:
        raise NotImplementedError


class NpzLinearPolicyWrapper(BasePolicyWrapper):
    """
    .npz ファイルに保存された線形＋tanh ポリシー。

    期待するフォーマット:
        - W: (obs_dim,) もしくは (1, obs_dim) の numpy 配列
        - b: スカラー or 形状 (1,) の numpy 配列（オプション）

    アクション:
        a = tanh(W · obs + b)  --> [-1, +1] の連続値
    """

    def __init__(self, path: Path, obs_dim: int, logger: Optional[logging.Logger] = None) -> None:
        self.path = Path(path)
        self.obs_dim = obs_dim
        self.logger = logger or logging.getLogger(__name__)

        if not self.path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.path}")

        data = np.load(self.path)
        if "W" not in data:
            raise KeyError(f"npz policy must contain 'W' array: {self.path}")

        self.W = np.asarray(data["W"], dtype=np.float64).reshape(-1)
        if self.W.shape[0] != self.obs_dim:
            raise ValueError(
                f"npz policy W dim mismatch: expected {self.obs_dim}, got {self.W.shape[0]}"
            )

        self.b = 0.0
        if "b" in data:
            self.b = float(np.asarray(data["b"]).reshape(-1)[0])

    def predict(self, obs: np.ndarray) -> float:
        z = float(np.dot(self.W, obs) + self.b)
        return float(np.tanh(z))


class SB3PPOPolicyWrapper(BasePolicyWrapper):
    """
    Stable-Baselines3 の PPO モデルをラップ。

    期待するフォーマット:
        - policy_path: *.zip (model.save の出力)

    注意:
        - stable_baselines3 がインストールされていない場合は ImportError を投げる。
        - obs は (obs_dim,) の numpy 配列を受け取り、
          model.predict(obs[None, :], deterministic=True) を呼び出す。
    """

    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = Path(path)
        self.logger = logger or logging.getLogger(__name__)

        if not self.path.exists():
            raise FileNotFoundError(f"SB3 model not found: {self.path}")

        try:
            from stable_baselines3 import PPO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "stable_baselines3 is required to load .zip PPO models. "
                "Install with: pip install stable-baselines3"
            ) from e

        self.model = PPO.load(str(self.path), device="cpu")

    def predict(self, obs: np.ndarray) -> float:
        a, _ = self.model.predict(obs[None, :], deterministic=True)
        # 連続なら shape(1,1) or (1,), 離散なら int
        if np.isscalar(a):
            return float(a)
        a = np.asarray(a).reshape(-1)
        return float(a[0])


class TorchPolicyWrapper(BasePolicyWrapper):
    """
    PyTorch の .pt / .pth をロードして推論するラッパー。

    対応する保存形式（StepE で実際に使っている形式を優先）:
      A) torch.save(nn.Module, path) の「モジュールそのもの」
      B) {"policy": nn.Module} / {"model": nn.Module} 等、dict に nn.Module が入っている
      C) StepE(diffpg) が保存する dict 形式:
         {
           "cfg": {...},
           "mu":  Tensor | list[Tensor] | tuple[Tensor],   # 重み（層ごと）
           "b":   Tensor | list[Tensor] | tuple[Tensor],   # バイアス（層ごと）
           ...（他のメタ情報）
         }
         ※この形式は nn.Module を含まないため、ここで前向き計算を実装する。

    入力:
      - obs: (obs_dim,) の numpy 配列

    出力:
      - 連続 action（ratio）として float を返す（[-1,1] を想定）
    """

    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = Path(path)
        self.logger = logger or logging.getLogger(__name__)

        if not self.path.exists():
            raise FileNotFoundError(f"Torch model not found: {self.path}")

        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("PyTorch is required to load .pt/.pth policies.") from e

        self._torch = torch
        self.kind: str = "module"
        self.cfg: Dict[str, Any] = {}

        # PyTorch 2.6+ では torch.load の既定が weights_only=True に変更されたため、
        # 自リポジトリ生成物（信頼できるチェックポイント）をロードする場合は weights_only=False を明示する。
        try:
            obj = torch.load(str(self.path), map_location="cpu", weights_only=False)
        except TypeError:
            # 旧バージョン互換（weights_only 引数が無い場合）
            obj = torch.load(str(self.path), map_location="cpu")

        # --- StepE(diffpg) dict 形式（nn.Moduleなし）を先に判定 ---
        if isinstance(obj, dict) and ("cfg" in obj) and ("mu" in obj) and ("b" in obj):
            self.kind = "diffpg_dict"
            try:
                self.cfg = dict(obj.get("cfg") or {})
            except Exception:
                self.cfg = {}

            self._mu = obj.get("mu")
            self._b = obj.get("b")

            # mu/b を「層ごとのリスト」に正規化（1層でもOK）
            def _to_list(x: Any) -> List[Any]:
                if isinstance(x, (list, tuple)):
                    return list(x)
                return [x]

            mu_list = _to_list(self._mu)
            b_list = _to_list(self._b)

            # b が 1個しか無いのに mu が複数、等の事故に備え、短い方を末尾で埋める
            if len(b_list) < len(mu_list) and len(b_list) > 0:
                b_list = b_list + [b_list[-1]] * (len(mu_list) - len(b_list))
            if len(mu_list) < len(b_list) and len(mu_list) > 0:
                mu_list = mu_list + [mu_list[-1]] * (len(b_list) - len(mu_list))

            # tensor化（cpu/float32）
            mu_t: List[torch.Tensor] = []
            b_t: List[torch.Tensor] = []
            for w in mu_list:
                if torch.is_tensor(w):
                    mu_t.append(w.detach().to("cpu"))
                else:
                    mu_t.append(torch.as_tensor(np.asarray(w), dtype=torch.float32, device="cpu"))
            for bb in b_list:
                if torch.is_tensor(bb):
                    b_t.append(bb.detach().to("cpu"))
                else:
                    b_t.append(torch.as_tensor(np.asarray(bb), dtype=torch.float32, device="cpu"))

            self.mu_layers = mu_t
            self.b_layers = b_t

            # 入力次元（推定）: 最初の重みから決定
            # w: (out, in) or (in,) の可能性がある
            w0 = self.mu_layers[0]
            if w0.ndim == 2:
                self.in_dim = int(w0.shape[1])
            elif w0.ndim == 1:
                self.in_dim = int(w0.shape[0])
            else:
                raise ValueError(f"Unsupported mu tensor ndim={w0.ndim} in {self.path}")

            # 出力は ratio 想定なので最後に tanh を適用する
            return

        # --- 一般的な nn.Module 形式 ---
        model = obj

        # dict の場合は、よくあるキーから nn.Module を探す
        if isinstance(model, dict):
            for k in ["policy", "model", "net", "actor", "pi", "module"]:
                if k in model:
                    model = model[k]
                    break

            # それでも dict なら（未知フォーマット）
            if isinstance(model, dict):
                # state_dict は通常 {str: Tensor}
                if all(isinstance(v, torch.Tensor) for v in model.values()):
                    raise ValueError(
                        f"Torch policy file looks like a state_dict only and cannot be used directly by engine: {self.path}"
                    )
                raise ValueError(
                    f"Torch policy file dict format is unsupported (no nn.Module found): {self.path}"
                )

        self.model = model
        # eval モードに
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            try:
                self.model.to("cpu")
            except Exception:
                pass

    def _extract_scalar(self, out: Any) -> float:
        torch = self._torch
        # tuple/list -> first
        if isinstance(out, (tuple, list)) and len(out) > 0:
            out = out[0]

        # dict -> common keys
        if isinstance(out, dict):
            for k in ["action", "a", "ratio", "mu", "mean", "pred"]:
                if k in out:
                    out = out[k]
                    break

        if torch.is_tensor(out):
            out = out.detach().cpu().numpy()

        out = np.asarray(out).reshape(-1)
        if out.size == 0:
            raise ValueError("Torch policy output is empty.")
        return float(out[0])

    def _pad_or_trunc(self, x: np.ndarray, n: int) -> np.ndarray:
        """
        入力次元が合わない場合の安全策（当面の動作確認用）。
        - x が短い: 末尾に 0 を pad
        - x が長い: 末尾を切り捨て
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == n:
            return x
        if x.size < n:
            pad = np.zeros((n - x.size,), dtype=np.float32)
            return np.concatenate([x, pad], axis=0)
        return x[:n]

    def predict(self, obs: np.ndarray) -> float:
        torch = self._torch

        # --- StepE(diffpg) dict 形式 ---
        if self.kind == "diffpg_dict":
            x = self._pad_or_trunc(obs, self.in_dim)
            x_t = torch.as_tensor(x, dtype=torch.float32, device="cpu").view(1, -1)

            # 前向き: Linear + tanh を層ごとに適用（最後も tanh で [-1,1]）
            h = x_t
            for i, (w, b) in enumerate(zip(self.mu_layers, self.b_layers)):
                if w.ndim == 1:
                    # (in,) -> dot
                    y = (h * w.view(1, -1)).sum(dim=1, keepdim=True)
                elif w.ndim == 2:
                    # (out, in)
                    y = h @ w.t()
                else:
                    raise ValueError(f"Unsupported weight ndim={w.ndim} in {self.path}")

                # bias
                if b is not None:
                    b2 = b
                    if b2.ndim == 0:
                        y = y + b2.view(1, 1)
                    elif b2.ndim == 1:
                        y = y + b2.view(1, -1)
                    else:
                        y = y + b2.view(1, -1)

                # activation: tanh
                h = torch.tanh(y)

            out = h
            return float(out.detach().cpu().numpy().reshape(-1)[0])

        # まずはモデル側が predict / act を提供していれば優先
        if hasattr(self.model, "predict") and callable(getattr(self.model, "predict")):
            try:
                out = self.model.predict(obs)
                return self._extract_scalar(out)
            except Exception:
                pass

        if hasattr(self.model, "act") and callable(getattr(self.model, "act")):
            try:
                out = self.model.act(obs)
                return self._extract_scalar(out)
            except Exception:
                pass

        # 通常 forward 呼び出し
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).view(1, -1)
        with torch.no_grad():
            out = self.model(obs_t)  # type: ignore
        return self._extract_scalar(out)


# =====================================
# Regime detection / routing
# =====================================

class RegimeDetector:
    """
    StepA（tech）にある BNF 列をそのまま使って局面を判定する（確定版）。

    参照する列名（StepA tech の実データに合わせて固定）
    - BNF_DivDownVolUp : 0/1（もしくは 0.0/1.0）
    - BNF_EnergyFade   : 0/1（もしくは 0.0/1.0）
    - BNF_PanicScore   : 連続値（補助、現状の regime 決定では閾値未使用）

    優先順位:
        DivDownVolUp > EnergyFade > Normal
    """

    COL_DIV_DOWN_VOL_UP = "BNF_DivDownVolUp"
    COL_ENERGY_FADE = "BNF_EnergyFade"
    COL_PANIC_SCORE = "BNF_PanicScore"

    def detect(self, tech_row: Mapping[str, Any]) -> str:
        try:
            div = float(tech_row[self.COL_DIV_DOWN_VOL_UP])
            fade = float(tech_row[self.COL_ENERGY_FADE])
        except KeyError as e:
            raise KeyError(
                f"RegimeDetector expected BNF columns "
                f"({self.COL_DIV_DOWN_VOL_UP}, {self.COL_ENERGY_FADE}) but missing: {e}"
            ) from e

        if div >= 0.5:
            return "DivDownVolUp"
        if fade >= 0.5:
            return "EnergyFade"
        return "Normal"


def _to_date(d: Union[date, datetime, str]) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        # ISO 8601 or YYYY-MM-DD
        return datetime.fromisoformat(d).date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _load_router_table(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"router_table not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("router_table must be a mapping")

    # schema validation (確定)
    version = data.get("version", None)
    if version != 1:
        raise ValueError(f"router_table version must be 1, got: {version}")

    regimes = data.get("regimes", None)
    if not isinstance(regimes, dict) or not regimes:
        raise ValueError("router_table.regimes must be a non-empty mapping")

    if "Normal" not in regimes:
        raise ValueError("router_table.regimes must include 'Normal'")

    for rname, rinfo in regimes.items():
        if not isinstance(rinfo, dict):
            raise ValueError(f"router_table.regimes.{rname} must be mapping")
        if "primary_agent" not in rinfo or not isinstance(rinfo["primary_agent"], str):
            raise ValueError(f"router_table.regimes.{rname}.primary_agent must be str")

    return data


class RegimeRouter:
    """
    regime -> agent を切り替えるルータ（ヒステリシス：min_hold_days）。
    """

    def __init__(self, router_table: Dict[str, Any], min_hold_days: int) -> None:
        self.router_table = router_table
        self.regimes: Dict[str, Dict[str, Any]] = router_table["regimes"]
        self.min_hold_days = int(min_hold_days)

        self._selected_agent: Optional[str] = None
        self._selected_regime: Optional[str] = None
        self._hold_days: int = 0
        self._last_trade_date: Optional[date] = None

    def desired_agent(self, regime: str) -> str:
        if regime in self.regimes:
            return str(self.regimes[regime]["primary_agent"])
        # unknown regime -> Normal
        return str(self.regimes["Normal"]["primary_agent"])

    def select(self, regime: str, trading_date: Union[date, datetime, str]) -> str:
        d = _to_date(trading_date)
        desired = self.desired_agent(regime)

        # 初回
        if self._selected_agent is None:
            self._selected_agent = desired
            self._selected_regime = regime
            self._hold_days = 1
            self._last_trade_date = d
            return self._selected_agent

        # 同日複数回呼ばれても hold_days を進めない
        if self._last_trade_date != d:
            self._hold_days += 1
            self._last_trade_date = d

        # 切り替え判定
        if desired != self._selected_agent:
            if self._hold_days >= max(1, self.min_hold_days):
                self._selected_agent = desired
                self._selected_regime = regime
                self._hold_days = 1

        return self._selected_agent

    @property
    def hold_days(self) -> int:
        return int(self._hold_days)

    @property
    def selected_agent(self) -> Optional[str]:
        return self._selected_agent

    @property
    def selected_regime(self) -> Optional[str]:
        return self._selected_regime


# =====================================
# LivePolicyRunner
# =====================================

class LivePolicyRunner:
    """
    学習済みRLポリシーを用いて、観測ベクトルからポジション比率を決定するクラス。

    単体RL / MARL / Regime Router 共通版。
    """

    def __init__(
        self,
        cfg: LivePolicyConfig,
        policies: Dict[str, BasePolicyWrapper],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.policies = policies
        self.logger = logger or logging.getLogger(__name__)

        self.mode: LivePolicyMode = cfg.mode
        self.agent_names: List[str] = list(cfg.agent_names or [])
        self.action_type: ActionType = cfg.action_type
        self.obs_dim: int = int(cfg.obs_dim)

        self._last_ratio: Optional[float] = None
        self._last_raw_actions: Dict[str, float] = {}
        self._last_actions: Dict[str, float] = {}

        # router state
        self._router_detector: Optional[RegimeDetector] = None
        self._router: Optional[RegimeRouter] = None
        self._router_last_info: Optional[Dict[str, Any]] = None

        # decision log init
        if self.cfg.log_decisions and self.cfg.decisions_log_path is not None:
            self._init_decision_log(self.cfg.decisions_log_path)

        # router log init
        if self.mode == "router" and self.cfg.router_log_path is not None:
            self._init_router_log(self.cfg.router_log_path)

        # router init
        if self.mode == "router":
            assert self.cfg.router_table_path is not None
            router_table = _load_router_table(Path(self.cfg.router_table_path))
            min_hold = self._resolve_min_hold_days(router_table)
            self._router_detector = RegimeDetector()
            self._router = RegimeRouter(router_table, min_hold_days=min_hold)

    @classmethod
    def from_config(cls, cfg: LivePolicyConfig) -> "LivePolicyRunner":
        """
        LivePolicyConfig から適切な PolicyWrapper 群を生成し、
        LivePolicyRunner インスタンスを構築する。
        """
        logger = logging.getLogger(__name__)

        if cfg.policy_paths is None or not cfg.agent_names:
            raise ValueError("policy_paths and agent_names must be set in LivePolicyConfig.")

        policies: Dict[str, BasePolicyWrapper] = {}
        for name in cfg.agent_names:
            path = Path(cfg.policy_paths[name])
            suffix = path.suffix.lower()
            if suffix == ".npz":
                policies[name] = NpzLinearPolicyWrapper(path, obs_dim=cfg.obs_dim, logger=logger)
            elif suffix == ".zip":
                policies[name] = SB3PPOPolicyWrapper(path, logger=logger)
            elif suffix in (".pt", ".pth"):
                policies[name] = TorchPolicyWrapper(path, logger=logger)
            else:
                raise ValueError(f"Unsupported policy format: {path}")

        return cls(cfg=cfg, policies=policies, logger=logger)

    # -------------------------
    # Public API
    # -------------------------

    def decide_position(
        self,
        obs: Union[np.ndarray, List[float]],
        *,
        trading_date: Optional[Union[date, datetime, str]] = None,
        tech_row: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """
        観測ベクトル obs からポジション比率を決定する（例外処理・クリップなし）。

        Router mode の場合:
            trading_date と tech_row が必須。

        Returns
        -------
        ratio : float
            ポジション比率。通常 [-1, +1] の範囲を想定。
            >0 : SOXL ロング、<0 : SOXS ロング、≈0 : ノーポジ など。
        """
        obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs_arr.shape[0] != self.obs_dim:
            raise ValueError(f"obs length mismatch: expected {self.obs_dim}, got {obs_arr.shape[0]}")

        if not np.all(np.isfinite(obs_arr)):
            raise ValueError("obs contains NaN or infinite values")

        if self.mode == "single":
            ratio = self._decide_single(obs_arr)
        elif self.mode == "marl":
            ratio = self._decide_marl(obs_arr)
        elif self.mode == "router":
            if trading_date is None or tech_row is None:
                raise ValueError("router mode requires trading_date and tech_row")
            ratio = self._decide_router(obs_arr, trading_date=trading_date, tech_row=tech_row)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._last_ratio = float(ratio)
        return float(ratio)

    def safe_decide_position(
        self,
        obs: Union[np.ndarray, List[float]],
        *,
        trading_date: Optional[Union[date, datetime, str]] = None,
        tech_row: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """
        decide_position の安全版。

        - obs の長さチェック
        - NaN / inf を 0.0 に置換
        - decide_position 内で例外が発生した場合は ratio=0.0 を返す
        - 最終的な ratio を [-1, +1] にクリップ
        """
        obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)

        if obs_arr.shape[0] != self.obs_dim:
            self.logger.error(
                "safe_decide_position: obs length mismatch: expected %d, got %d",
                self.obs_dim,
                obs_arr.shape[0],
            )
            return 0.0

        mask_finite = np.isfinite(obs_arr)
        if not np.all(mask_finite):
            self.logger.warning("safe_decide_position: obs contains non-finite values; replacing with 0.0")
            obs_arr[~mask_finite] = 0.0

        try:
            ratio = self.decide_position(obs_arr, trading_date=trading_date, tech_row=tech_row)
        except Exception as e:  # pragma: no cover
            self.logger.exception("safe_decide_position: error in decide_position: %s", e)
            ratio = 0.0

        ratio = float(np.clip(float(ratio), -1.0, 1.0))

        if self.cfg.log_decisions:
            self._log_decision(obs_arr, ratio)

        if self.mode == "router" and self.cfg.router_log_path is not None:
            self._log_router_decision(ratio_total=ratio)

        return ratio

    def get_last_ratio(self) -> Optional[float]:
        return self._last_ratio

    def get_last_router_info(self) -> Optional[Dict[str, Any]]:
        return self._router_last_info

    # -------------------------
    # Internal
    # -------------------------

    def _convert_raw_to_ratio(self, raw: float) -> float:
        if self.action_type == "ratio":
            return float(raw)

        if self.action_type == "discrete":
            if self.cfg.discrete_action_map is None:
                # デフォルト：3値 {-1, 0, +1}
                idx = int(round(raw))
                idx = int(np.clip(idx, 0, 2))
                return float([-1.0, 0.0, 1.0][idx])
            idx = int(round(raw))
            if idx not in self.cfg.discrete_action_map:
                raise ValueError(f"discrete_action_map missing idx={idx}")
            return float(self.cfg.discrete_action_map[idx])

        raise ValueError(f"Unknown action_type: {self.action_type}")

    def _decide_single(self, obs: np.ndarray) -> float:
        name = self.agent_names[0]
        policy = self.policies[name]

        raw = float(policy.predict(obs))
        ratio = self._convert_raw_to_ratio(raw)

        self._last_raw_actions = {name: raw}
        self._last_actions = {name: ratio}
        return ratio

    def _decide_marl(self, obs: np.ndarray) -> float:
        assert self.cfg.marl_weights is not None, "marl_weights must be set for MARL mode."

        raw_actions: Dict[str, float] = {}
        ratio_actions: Dict[str, float] = {}

        for name in self.agent_names:
            policy = self.policies[name]
            raw = float(policy.predict(obs))
            r = self._convert_raw_to_ratio(raw)
            raw_actions[name] = raw
            ratio_actions[name] = r

        weights = self.cfg.marl_weights
        w_sum = 0.0
        r_sum = 0.0
        for name in self.agent_names:
            w = float(weights[name])
            a = float(ratio_actions[name])
            w_sum += w
            r_sum += w * a

        ratio_total = r_sum / w_sum if w_sum != 0 else 0.0

        self._last_raw_actions = raw_actions
        self._last_actions = ratio_actions
        return float(ratio_total)

    def _decide_router(
        self,
        obs: np.ndarray,
        *,
        trading_date: Union[date, datetime, str],
        tech_row: Mapping[str, Any],
    ) -> float:
        assert self._router_detector is not None
        assert self._router is not None

        regime = self._router_detector.detect(tech_row)
        desired = self._router.desired_agent(regime)
        selected = self._router.select(regime, trading_date=trading_date)

        if selected not in self.policies:
            raise KeyError(f"Selected agent '{selected}' policy not loaded.")

        policy = self.policies[selected]
        raw = float(policy.predict(obs))
        ratio = self._convert_raw_to_ratio(raw)

        self._last_raw_actions = {selected: raw}
        self._last_actions = {selected: ratio}

        self._router_last_info = {
            "trade_date": _to_date(trading_date).isoformat(),
            "regime": regime,
            "desired_agent": desired,
            "selected_agent": selected,
            "hold_days": self._router.hold_days,
            "min_hold_days": self._router.min_hold_days,
        }
        return ratio

    def _resolve_min_hold_days(self, router_table: Dict[str, Any]) -> int:
        # cfg があれば優先
        if self.cfg.router_min_hold_days is not None:
            return int(self.cfg.router_min_hold_days)

        hy = router_table.get("hysteresis", {})
        if isinstance(hy, dict) and "min_hold_days" in hy:
            try:
                return int(hy["min_hold_days"])
            except Exception:
                pass
        return 2

    # -------------------------
    # Logging
    # -------------------------

    def _init_decision_log(self, path: Union[str, Path]) -> None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            return

        headers = ["datetime", "symbol", "mode", "obs_dim", "ratio_total"]
        for name in self.agent_names:
            headers.append(f"ratio_{name}")
        for name in self.agent_names:
            headers.append(f"raw_{name}")

        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        self.logger.info("Initialized decisions log: %s", log_path)

    def _log_decision(self, obs: np.ndarray, ratio_total: float) -> None:
        if self.cfg.decisions_log_path is None:
            return

        log_path = Path(self.cfg.decisions_log_path)
        now_str = datetime.now().isoformat(timespec="seconds")

        row = [now_str, self.cfg.symbol, self.mode, self.obs_dim, ratio_total]
        for name in self.agent_names:
            row.append(self._last_actions.get(name))
        for name in self.agent_names:
            row.append(self._last_raw_actions.get(name))

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _init_router_log(self, path: Union[str, Path]) -> None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            return

        headers = [
            "datetime",
            "symbol",
            "trade_date",
            "regime",
            "desired_agent",
            "selected_agent",
            "hold_days",
            "min_hold_days",
            "ratio_total",
        ]

        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        self.logger.info("Initialized router log: %s", log_path)

    def _log_router_decision(self, *, ratio_total: float) -> None:
        if self.cfg.router_log_path is None:
            return
        info = self._router_last_info
        if not info:
            return

        log_path = Path(self.cfg.router_log_path)
        now_str = datetime.now().isoformat(timespec="seconds")

        row = [
            now_str,
            self.cfg.symbol,
            info.get("trade_date"),
            info.get("regime"),
            info.get("desired_agent"),
            info.get("selected_agent"),
            info.get("hold_days"),
            info.get("min_hold_days"),
            ratio_total,
        ]

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
