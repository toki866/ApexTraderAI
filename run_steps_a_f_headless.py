# -*- coding: utf-8 -*-
r"""
run_steps_a_f_headless.py  (SAFE v2: unicodeescape を根本回避)

目的
- soxl_rl_gui リポジトリで StepA〜StepF を「順番に」実行して、どこで落ちるかを 1 本のコマンドで再現できるようにする。
- あなたの「自動デバッグ（autodebug）」は基本的に「あるコマンドを実行→落ちたら修正→再実行」の反復なので、
  “A〜F をまとめて叩ける単一コマンド” を用意すると最短で回せます。

重要（環境）
- PySide6 を import するコードが混ざる可能性があるため、Anaconda Prompt では soxl_gui 環境で実行してください。
  (base) だと QtCore DLL エラーが出ることがあります。

UnicodeEscape エラー対策
- Windows のパス例（例: C:\Users\...）を docstring に書くと、"\U" が Unicode エスケープとして解釈されて
  SyntaxError: (unicode error) 'unicodeescape' ... が起きることがあります。
- このファイルは「raw 文字列 docstring（先頭に r を付けた docstring）」にしているため、上記問題を根本回避します。

使い方（Anaconda Prompt / cmd.exe）
1) conda activate soxl_gui
2) cd /d "C:\Users\ss-to\OneDrive\デスクトップ\Python\soxl_rl_gui"
3) python run_steps_a_f_headless.py --symbol SOXL --test-start 2022-01-03

オプション
- --skip A,B,C,D,E,F  : 実行しないステップを指定（複数可）
- --dry-run           : import とシグネチャ確認だけ（実行しない）
- --output-dir        : 期待する出力ディレクトリ（デフォルト: output）
- --use-sb3           : StepE が SB3-PPO をサポートしていればそれを優先して呼ぶ

注意
- 本スクリプトは「現時点の実装の違い（run()の引数など）」に強いように、
  inspect.signature を使って “呼べそうな形” を自動で当てに行きます。
- それでも呼べない場合は、どの関数が何を要求しているかを分かる形で例外にして止めます。
  → その Traceback を貼ってくれれば、こちらで “最短で揃える修正” に入れます。

期待出力チェック
- StepA: output/stepA_prices_{sym}.csv, output/stepA_features_{sym}.csv
- StepB: output/stepB_pred_time_all_{sym}.csv（過去ログでこの名前を使っていたため）
  ※ 実際の命名が違う場合は、エラー内容を見てこのチェック側を合わせます。
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import sys
import traceback
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _print(msg: str) -> None:
    print(msg, flush=True)


def _repo_root_from_cwd() -> Path:
    """
    soxl_rl_gui のルートにいる前提。
    もし違っても、ai_core が見つかるディレクトリまで遡って探す。
    """
    cwd = Path.cwd().resolve()
    cur = cwd
    for _ in range(8):
        if (cur / "ai_core").exists():
            return cur
        cur = cur.parent
    return cwd


def _safe_import(module: str) -> Any:
    __import__(module)
    return sys.modules[module]


def _construct(cls: Any, available: Dict[str, Any]) -> Any:
    """
    クラスの __init__ シグネチャを見て、渡せる引数だけで生成を試す。
    """
    sig = inspect.signature(cls)
    kwargs: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if name in available:
            kwargs[name] = available[name]
    try:
        return cls(**kwargs)  # type: ignore[misc]
    except Exception:
        return cls()  # type: ignore[misc]


def _build_date_range(test_start: str, train_years: int, test_months: int) -> Tuple[str, str, str, str]:
    ts = pd.Timestamp(test_start)
    train_start = ts - pd.DateOffset(years=train_years)
    train_end = ts - pd.tseries.offsets.BDay(1)
    test_end = ts + pd.DateOffset(months=test_months) - pd.tseries.offsets.BDay(1)
    return (
        train_start.strftime("%Y-%m-%d"),
        pd.Timestamp(train_end).strftime("%Y-%m-%d"),
        ts.strftime("%Y-%m-%d"),
        pd.Timestamp(test_end).strftime("%Y-%m-%d"),
    )


def _instantiate_date_range(date_range_cls: Any, train_start: str, train_end: str, test_start: str, test_end: str) -> Any:
    """
    DateRange のフィールド名/シグネチャに合わせて組み立てる。
    """
    available = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "trainStart": train_start,
        "trainEnd": train_end,
        "testStart": test_start,
        "testEnd": test_end,
    }
    try:
        if is_dataclass(date_range_cls):
            field_names = [f.name for f in dataclasses.fields(date_range_cls)]
            kwargs = {k: available[k] for k in field_names if k in available}
            return date_range_cls(**kwargs)  # type: ignore[misc]
    except Exception:
        pass

    try:
        return _construct(date_range_cls, available)
    except Exception:
        return {"train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end}


def _method_candidates(obj: Any) -> List[str]:
    names: List[str] = []
    for n in ("run_all", "run", "execute", "process", "main"):
        if hasattr(obj, n) and callable(getattr(obj, n)):
            names.append(n)
    return names


def _try_call(obj: Any, method_name: str, provided: Dict[str, Any]) -> Any:
    fn = getattr(obj, method_name)
    sig = inspect.signature(fn)

    kwargs: Dict[str, Any] = {}

    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue

        if p.name in provided:
            kwargs[p.name] = provided[p.name]
            continue

        if p.name in ("sym", "ticker") and "symbol" in provided:
            kwargs[p.name] = provided["symbol"]
            continue
        if p.name in ("dr",) and "date_range" in provided:
            kwargs[p.name] = provided["date_range"]
            continue
        if p.name in ("config", "cfg") and "cfg" in provided:
            kwargs[p.name] = provided["cfg"]
            continue

        if p.default is p.empty:
            raise TypeError(f"Cannot satisfy required param '{p.name}' for {type(obj).__name__}.{method_name}{sig}")

    return fn(**kwargs)


def _import_service(mod_name: str, cls_name: str) -> Any:
    m = _safe_import(mod_name)
    if not hasattr(m, cls_name):
        raise AttributeError(f"{mod_name} has no {cls_name}")
    return getattr(m, cls_name)


def _maybe_import(mod_name: str, name: str) -> Optional[Any]:
    try:
        m = _safe_import(mod_name)
        return getattr(m, name, None)
    except Exception:
        return None


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[Missing Output] {label}: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--test-start", default="2022-01-03")
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--skip", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-sb3", action="store_true")
    ap.add_argument("--config-yaml", default="")
    args = ap.parse_args(argv)

    repo_root = _repo_root_from_cwd()
    output_dir = (repo_root / args.output_dir).resolve()

    _print("=== run_steps_a_f_headless ===")
    _print(f"repo_root   = {repo_root}")
    _print(f"output_dir  = {output_dir}")
    _print(f"symbol      = {args.symbol}")
    _print(f"test_start  = {args.test_start} (train_years={args.train_years}, test_months={args.test_months})")
    _print(f"skip        = {args.skip!r}")
    _print(f"dry_run     = {args.dry_run}")
    _print(f"use_sb3     = {args.use_sb3}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    DateRange = _maybe_import("ai_core.types.common", "DateRange")
    if DateRange is None:
        raise ImportError("ai_core.types.common.DateRange が import できません。")

    train_start, train_end, test_start, test_end = _build_date_range(args.test_start, args.train_years, args.test_months)
    date_range_obj = _instantiate_date_range(DateRange, train_start, train_end, test_start, test_end)
    _print(f"date_range  = train={train_start}~{train_end}, test={test_start}~{test_end}")
    _print(f"DateRange   = {type(date_range_obj)}")

    AppConfig = _maybe_import("ai_core.config.app_config", "AppConfig")
    DummyAppConfig = _maybe_import("ai_core.config.dummy_app_config", "DummyAppConfig")

    app_config_obj: Any = None
    if DummyAppConfig is not None:
        try:
            app_config_obj = _construct(DummyAppConfig, {"repo_root": repo_root, "root_dir": repo_root, "base_dir": repo_root})
        except Exception:
            app_config_obj = DummyAppConfig()
        _print(f"AppConfig   = DummyAppConfig ({type(app_config_obj)})")
    elif AppConfig is not None:
        try:
            if args.config_yaml and hasattr(AppConfig, "from_yaml"):
                app_config_obj = AppConfig.from_yaml(args.config_yaml)  # type: ignore[attr-defined]
            else:
                app_config_obj = _construct(AppConfig, {"repo_root": repo_root, "root_dir": repo_root, "base_dir": repo_root})
            _print(f"AppConfig   = AppConfig ({type(app_config_obj)})")
        except Exception:
            _print("[WARN] AppConfig の生成に失敗。app_config_obj=None のまま進めます（サービスが必須ならそこで止まります）。")
            app_config_obj = None
    else:
        _print("[WARN] AppConfig/DummyAppConfig が見つかりません。app_config_obj=None のまま進めます。")

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    StepAService = _import_service("ai_core.services.step_a_service", "StepAService")
    StepBService = _import_service("ai_core.services.step_b_service", "StepBService")
    StepCService = _import_service("ai_core.services.step_c_service", "StepCService")
    StepDService = _import_service("ai_core.services.step_d_service", "StepDService")
    StepEService = _import_service("ai_core.services.step_e_service", "StepEService")
    StepFService = _import_service("ai_core.services.step_f_service", "StepFService")

    StepAConfig = _maybe_import("ai_core.services.step_a_service", "StepAConfig") or _maybe_import("ai_core.services.step_a_service", "Config")
    StepBConfig = _maybe_import("ai_core.services.step_b_service", "StepBConfig") or _maybe_import("ai_core.config.step_b_config", "StepBConfig")
    StepCConfig = _maybe_import("ai_core.services.step_c_service", "StepCConfig") or _maybe_import("ai_core.services.step_c_service", "Config")
    StepDConfig = _maybe_import("ai_core.services.step_d_service", "StepDConfig") or _maybe_import("ai_core.services.step_d_service", "Config")
    StepEConfig = _maybe_import("ai_core.services.step_e_service", "StepEConfig") or _maybe_import("ai_core.services.step_e_service", "Config")
    StepFConfig = _maybe_import("ai_core.services.step_f_service", "StepFConfig") or _maybe_import("ai_core.services.step_f_service", "Config")

    if args.dry_run:
        _print("=== DRY RUN: signatures ===")
        for name, cls in [
            ("StepAService", StepAService),
            ("StepBService", StepBService),
            ("StepCService", StepCService),
            ("StepDService", StepDService),
            ("StepEService", StepEService),
            ("StepFService", StepFService),
        ]:
            _print(f"{name}.__init__ = {inspect.signature(cls)}")
        return 0

    def make_service(cls: Any) -> Any:
        try:
            return _construct(cls, {"app_config": app_config_obj, "config": app_config_obj})
        except Exception:
            return cls(app_config_obj) if app_config_obj is not None else cls()

    stepA = make_service(StepAService)
    stepB = make_service(StepBService)
    stepC = make_service(StepCService)
    stepD = make_service(StepDService)
    stepE = make_service(StepEService)
    stepF = make_service(StepFService)

    def make_cfg(cfg_cls: Optional[Any]) -> Any:
        if cfg_cls is None:
            return None
        try:
            cfg = _construct(cfg_cls, {"symbol": args.symbol, "date_range": date_range_obj, "output_dir": output_dir})
            for k, v in (("symbol", args.symbol), ("date_range", date_range_obj)):
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        pass
            return cfg
        except Exception:
            return None

    cfgA = make_cfg(StepAConfig)
    cfgB = make_cfg(StepBConfig)
    cfgC = make_cfg(StepCConfig)
    cfgD = make_cfg(StepDConfig)
    cfgE = make_cfg(StepEConfig)
    cfgF = make_cfg(StepFConfig)

    base_provided = {
        "symbol": args.symbol,
        "date_range": date_range_obj,
        "app_config": app_config_obj,
        "config": app_config_obj,
        "output_dir": output_dir,
        "repo_root": repo_root,
    }

    def run_one(label: str, service: Any, cfg: Any, expect: List[Tuple[str, Path]]) -> Any:
        _print(f"\n--- {label} ---")
        _print(f"service = {type(service)}")
        if cfg is not None:
            _print(f"cfg     = {type(cfg)}")

        provided = dict(base_provided)
        provided["cfg"] = cfg
        provided["config"] = cfg

        if label == "StepE" and args.use_sb3 and hasattr(service, "run_sb3_ppo_split"):
            _print("calling: StepE.run_sb3_ppo_split(cfg)")
            res = service.run_sb3_ppo_split(cfg)  # type: ignore[attr-defined]
        else:
            last_err: Optional[BaseException] = None
            for m in _method_candidates(service):
                try:
                    _print(f"trying: {label}.{m}{inspect.signature(getattr(service, m))}")
                    res = _try_call(service, m, provided)
                    break
                except Exception as e:
                    last_err = e
                    continue
            else:
                raise RuntimeError(f"{label}: callable method not found or signature mismatch. last_err={last_err}")

        _print(f"{label} result type = {type(res)}")

        for exp_label, exp_path in expect:
            _ensure_exists(exp_path, exp_label)
            _print(f"[OK] {exp_label}: {exp_path}")

        return res

    exp_stepA = [
        ("stepA_prices", output_dir / f"stepA_prices_{args.symbol}.csv"),
        ("stepA_features", output_dir / f"stepA_features_{args.symbol}.csv"),
    ]
    exp_stepB = [
        ("stepB_pred_time_all", output_dir / f"stepB_pred_time_all_{args.symbol}.csv"),
    ]
    exp_none: List[Tuple[str, Path]] = []

    try:
        if "A" not in skip:
            run_one("StepA", stepA, cfgA, exp_stepA)
        if "B" not in skip:
            run_one("StepB", stepB, cfgB, exp_stepB)
        if "C" not in skip:
            run_one("StepC", stepC, cfgC, exp_none)
        if "D" not in skip:
            run_one("StepD", stepD, cfgD, exp_none)
        if "E" not in skip:
            run_one("StepE", stepE, cfgE, exp_none)
        if "F" not in skip:
            run_one("StepF", stepF, cfgF, exp_none)

        _print("\n=== ALL DONE (A~F) ===")
        return 0
    except Exception as e:
        _print("\n=== FAILED ===")
        _print(f"{type(e).__name__}: {e}")
        _print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
