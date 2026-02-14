from __future__ import annotations

"""
ai_core/backend/backend_controller.py

GUI（MainWindow / TabE / TabF など）と StepE / StepF サービスの橋渡しを行う
バックエンドコントローラ。

今回の修正ポイント（重要）
- 「循環import」を起こしていた自己import（例: from ai_core.backend.backend_controller import BackendController）を排除。
- package import の都合で ai_core.backend.__init__ が先に実行されても壊れないように、
  backend_controller.py 側は ai_core.backend（パッケージ）を参照しない。

役割
- QThreadPool + QRunnable（Worker）で StepE / StepF の長時間処理を非同期実行する
- 結果 / エラー / ログを Qt Signal 経由で GUI に返す
- サービスへの協調的キャンセル要求を転送する

補足
- `FullPipelineConfig` は「backend/__init__.py が re-export する」運用や、GUI側の互換のために定義している。
  このファイル単体では A→F 全自動パイプライン（start_full_pipeline）は実装しない。
  GUI側は hasattr でガードする想定（実装が揃ったら追加する）。
"""

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

# ---- 依存（存在しない可能性があるので保護） ----

try:
    from ai_core.config.app_config import AppConfig  # type: ignore
except Exception:  # pragma: no cover
    AppConfig = Any  # type: ignore

try:
    from ai_core.types.common import DateRange  # type: ignore
except Exception:  # pragma: no cover
    DateRange = Any  # type: ignore

try:
    from ai_core.services.step_e_service import StepEService, StepEConfig, StepEResult  # type: ignore
except Exception:  # pragma: no cover
    StepEService = None  # type: ignore
    StepEConfig = Any  # type: ignore
    StepEResult = Any  # type: ignore

try:
    from ai_core.services.step_f_service import StepFService, StepFConfig, StepFResult  # type: ignore
except Exception:  # pragma: no cover
    StepFService = None  # type: ignore
    StepFConfig = Any  # type: ignore
    StepFResult = Any  # type: ignore


# ---- 互換用（GUI側や __init__.py の re-export 用） ----

@dataclass(frozen=True)
class FullPipelineConfig:
    """
    将来の「A→F 全ステップ実行」用コンフィグの器（互換用）。

    backend_controller.py では start_full_pipeline を実装しないため、
    この dataclass は “型・運用の置き場所” としてのみ提供する。

    GUI側では `hasattr(backend, "start_full_pipeline")` でガードし、
    実装が揃ったタイミングで backend_controller に start_full_pipeline を追加すること。
    """
    symbol: str
    date_range: DateRange
    stepB_config: Any = None
    stepC_config: Any = None
    stepD_config: Any = None
    stepE_config: Any = None
    stepF_config: Any = None


# ---- Worker 実装 ----

class WorkerSignals(QObject):
    """
    Worker（QRunnable）からバックエンド / GUI へ結果・エラー・ログを送るためのシグナル集。
    """
    finished = Signal(object)  # StepEResult / StepFResult 等、任意
    error = Signal(str)        # traceback 文字列
    log = Signal(str)          # ログ文字列


class Worker(QRunnable):
    """
    任意の関数 fn(*args, **kwargs) を QThreadPool 上で実行する汎用 Worker。
    """
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ---- BackendController ----

class BackendController(QObject):
    """
    GUI 層から呼び出され、StepE / StepF サービスを非同期実行するコントローラ。
    """

    # 型互換のため Signal(object) を使う（PySide6 はジェネリクスに弱い）
    stepE_finished = Signal(object)
    stepF_finished = Signal(object)
    log_message = Signal(str)

    def __init__(self, app_config: AppConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._app_config = app_config
        self._thread_pool: QThreadPool = QThreadPool.globalInstance()

        self._symbol: Optional[str] = None
        self._date_range: Optional[DateRange] = None

        self._stepE_service: Any = None
        self._stepF_service: Any = None

        self._init_services()

    # --------------------
    # init
    # --------------------

    def _init_services(self) -> None:
        if StepEService is None:
            self._log("[Backend][Init] StepEService が import できません（未配置/依存不足）。")
        else:
            try:
                self._stepE_service = StepEService(self._app_config)  # type: ignore[misc]
                self._log("[Backend][Init] StepEService created.")
            except Exception:
                self._stepE_service = None
                self._log("[Backend][Init] StepEService の生成に失敗しました。\n" + traceback.format_exc())

        if StepFService is None:
            self._log("[Backend][Init] StepFService が import できません（未配置/依存不足）。")
        else:
            try:
                self._stepF_service = StepFService(self._app_config)  # type: ignore[misc]
                self._log("[Backend][Init] StepFService created.")
            except Exception:
                self._stepF_service = None
                self._log("[Backend][Init] StepFService の生成に失敗しました。\n" + traceback.format_exc())

    # --------------------
    # public API
    # --------------------

    def set_symbol_and_range(self, symbol: str, date_range: DateRange) -> None:
        """
        今後のジョブで使う symbol / date_range を設定する。
        """
        self._symbol = symbol
        self._date_range = date_range

        ts = getattr(date_range, "train_start", None)
        te = getattr(date_range, "train_end", None)
        ps = getattr(date_range, "test_start", None)
        pe = getattr(date_range, "test_end", None)
        if ts is not None and te is not None and ps is not None and pe is not None:
            self._log(f"[Backend] symbol={symbol}, train={ts}~{te}, test={ps}~{pe}")
        else:
            self._log(f"[Backend] symbol={symbol}, date_range={date_range}")

    def start_stepE_training(self, cfg: StepEConfig) -> None:
        """
        StepE を非同期で実行。cfg に symbol/date_range を注入。
        """
        if self._symbol is None or self._date_range is None:
            self._log("[StepE] symbol/date_range が未設定です。先に set_symbol_and_range() を呼んでください。")
            return
        if self._stepE_service is None:
            self._log("[StepE] StepEService が利用できません（未配置/生成失敗）。")
            return

        self._inject_common_params(cfg)

        use_sb3 = bool(getattr(cfg, "use_sb3_ppo", False))
        agent_name = getattr(cfg, "agent_name", "unknown")
        if use_sb3:
            self._log(f"[StepE] Start training (SB3-PPO) symbol={getattr(cfg,'symbol',None)}, agent={agent_name}")
        else:
            self._log(f"[StepE] Start training (RLSingleAlgo) symbol={getattr(cfg,'symbol',None)}, agent={agent_name}")

        def job() -> object:
            service = self._stepE_service
            if use_sb3 and hasattr(service, "run_sb3_ppo_split"):
                return service.run_sb3_ppo_split(cfg)
            return service.run_all(cfg)

        self._run_in_thread(job, job_name="StepE", on_finished=lambda r: self._on_stepE_finished(r))

    def start_stepF_training(self, cfg: StepFConfig) -> None:
        """
        StepF を非同期で実行。cfg に symbol/date_range を注入。
        """
        if self._symbol is None or self._date_range is None:
            self._log("[StepF] symbol/date_range が未設定です。先に set_symbol_and_range() を呼んでください。")
            return
        if self._stepF_service is None:
            self._log("[StepF] StepFService が利用できません（未配置/生成失敗）。")
            return

        self._inject_common_params(cfg)

        dr = getattr(cfg, "date_range", None)
        if dr is not None:
            self._log(
                f"[StepF] Start MARL training symbol={getattr(cfg,'symbol',None)}, "
                f"train={getattr(dr,'train_start',None)}~{getattr(dr,'train_end',None)}, "
                f"test={getattr(dr,'test_start',None)}~{getattr(dr,'test_end',None)}"
            )
        else:
            self._log(f"[StepF] Start MARL training symbol={getattr(cfg,'symbol',None)}")

        def job() -> object:
            service = self._stepF_service
            if hasattr(service, "train_marl"):
                return service.train_marl(cfg)
            # 互換（将来/旧名）
            for name in ("run_marl", "run_all", "train", "run"):
                if hasattr(service, name):
                    return getattr(service, name)(cfg)
            raise AttributeError("StepFService has no train/run method.")

        self._run_in_thread(job, job_name="StepF", on_finished=lambda r: self._on_stepF_finished(r))

    def request_cancel(self, target: str) -> None:
        """
        StepE / StepF への協調的キャンセル要求。
        target: "E"/"F"/"StepE"/"StepF" 等
        """
        t = (target or "").lower().strip()
        try:
            if t in ("e", "stepe", "step_e", "step-e"):
                service = self._stepE_service
                label = "StepE"
            elif t in ("f", "stepf", "step_f", "step-f"):
                service = self._stepF_service
                label = "StepF"
            else:
                self._log(f"[Backend] request_cancel: 不明な target={target}")
                return

            if service is None:
                self._log(f"[Backend] request_cancel: {label} service が利用できません。")
                return

            if hasattr(service, "request_cancel"):
                getattr(service, "request_cancel")()
                self._log(f"[Backend] request_cancel() を {label} に送信しました。")
            else:
                self._log(f"[Backend] {label} 側に request_cancel() が実装されていません。")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"[Backend] cancel 処理中に例外発生: {type(e).__name__}: {e}\n{tb}")

    # --------------------
    # internal helpers
    # --------------------

    def _log(self, msg: str) -> None:
        try:
            self.log_message.emit(str(msg))
        except Exception:
            pass

    def _inject_common_params(self, cfg: Any) -> None:
        for k, v in (("symbol", self._symbol), ("date_range", self._date_range)):
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    self._log(f"[Backend] cfg.{k} の注入に失敗しました（cfg={type(cfg).__name__}）。")

    def _on_stepE_finished(self, result_obj: object) -> None:
        self.stepE_finished.emit(result_obj)
        success = getattr(result_obj, "success", None)
        message = getattr(result_obj, "message", "")
        if success is None:
            self._log("[StepE] 終了（結果に success フィールドが無い）")
        else:
            status = "success" if bool(success) else "failed"
            self._log(f"[StepE] 終了 ({status}): {message}")

    def _on_stepF_finished(self, result_obj: object) -> None:
        self.stepF_finished.emit(result_obj)
        success = getattr(result_obj, "success", None)
        message = getattr(result_obj, "message", "")
        if success is None:
            self._log("[StepF] 終了（結果に success フィールドが無い）")
        else:
            status = "success" if bool(success) else "failed"
            self._log(f"[StepF] 終了 ({status}): {message}")

    def _run_in_thread(self, fn: Callable[[], object], job_name: str, on_finished: Callable[[object], None]) -> None:
        worker = Worker(fn)

        worker.signals.log.connect(self._log)

        def _handle_finished(result: object) -> None:
            on_finished(result)

        def _handle_error(msg: str) -> None:
            self._log(f"[{job_name}] ERROR:\n{msg}")

        worker.signals.finished.connect(_handle_finished)
        worker.signals.error.connect(_handle_error)

        self._log(f"[{job_name}] ジョブをバックグラウンドで開始します。")
        self._thread_pool.start(worker)
