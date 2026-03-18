from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from ai_core.services.step_dprime_service import StepDPrimeConfig, StepDPrimeService
from ai_core.services.step_e_service import StepEConfig, StepEService
from ai_core.utils.file_ready_utils import write_status_marker


@dataclass
class DPrimePipelineOrchestratorConfig:
    symbol: str
    mode: str
    output_root: str
    stepd_cfg: StepDPrimeConfig
    force_cpu_dprime_final: bool = True


class DPrimePipelineOrchestrator:
    """Safe orchestrator: run (B->C) and (D' base+cluster) in parallel, then stream profile->StepE."""

    def __init__(self, *, step_e_service: StepEService, date_range, stepe_cfg_by_profile: Dict[str, StepEConfig]):
        self.step_e_service = step_e_service
        self.date_range = date_range
        self.stepe_cfg_by_profile = dict(stepe_cfg_by_profile)

    def _utcnow_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def run(
        self,
        cfg: DPrimePipelineOrchestratorConfig,
        *,
        run_step_b: Callable[[], None],
        run_step_c: Callable[[], None],
    ) -> Dict[str, object]:
        dprime = StepDPrimeService()
        stepd_dir = Path(cfg.output_root) / "stepDprime" / cfg.mode
        marker_dir = stepd_dir / "pipeline_markers"

        errors: List[BaseException] = []
        profiles_done: List[str] = []
        profiles_failed: List[str] = []
        agents_done: List[str] = []
        agents_failed: List[str] = []

        def _bc_lane() -> None:
            lane_started = self._utcnow_iso()
            print(f"[DPRIME_STREAM] lane=stepbc status=start started_at={lane_started}")
            try:
                stepb_started = self._utcnow_iso()
                print(f"[DPRIME_STREAM] lane=stepbc step=StepB status=start started_at={stepb_started}")
                write_status_marker(marker_dir, "StepB", "RUNNING", {"symbol": cfg.symbol, "started_at": stepb_started})
                try:
                    run_step_b()
                except BaseException as exc:
                    errors.append(exc)
                    write_status_marker(marker_dir, "StepB", "FAILED", {"symbol": cfg.symbol, "error": repr(exc), "ended_at": self._utcnow_iso()})
                    print(f"[DPRIME_STREAM] lane=stepbc step=StepB status=failed ended_at={self._utcnow_iso()} error={repr(exc)}")
                    print(f"[DPRIME_STREAM] lane=stepbc status=failed ended_at={self._utcnow_iso()} failed_step=StepB error={repr(exc)}")
                    return
                write_status_marker(marker_dir, "StepB", "READY", {"symbol": cfg.symbol, "ended_at": self._utcnow_iso()})
                print(f"[DPRIME_STREAM] lane=stepbc step=StepB status=ready ended_at={self._utcnow_iso()}")

                stepc_started = self._utcnow_iso()
                print(f"[DPRIME_STREAM] lane=stepbc step=StepC status=start started_at={stepc_started}")
                write_status_marker(marker_dir, "StepC", "RUNNING", {"symbol": cfg.symbol, "started_at": stepc_started})
                try:
                    run_step_c()
                except BaseException as exc:
                    errors.append(exc)
                    write_status_marker(marker_dir, "StepC", "FAILED", {"symbol": cfg.symbol, "error": repr(exc), "ended_at": self._utcnow_iso()})
                    print(f"[DPRIME_STREAM] lane=stepbc step=StepC status=failed ended_at={self._utcnow_iso()} error={repr(exc)}")
                    print(f"[DPRIME_STREAM] lane=stepbc status=failed ended_at={self._utcnow_iso()} failed_step=StepC error={repr(exc)}")
                    return
                write_status_marker(marker_dir, "StepC", "READY", {"symbol": cfg.symbol, "ended_at": self._utcnow_iso()})
                print(f"[DPRIME_STREAM] lane=stepbc step=StepC status=ready ended_at={self._utcnow_iso()}")
                print(f"[DPRIME_STREAM] lane=stepbc status=ready ended_at={self._utcnow_iso()}")
            except BaseException as exc:
                errors.append(exc)
                print(f"[DPRIME_STREAM] lane=stepbc status=failed ended_at={self._utcnow_iso()} error={repr(exc)}")

        def _dprime_base_lane() -> None:
            lane_started = self._utcnow_iso()
            print(f"[DPRIME_STREAM] lane=dprime_base status=start started_at={lane_started}")
            try:
                dprime.run_base_cluster(cfg.stepd_cfg)
                print(f"[DPRIME_STREAM] lane=dprime_base status=ready ended_at={self._utcnow_iso()}")
            except BaseException as exc:
                errors.append(exc)
                print(f"[DPRIME_STREAM] lane=dprime_base status=failed error={repr(exc)}")

        t1 = threading.Thread(target=_bc_lane, name="lane-stepbc", daemon=False)
        t2 = threading.Thread(target=_dprime_base_lane, name="lane-dprime-base", daemon=False)
        t1.start(); t2.start(); t1.join(); t2.join()
        if errors:
            raise RuntimeError(f"parallel lane failed: {errors[0]}")

        for profile in cfg.stepd_cfg.profiles:
            profile_started = self._utcnow_iso()
            print(
                f"[DPRIME_STREAM] profile={profile} status=start force_cpu={str(bool(cfg.force_cpu_dprime_final)).lower()} "
                f"started_at={profile_started}"
            )
            try:
                dprime.run_final_profile(cfg.stepd_cfg, profile, force_cpu=bool(cfg.force_cpu_dprime_final))
                profiles_done.append(profile)
                print(f"[DPRIME_STREAM] profile={profile} status=ready ended_at={self._utcnow_iso()}")
            except BaseException as exc:
                profiles_failed.append(profile)
                print(f"[DPRIME_STREAM] profile={profile} status=failed ended_at={self._utcnow_iso()} error={repr(exc)}")
                raise
            stepe_cfg = self.stepe_cfg_by_profile.get(profile)
            if stepe_cfg is not None:
                marker_name = f"StepE_{stepe_cfg.agent}"
                agent_started = self._utcnow_iso()
                print(
                    f"[DPRIME_STREAM] profile={profile} agent={stepe_cfg.agent} "
                    f"force_cpu={str(bool(cfg.force_cpu_dprime_final)).lower()} status=start started_at={agent_started}"
                )
                write_status_marker(marker_dir, marker_name, "RUNNING", {"profile": profile, "agent": stepe_cfg.agent, "started_at": agent_started, "force_cpu": bool(cfg.force_cpu_dprime_final)})
                try:
                    self.step_e_service.run_agent(stepe_cfg, date_range=self.date_range, symbol=cfg.symbol, mode=cfg.mode)
                except BaseException as exc:
                    agents_failed.append(str(stepe_cfg.agent))
                    write_status_marker(marker_dir, marker_name, "FAILED", {"profile": profile, "agent": stepe_cfg.agent, "error": repr(exc), "ended_at": self._utcnow_iso(), "force_cpu": bool(cfg.force_cpu_dprime_final)})
                    print(f"[DPRIME_STREAM] profile={profile} agent={stepe_cfg.agent} status=failed ended_at={self._utcnow_iso()} error={repr(exc)}")
                    raise
                agents_done.append(str(stepe_cfg.agent))
                write_status_marker(marker_dir, marker_name, "READY", {"profile": profile, "agent": stepe_cfg.agent, "ended_at": self._utcnow_iso(), "force_cpu": bool(cfg.force_cpu_dprime_final)})
                print(f"[DPRIME_STREAM] profile={profile} agent={stepe_cfg.agent} status=ready ended_at={self._utcnow_iso()}")

        return {
            "status": "done",
            "profiles_done": profiles_done,
            "agents_done": agents_done,
            "profiles_failed": profiles_failed,
            "agents_failed": agents_failed,
        }
