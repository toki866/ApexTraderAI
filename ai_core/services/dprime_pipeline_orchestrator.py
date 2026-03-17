from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

    def run(self, cfg: DPrimePipelineOrchestratorConfig, *, run_step_bc) -> Dict[str, object]:
        dprime = StepDPrimeService()
        stepd_dir = Path(cfg.output_root) / "stepDprime" / cfg.mode
        marker_dir = stepd_dir / "pipeline_markers"

        errors: List[BaseException] = []

        def _bc_lane() -> None:
            write_status_marker(marker_dir, "StepC", "RUNNING", {"symbol": cfg.symbol})
            try:
                run_step_bc()
                write_status_marker(marker_dir, "StepC", "READY", {"symbol": cfg.symbol})
            except BaseException as exc:
                errors.append(exc)
                write_status_marker(marker_dir, "StepC", "FAILED", {"symbol": cfg.symbol, "error": repr(exc)})

        def _dprime_base_lane() -> None:
            try:
                dprime.run_base_cluster(cfg.stepd_cfg)
            except BaseException as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_bc_lane, name="lane-stepbc", daemon=False)
        t2 = threading.Thread(target=_dprime_base_lane, name="lane-dprime-base", daemon=False)
        t1.start(); t2.start(); t1.join(); t2.join()
        if errors:
            raise RuntimeError(f"parallel lane failed: {errors[0]}")

        profiles_done: List[str] = []
        for profile in cfg.stepd_cfg.profiles:
            dprime.run_final_profile(cfg.stepd_cfg, profile, force_cpu=bool(cfg.force_cpu_dprime_final))
            profiles_done.append(profile)
            stepe_cfg = self.stepe_cfg_by_profile.get(profile)
            if stepe_cfg is not None:
                marker_name = f"StepE_{stepe_cfg.agent}"
                write_status_marker(marker_dir, marker_name, "RUNNING", {"profile": profile, "agent": stepe_cfg.agent})
                try:
                    self.step_e_service.run_agent(stepe_cfg, date_range=self.date_range, symbol=cfg.symbol, mode=cfg.mode)
                except BaseException as exc:
                    write_status_marker(marker_dir, marker_name, "FAILED", {"profile": profile, "agent": stepe_cfg.agent, "error": repr(exc)})
                    raise
                write_status_marker(marker_dir, marker_name, "READY", {"profile": profile, "agent": stepe_cfg.agent})

        return {"status": "done", "profiles_done": profiles_done}
