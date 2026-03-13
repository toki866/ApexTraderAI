from pathlib import Path


def test_stepf_workflow_emits_entry_and_subprocess_logs() -> None:
    workflow = Path('.github/workflows/run_desktop_pipeline.yml').read_text(encoding='utf-8')
    assert '[STEPF_ENTRY] begin' in workflow
    assert '[STEPF_ENTRY] output_root=' in workflow
    assert '[STEPF_ENTRY] stepf_dir=' in workflow
    assert '[STEPF_ENTRY] before_mode_loop' in workflow
    assert '[STEPF_ENTRY] before_service_run' in workflow
    assert '[STEPF_SUBPROCESS] returncode=' in workflow


def test_stepf_workflow_precreates_stepf_dirs_and_avoids_dir_missing_flag() -> None:
    workflow = Path('.github/workflows/run_desktop_pipeline.yml').read_text(encoding='utf-8')
    assert "New-Item -ItemType Directory -Path $stepFRootDir -Force | Out-Null" in workflow
    assert "New-Item -ItemType Directory -Path $stepFSimDir -Force | Out-Null" in workflow
    assert '[STEPF_MULTI] failure_stepF_dir_missing=1' not in workflow
