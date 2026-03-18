from types import SimpleNamespace

from ai_core.services.step_b_service import StepBService


def test_device_evidence_prefers_agent_metadata() -> None:
    svc = StepBService(app_config=SimpleNamespace())

    full = SimpleNamespace(
        device_execution="cpu",
        info={
            "device_requested": "cuda",
            "device_execution": "cpu",
            "device_execution_verified": True,
            "device_execution_evidence": {
                "train_model_devices": ["cpu"],
                "train_tensor_devices": ["cpu"],
            },
            "device_resolution_source": "cfg.device",
            "device_fallback_reason": "requested_cuda_unavailable",
        },
    )
    periodic = SimpleNamespace(
        device_execution="cpu",
        info={
            "device_requested": "auto",
            "device_execution": "cpu",
            "device_execution_verified": True,
            "device_execution_evidence": {
                "infer_model_devices": ["cpu"],
            },
            "device_resolution_source": "torch.cuda.is_available",
            "device_fallback_reason": "auto_cuda_unavailable",
        },
    )

    summary = svc._device_evidence(("full", full), ("periodic", periodic))

    assert summary["device_requested"] == "cuda"
    assert summary["device_execution"] == "cpu"
    assert summary["device_execution_verified"] is True
    assert summary["device_resolution_source"] == "cfg.device"
    assert summary["device_fallback_reason"] == "requested_cuda_unavailable"
    assert summary["device_evidence_source"] == "agent_result[full]"
    assert summary["device_execution_evidence"]["train_model_devices"] == ["cpu"]
    assert summary["agent_device_evidence"]["periodic"]["device_fallback_reason"] == "auto_cuda_unavailable"


def test_device_evidence_defaults_to_unknown_without_agent_runtime_data() -> None:
    svc = StepBService(app_config=SimpleNamespace())

    summary = svc._device_evidence(("full", SimpleNamespace(info={})))

    assert summary["device_execution"] == "unknown"
    assert summary["device_execution_verified"] is False
    assert summary["device_fallback_reason"] == "missing_device_execution_evidence"
