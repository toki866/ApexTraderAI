from __future__ import annotations

from ai_core.services.step_e_service import StepEConfig, _training_config_summary


def test_training_config_summary_includes_policy_and_dprime_fields() -> None:
    cfg = StepEConfig(
        agent="dprime_all_features_h01",
        seed=123,
        obs_profile="D",
        use_stepd_prime=True,
        use_dprime_state=False,
        dprime_profile="dprime_all_features_h01",
        dprime_sources="all_features",
        dprime_horizons="1,5,10,20",
        policy_kind="ppo",
        ppo_total_timesteps=90000,
        ppo_n_epochs=4,
        ppo_n_steps=1024,
        ppo_batch_size=256,
    )

    summary = _training_config_summary(cfg, device="cuda")

    assert summary["policy_kind"] == "ppo"
    assert summary["ppo_total_timesteps"] == 90000
    assert summary["ppo_n_epochs"] == 4
    assert summary["ppo_n_steps"] == 1024
    assert summary["ppo_batch_size"] == 256
    assert summary["device"] == "cuda"
    assert summary["seed"] == 123
    assert summary["obs_profile"] == "D"
    assert summary["use_stepd_prime"] is True
    assert summary["dprime_profile"] == "dprime_all_features_h01"
