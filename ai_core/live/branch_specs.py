from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    dprime_profile: str
    agent_id: str
    stepb_required: bool
    stepb_pred_k: int
    stepb_required_steps: List[int]


BRANCHES: Dict[str, BranchSpec] = {
    "dprime_bnf_h01": BranchSpec("dprime_bnf_h01", "dprime_bnf_h01", "dprime_bnf_h01", True, 20, [1]),
    "dprime_bnf_h02": BranchSpec("dprime_bnf_h02", "dprime_bnf_h02", "dprime_bnf_h02", True, 20, [1, 5, 10, 20]),
    "dprime_bnf_3scale": BranchSpec("dprime_bnf_3scale", "dprime_bnf_3scale", "dprime_bnf_3scale", True, 20, list(range(1, 21))),
    "dprime_mix_h01": BranchSpec("dprime_mix_h01", "dprime_mix_h01", "dprime_mix_h01", True, 20, [1]),
    "dprime_mix_h02": BranchSpec("dprime_mix_h02", "dprime_mix_h02", "dprime_mix_h02", True, 20, [1, 5, 10, 20]),
    "dprime_mix_3scale": BranchSpec("dprime_mix_3scale", "dprime_mix_3scale", "dprime_mix_3scale", True, 20, list(range(1, 21))),
    "dprime_all_features_h01": BranchSpec("dprime_all_features_h01", "dprime_all_features_h01", "dprime_all_features_h01", True, 20, [1]),
    "dprime_all_features_h02": BranchSpec("dprime_all_features_h02", "dprime_all_features_h02", "dprime_all_features_h02", True, 20, [1, 5, 10, 20]),
    "dprime_all_features_h03": BranchSpec("dprime_all_features_h03", "dprime_all_features_h03", "dprime_all_features_h03", True, 20, list(range(1, 21))),
    "dprime_all_features_3scale": BranchSpec("dprime_all_features_3scale", "dprime_all_features_3scale", "dprime_all_features_3scale", True, 20, list(range(1, 21))),
}

DEFAULT_SAFE_BRANCHES = ["dprime_bnf_h01", "dprime_all_features_h01"]
