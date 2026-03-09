#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_manifest import check_step_artifacts


def _parse_agents(v: str) -> Tuple[str, ...]:
    items = [x.strip() for x in str(v or "").split(",") if x.strip()]
    return tuple(sorted(dict.fromkeys(items)))


def _parse_horizons(v: str) -> Tuple[int, ...]:
    out: List[int] = []
    for p in str(v or "").replace(" ", ",").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def _load_manifest(output_root: Path) -> Optional[dict]:
    mpath = output_root / "run_manifest.json"
    if not mpath.exists():
        return None
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _signature_from_manifest(m: Optional[dict]) -> dict:
    if not isinstance(m, dict):
        return {}
    sig = m.get("signature", {})
    return sig if isinstance(sig, dict) else {}


def _profile_set_from_manifest(m: Optional[dict]) -> Tuple[str, ...]:
    if not isinstance(m, dict):
        return ()
    agents = m.get("steps", {}).get("E", {}).get("agents", {})
    if isinstance(agents, dict) and agents:
        return tuple(sorted(a for a in agents.keys() if str(a).strip()))
    sig = _signature_from_manifest(m)
    parsed = _parse_agents(",".join(sig.get("stepe_agents", []) if isinstance(sig.get("stepe_agents"), list) else []))
    return parsed


def _build_reuse_key(args: argparse.Namespace) -> Dict[str, object]:
    # StepA-aligned core key (SOXS excluded by design)
    base = {
        "mode": str(args.mode),
        "primary_symbol": str(args.symbol),
        "test_start": str(args.test_start or ""),
        "train_years": int(args.train_years),
        "test_months": int(args.test_months),
        "enable_mamba": bool(int(args.enable_mamba)),
        "enable_mamba_periodic": bool(int(args.enable_mamba_periodic)),
        "mamba_lookback": int(args.mamba_lookback) if args.mamba_lookback is not None else None,
        "mamba_horizons": list(_parse_horizons(args.mamba_horizons)),
    }
    step = str(args.step).upper()
    if step == "DPRIME":
        base["profile_set"] = list(_parse_agents(args.profile_set))
    elif step == "E":
        base["stepe_agents"] = list(_parse_agents(args.stepe_agents))
    elif step == "F":
        base["stepe_agents"] = list(_parse_agents(args.stepe_agents))
        base["router_signature"] = str(args.router_signature or "")
    return base


def _matches_key(step: str, key: Dict[str, object], m: Optional[dict]) -> bool:
    sig = _signature_from_manifest(m)
    if not sig:
        return True  # weak/missing manifest allowed; artifact contract may still pass

    def eq(name: str, sig_name: Optional[str] = None) -> bool:
        k = sig_name or name
        return key.get(name) == sig.get(k)

    if not eq("mode"):
        return False
    if key.get("primary_symbol") != sig.get("symbol"):
        return False
    if not eq("test_start"):
        return False
    if int(key.get("train_years")) != int(sig.get("train_years", -1)):
        return False
    if int(key.get("test_months")) != int(sig.get("test_months", -1)):
        return False
    if bool(key.get("enable_mamba")) != bool(sig.get("enable_mamba", False)):
        return False
    if bool(key.get("enable_mamba_periodic")) != bool(sig.get("enable_mamba_periodic", False)):
        return False
    if key.get("mamba_lookback") != sig.get("mamba_lookback"):
        return False
    if tuple(key.get("mamba_horizons", [])) != tuple(sig.get("mamba_horizons", [])):
        return False

    if step == "DPRIME":
        expected = tuple(sorted(key.get("profile_set", [])))
        if expected:
            actual = _profile_set_from_manifest(m)
            if actual and actual != expected:
                return False
    if step in ("E", "F"):
        expected_agents = tuple(sorted(key.get("stepe_agents", [])))
        if expected_agents:
            actual_agents = tuple(sorted(sig.get("stepe_agents", [])))
            if actual_agents and actual_agents != expected_agents:
                return False
    return True


def _check_manifest_reuse(step: str, manifest: Optional[dict]) -> Tuple[bool, str, str, bool]:
    manifest_reuse = False
    manifest_reason = "manifest_missing_or_incomplete"
    parse_error_type = "none"
    fallback_to_artifact_check = False

    if manifest is not None:
        steps = manifest.get("steps", {}) if isinstance(manifest, dict) else {}
        node = steps.get(step, {}) if isinstance(steps, dict) else {}
        status = str(node.get("status", "")) if isinstance(node, dict) else ""
        if status in ("complete", "reuse"):
            if step == "E":
                agents = node.get("agents", {}) if isinstance(node, dict) else {}
                vals = [v.get("status") for v in agents.values() if isinstance(v, dict)] if isinstance(agents, dict) else []
                if vals and all(v in ("complete", "reuse") for v in vals):
                    manifest_reuse = True
                    manifest_reason = "manifest_step_and_agents_complete"
                else:
                    manifest_reason = "manifest_step_complete_but_agents_incomplete"
            else:
                manifest_reuse = True
                manifest_reason = "manifest_step_complete_or_reuse"
        else:
            manifest_reason = f"manifest_step_status={status or 'missing'}"

    return manifest_reuse, manifest_reason, parse_error_type, fallback_to_artifact_check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--scan-root', default='')
    ap.add_argument('--step', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--test-start', default='')
    ap.add_argument('--train-years', type=int, default=8)
    ap.add_argument('--test-months', type=int, default=3)
    ap.add_argument('--enable-mamba', type=int, default=1)
    ap.add_argument('--enable-mamba-periodic', type=int, default=0)
    ap.add_argument('--mamba-lookback', type=int, default=None)
    ap.add_argument('--mamba-horizons', default='')
    ap.add_argument('--profile-set', default='')
    ap.add_argument('--stepe-agents', default='')
    ap.add_argument('--router-signature', default='')
    args = ap.parse_args()

    step = args.step.upper()
    requested_output_root = Path(args.output_root)
    scan_root = Path(args.scan_root) if args.scan_root else requested_output_root.parent

    key = _build_reuse_key(args)

    matched_output_root: Optional[Path] = None
    manifest_reuse = False
    artifact_reuse = False
    manifest_reason = 'manifest_missing_or_incomplete'
    parse_error_type = 'none'
    fallback_to_artifact_check = False

    candidates: List[Path] = []
    if requested_output_root.exists():
        candidates.append(requested_output_root)
    if scan_root.exists():
        try:
            for run_dir in sorted(scan_root.iterdir(), key=lambda p: p.name, reverse=True):
                out = run_dir / 'output'
                if out.exists() and out not in candidates:
                    candidates.append(out)
        except Exception:
            pass

    for output_root in candidates:
        m = _load_manifest(output_root)
        if not _matches_key(step, key, m):
            continue

        mr, mreason, ptype, fallback = _check_manifest_reuse(step, m)
        ar = bool(check_step_artifacts(step, output_root, args.symbol, args.mode))
        if mr or ar:
            matched_output_root = output_root
            manifest_reuse = mr
            artifact_reuse = ar
            manifest_reason = mreason
            parse_error_type = ptype
            fallback_to_artifact_check = fallback
            break

    status = 'reuse' if matched_output_root is not None else 'run'
    reason = 'reuse_disabled_or_forced'
    if status == 'reuse':
        reason = 'manifest_matched' if manifest_reuse else 'artifact_contract_matched'
    else:
        reason = 'no_matching_run_with_contract'

    payload = {
      'status': status,
      'manifest_reuse': manifest_reuse,
      'artifact_reuse': artifact_reuse,
      'reason': reason,
      'manifest_reason': manifest_reason,
      'manifest_path': str((matched_output_root or requested_output_root) / 'run_manifest.json'),
      'parse_error_type': parse_error_type,
      'fallback_to_artifact_check': bool(fallback_to_artifact_check),
      'run_id': (matched_output_root.parent.name if matched_output_root else ''),
      'matched_output_root': str(matched_output_root) if matched_output_root else '',
      'selected_policy': 'latest_matching_run',
      'reuse_key': key,
      'reuse_match_found': bool(matched_output_root),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
