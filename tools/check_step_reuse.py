#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_manifest import (
    build_canonical_output_root,
    load_split_summary,
    required_outputs_for_step,
    split_summary_matches,
    validate_step_outputs,
)


def _decision(step: str, canonical_root: Path, symbol: str, mode: str, test_start: str, train_years: int, test_months: int) -> Dict[str, object]:
    step = step.upper()
    step_dir_name = 'stepDprime' if step == 'DPRIME' else f'step{step}'
    candidate_step_dir = canonical_root / step_dir_name / mode
    print(f"[reuse] step={step} candidate={candidate_step_dir}", file=sys.stderr)

    if not canonical_root.exists():
        return {"status":"run","reason":"missing_step_dir","final":"execute","symbol":"pass","mode":"pass","split":"fail","required_outputs":"fail","validation":"skip"}

    summary = load_split_summary(canonical_root)
    if summary is None:
        return {"status":"run","reason":"split_summary_missing","final":"execute","symbol":"pass","mode":"pass","split":"fail","required_outputs":"fail","validation":"skip"}

    symbol_ok = str(summary.get('symbol','')).upper() == str(symbol).upper()
    mode_ok = str(summary.get('mode','')).lower() == str(mode).lower()
    split_ok = split_summary_matches(summary, symbol=symbol, mode=mode, test_start=test_start, train_years=train_years, test_months=test_months)

    if not symbol_ok:
        return {"status":"run","reason":"symbol_mismatch","final":"execute","symbol":"fail","mode":"pass" if mode_ok else "fail","split":"fail","required_outputs":"skip","validation":"skip"}
    if not mode_ok:
        return {"status":"run","reason":"mode_mismatch","final":"execute","symbol":"pass","mode":"fail","split":"fail","required_outputs":"skip","validation":"skip"}
    if not split_ok:
        return {"status":"run","reason":"split_mismatch","final":"execute","symbol":"pass","mode":"pass","split":"fail","required_outputs":"skip","validation":"skip"}

    valid, v_reason = validate_step_outputs(step, canonical_root, symbol, mode)
    req_ok = v_reason != 'missing_required_outputs'
    if not valid:
        reason = v_reason
        return {"status":"run","reason":reason,"final":"execute","symbol":"pass","mode":"pass","split":"pass","required_outputs":"pass" if req_ok else "fail","validation":"fail" if req_ok else "skip"}

    return {"status":"skip","reason":"reuse","final":"reuse","symbol":"pass","mode":"pass","split":"pass","required_outputs":"pass","validation":"pass"}


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

    canonical_root = build_canonical_output_root(_REPO_ROOT / 'output', args.mode, args.symbol, args.test_start)
    d = _decision(args.step, canonical_root, args.symbol, args.mode, args.test_start, args.train_years, args.test_months)

    print(
        f"[reuse] step={args.step.upper()} symbol={d['symbol']} mode={d['mode']} split={d['split']} required_outputs={d['required_outputs']} validation={d['validation']}",
        file=sys.stderr,
    )
    if d['status'] == 'skip':
        print(f"[reuse] step={args.step.upper()} final=reuse", file=sys.stderr)
    else:
        print(f"[reuse] step={args.step.upper()} final=execute reason={d['reason']}", file=sys.stderr)

    payload = {
      'status': d['status'],
      'manifest_reuse': False,
      'artifact_reuse': d['status'] == 'skip',
      'reason': d['reason'],
      'manifest_reason': 'disabled',
      'manifest_path': str(canonical_root / 'run_manifest.json'),
      'parse_error_type': 'none',
      'fallback_to_artifact_check': False,
      'run_id': '',
      'matched_output_root': str(canonical_root),
      'selected_policy': 'canonical_output_root',
      'reuse_key': {
          'mode': args.mode,
          'symbol': args.symbol,
          'test_start': args.test_start,
          'train_years': int(args.train_years),
          'test_months': int(args.test_months),
          'required_outputs': list(required_outputs_for_step(args.step, args.symbol, args.mode)),
      },
      'reuse_match_found': d['status'] == 'skip',
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
