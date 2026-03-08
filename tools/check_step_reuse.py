#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_manifest import check_step_artifacts

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--step', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--mode', required=True)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    step = args.step.upper()
    manifest_reuse = False
    manifest_reason = 'manifest_missing_or_incomplete'
    manifest_path = output_root / 'run_manifest.json'
    parse_error_type = 'none'
    fallback_to_artifact_check = False
    if manifest_path.exists():
      try:
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        steps = m.get('steps', {}) if isinstance(m, dict) else {}
        node = steps.get(step, {}) if isinstance(steps, dict) else {}
        status = str(node.get('status', '')) if isinstance(node, dict) else ''
        if status in ('complete', 'reuse'):
          if step == 'E':
            agents = node.get('agents', {}) if isinstance(node, dict) else {}
            vals = [v.get('status') for v in agents.values() if isinstance(v, dict)] if isinstance(agents, dict) else []
            if vals and all(v in ('complete', 'reuse') for v in vals):
              manifest_reuse = True
              manifest_reason = 'manifest_step_and_agents_complete'
            else:
              manifest_reason = 'manifest_step_complete_but_agents_incomplete'
          else:
            manifest_reuse = True
            manifest_reason = 'manifest_step_complete_or_reuse'
        else:
          manifest_reason = f'manifest_step_status={status or "missing"}'
      except Exception as ex:
        parse_error_type = ex.__class__.__name__
        fallback_to_artifact_check = True
        manifest_reason = f'manifest_parse_error:{parse_error_type}'

    artifact_reuse = bool(check_step_artifacts(step, output_root, args.symbol, args.mode))
    status = 'reuse' if (manifest_reuse or artifact_reuse) else 'run'
    reason = 'manifest_matched' if manifest_reuse else ('artifact_contract_matched' if artifact_reuse else manifest_reason)
    print(json.dumps({
      'status': status,
      'manifest_reuse': manifest_reuse,
      'artifact_reuse': artifact_reuse,
      'reason': reason,
      'manifest_reason': manifest_reason,
      'manifest_path': str(manifest_path),
      'parse_error_type': parse_error_type,
      'fallback_to_artifact_check': bool(fallback_to_artifact_check),
      'run_id': output_root.parent.name,
    }))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
