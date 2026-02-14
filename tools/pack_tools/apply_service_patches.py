#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_service_patches.py

目的:
- patches/services/*.py の内容を参照して、
  ai_core/services/step_a_service.py / step_d_service.py を安全に更新するための補助ツール。
- 実際の差し替えは「正本ファイル」に対して行う（パッチファイルは import されないため）。

方針:
- 既存ファイルを old/pack_tools/... にバックアップしてから上書き。
- 「完全上書き」方式（最も確実）。必要なら差分方式に変更可能。

使い方:
  cd /d C:\Users\...\soxl_rl_gui
  python tools\pack_tools\apply_service_patches.py --repo-root . --apply step_a,step_d

※注意:
- これは“完成形パッチファイルを正本へ反映する”だけ。
- StepDのEnvelope/Event設計自体が未来を見うる点は別問題（オンライン化が必要）。
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def backup(repo_root: Path, rel: Path) -> Path:
    src = repo_root / rel
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = repo_root / "old" / "pack_tools" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix
    stem = src.name[:-len(suffix)] if suffix else src.name
    bak = dst.with_name(f"{stem}_old_{ts}{suffix}")
    shutil.copy2(src, bak)
    print(f"[backup] {rel} -> {bak.relative_to(repo_root)}")
    return bak


def overwrite(repo_root: Path, rel_dst: Path, src_patch: Path) -> None:
    dst = repo_root / rel_dst
    if not dst.exists():
        raise SystemExit(f"ERROR: target not found: {dst}")
    backup(repo_root, rel_dst)
    shutil.copy2(src_patch, dst)
    print(f"[apply] {src_patch.relative_to(repo_root)} -> {rel_dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--apply", default="step_a,step_d", help="comma list: step_a,step_d")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "ai_core").is_dir():
        raise SystemExit(f"ERROR: ai_core not found under repo-root: {repo_root}")

    apply_set = {s.strip() for s in str(args.apply).split(",") if s.strip()}

    # patch sources (this script sits in tools/pack_tools)
    patch_root = repo_root / "patches" / "services"
    if "step_a" in apply_set:
        overwrite(
            repo_root,
            Path("ai_core/services/step_a_service.py"),
            patch_root / "step_a_service_no_bfill.py",
        )
    if "step_d" in apply_set:
        overwrite(
            repo_root,
            Path("ai_core/services/step_d_service.py"),
            patch_root / "step_d_service_no_center_no_bfill.py",
        )

    print("[OK] Done.")


if __name__ == "__main__":
    main()
