# -*- coding: utf-8 -*-
"""
Apply router_bandit fixes (dynamic n_actions + numpy scalar) with backups.

Usage (repo root):
  python tools/apply_router_bandit_fix_v3.py --zip router_bandit_fix_v3.zip

If you already extracted the zip somewhere, you can pass that path.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import zipfile
from datetime import datetime

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to router_bandit_fix_v3.zip")
    ap.add_argument("--repo-root", default=".", help="Repo root (default: current dir)")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    zpath = Path(args.zip).resolve()
    if not zpath.exists():
        raise FileNotFoundError(str(zpath))

    # Extract to temp
    tmp = repo / "_tmp_router_bandit_fix_v3"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(tmp)

    # Targets
    targets = [
        ("router_bandit/backtest_runner.py", repo / "router_bandit" / "backtest_runner.py"),
        ("router_bandit/routers.py",        repo / "router_bandit" / "routers.py"),
        ("router_bandit/linucb.py",         repo / "router_bandit" / "linucb.py"),
    ]

    # Backup dir
    old = repo / "router_bandit" / "old"
    old.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for rel, dst in targets:
        src = tmp / rel
        if not src.exists():
            raise FileNotFoundError(str(src))
        if dst.exists():
            bkp = old / f"{dst.stem}_old_{ts}{dst.suffix}"
            shutil.copy2(dst, bkp)
            print(f"[OK] backup -> {bkp}")
        else:
            print(f"[WARN] target missing, will create -> {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[OK] overwrite -> {dst}")

    # Verify key strings
    bt = (repo / "router_bandit" / "backtest_runner.py").read_text(encoding="utf-8", errors="ignore")
    rt = (repo / "router_bandit" / "routers.py").read_text(encoding="utf-8", errors="ignore")
    ok = True
    if "len(agent_names) + 1" not in bt:
        print("[NG] backtest_runner.py missing 'len(agent_names) + 1'")
        ok = False
    if "n_actions=11" in bt or "\"n_actions\": 11" in bt:
        print("[NG] backtest_runner.py still hardcodes n_actions=11")
        ok = False
    if "range(11)" in rt:
        print("[NG] routers.py still contains range(11)")
        ok = False
    if ok:
        print("[OK] verification passed")
    else:
        print("[WARN] verification failed; please inspect files")

    shutil.rmtree(tmp, ignore_errors=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
