# 不要ファイル候補の棚卸し（ApexTraderAI）

最終更新: 2026-02-14

## 判定基準
- **不要候補（高）**: ZIP適用手順や一時パッチ説明など、実行コードではなく「一度使ったら役目を終える」ファイル。
- **不要候補（中）**: 同名・同用途の重複スクリプト（どちらかに統一可能）。
- **要確認**: 直近運用で参照される可能性があるため、削除前に参照元確認が必要。

## 不要候補（高）
トップレベルに残っているパッチ適用メモ類。いずれも「ZIP展開して実行」前提の説明文で、恒常的なドキュメントとしては重複・陳腐化しやすい。

- `README_apply_patch_v3.txt`
- `README_apply_stepE_fix.txt`
- `README_PATCH_V4C.txt`
- `README_fix_stepE_import_re_v2.txt`
- `README_router_bandit_fix_v4.txt`
- `README_stepE_fix_restore_and_no_leak.txt`
- `README_profit_check.txt`
- `README.txt`
- `PATCH_NOTES.md`

**推奨対応**
1. 履歴価値があるものだけ `docs/` 配下に統合（例: `docs/patch_history.md`）。
2. 残りは削除、もしくは `archive/` に移動して通常導線から外す。

## 不要候補（中）

### 1) 実行スクリプトの重複
- `run_stepd_prime.py`
- `tools/run_stepd_prime.py`

同名スクリプトが2箇所に存在。片方をラッパー化（importして呼び出すのみ）または削除で整理可能。

### 2) ツールの重複
- `tools/build_packs_from_raw.py`
- `tools/pack_tools/build_packs_from_raw.py`
- `tools/build_split_dates.py`
- `tools/pack_tools/build_split_dates.py`
- `tools/validate_no_leak.py`
- `tools/pack_tools/validate_no_leak.py`

`tools/pack_tools/` と `tools/` 直下に同名ツールが共存しているため、運用導線を一本化すると保守コストを削減可能。

## 要確認
- `tools/_payload_stepE/step_e_service.py`

`ai_core/services/step_e_service.py` のペイロード用コピーに見える。自動パッチスクリプトで参照している可能性があるため、`rg "_payload_stepE|step_e_service.py" tools/` で参照を確認してから削除判断する。

## 今回の結論
- まずはトップレベルのパッチ説明ファイル群を統合・退避するだけで、リポジトリ可読性が大きく改善する。
- 次点で `tools/` 系の重複スクリプトを「正本1つ + 必要なら薄いラッパー」に揃えるのが安全。
