# StepE 修復パック（restore + no-leak）
このZIPをリポジトリ直下に展開すると tools/ に2つの修復スクリプトが入ります。

## 推奨（全部まとめて直す）
python tools\fix_step_e_restore_and_patch_no_leak.py

完了後：
python -m py_compile ai_core\services\step_e_service.py
python tools\run_stepE_compare_stepdprime10.py --symbol SOXL --mode sim
