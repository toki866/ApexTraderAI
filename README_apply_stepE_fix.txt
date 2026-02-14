StepE 修正パッチ（ValueError unpack / 予測列の誤DROP / no-leak forward return）

1) このZIPを展開して、リポジトリ直下の tools/ に置く
   soxl_rl_gui/tools/fix_step_e_unpack_and_no_leak.py

2) 反映
   python tools\fix_step_e_unpack_and_no_leak.py

3) コンパイル確認（任意）
   python -m py_compile ai_core\services\step_e_service.py

4) 再実行
   python tools\run_stepE_compare_stepdprime10.py --symbol SOXL --mode sim

5) リーク監査（推奨）
   python tools\stepE_column_audit.py --symbol SOXL --mode sim

ポイント
- label / label_available / close_true / REALCLOSE 等は obs から除外（リーク遮断）
- Pred_*（例: Pred_y_from_anchor）は「予測値」なので obs に入れてOK
- reward 用の return は ret_fwd1 (Close[t]→Close[t+1]) を用意し、obsには混ぜません
