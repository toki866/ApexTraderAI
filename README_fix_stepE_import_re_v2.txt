# fix_stepE_import_re_v2

目的:
- StepE の import 時に発生する
  NameError: name 're' is not defined
  を確実に解消します。

原因:
- step_e_service.py の module top-level で `re.compile(...)` を呼んでいるのに、
  `import re` がその後に書かれている (または無い) パターンがあります。
- 既存の fixer は「import re がファイル内にあるか」だけ見ていたため、
  “後ろにある import re” を見つけてしまい、修正しないことがありました。

使い方:
1) このZIPを repo 直下に展開（tools/fix_step_e_import_re.py を上書き）
2) 走らせる:
   python tools\fix_step_e_import_re.py
3) もう一度:
   python tools\run_stepE_compare_stepdprime10.py --symbol SOXL --mode sim

バックアップ:
- 修正前の ai_core/services/step_e_service.py は old/step_e_service_old_YYYYMMDD_HHMMSS_01.py に保存されます。
