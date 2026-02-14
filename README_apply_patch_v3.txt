\
適用手順（Windows）
==================
1) ZIPをリポジトリ直下に展開して上書き。
2) python -m py_compile ai_core\services\step_e_service.py
   python -m py_compile ai_core\services\step_f_service.py
3) python tools\run_stepE_compare_stepdprime10.py --symbol SOXL --mode sim
4) python tools\check_stepE_reward_alignment.py --symbol SOXL --mode sim --all
   Top match が cc_next ならOK
