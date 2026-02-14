router_bandit_fix_v4 (pair-PnL + Phase2 CSV adapter)

修正点
1) router_bandit/backtest_runner.py
   reward_next を abs(pos)*ret_next - cost に変更。
   ret_next は pos の符号で SOXL / SOXS の next-day return を選択するので、
   pos<0 を「SOXS ロング」として扱う場合は abs(pos) が正しい。

2) router_bandit/io_utils.py
   Phase2 本物CSVの列名ゆれに対応:
     - phase_cluster -> phase
     - agreement_rate 等 -> agreement_label
     - agreement_score 等 -> agreement_dist
   agreement_label が無い場合:
     - phase1..phase4 があれば多数決一致率（0.25刻み）を生成
   agreement_dist が無いが disp がある場合:
     - agreement_dist = exp(-disp/tau) を近似生成（tau=median(disp), fallback=1.0）
   agreement_label は 0..1 に正規化（0..100表記なら /100）

適用手順
- 既存の router_bandit/backtest_runner.py と router_bandit/io_utils.py を old/ にバックアップしてから
  このZIPをリポジトリ直下に展開して上書き。

使用
- Phase2 本物CSVを使う:
  python tools\run_router_bandit_backtest.py ... --phase2-state <Phase2 CSV>
- 列名が phase_cluster のままでもOK（自動変換）
