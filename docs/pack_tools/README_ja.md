# pack_tools_bundle（train/test/display/ops 生成ツール）

## 目的
- train/test は display を読まない運用に固定して、未来混入を防ぐ
- 境界の正本として split_dates.csv を作る
- 監査用に run_manifest / data_hashes を残す

## 1) split_dates.csv を作る
例（SOXL パイロット）:
```bat
python tools\build_split_dates.py ^
  --symbol SOXL ^
  --test-start 2022-01-03 ^
  --test-end 2022-03-31 ^
  --train-years 8 ^
  --prices-csv output\stepA\stepA_prices_SOXL.csv ^
  --out output\runs\SOXL_20260115_073000\packs\ops\split_dates.csv
```

## 2) packs を作る（raw → packs）
```bat
python tools\build_packs_from_raw.py ^
  --symbol SOXL ^
  --input-root output ^
  --run-id SOXL_20260115_073000 ^
  --test-start 2022-01-03 --test-end 2022-03-31 --train-years 8 ^
  --out-root output\runs\SOXL_20260115_073000\packs
```

## 3) リーク監査（簡易）
```bat
python tools\validate_no_leak.py ^
  --features-csv output\stepA\stepA_features_SOXL.csv ^
  --stepa-code ai_core\services\step_a_service.py ^
  --stepd-code ai_core\services\step_d_service.py
```

## 4) パッチ（任意）
- StepA: patches/step_a_service_no_bfill.py（bfill禁止）
- StepD: patches/step_d_service_no_center_no_bfill.py（center/bfill削除）

※ StepD のイベント自体は「全期間を見て」作るため、RL観測に使う場合は根本的に未来情報を含み得ます。
完全因果化は別実装（オンライン envelope）が必要です。
