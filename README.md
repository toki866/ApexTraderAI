# ApexTraderAI

## 実行方法（GitHub / ローカル共通）

実行エントリーポイントは `tools/run_pipeline.py` に統一しています。

### パイプライン標準仕様（重要）

- 標準ステップは **`A,B,C,DPRIME,E,F`** です。
- **`D` は廃止運用（Dは使わない）**で、通常のパイプラインでは使いません。
- `DPRIME` は **「StepD上位版」** です。`DPRIME` 単独で、チャート圧縮（Phase2）と RL state 生成まで実施します。
- `StepE` は `DPRIME` が生成した state（必要に応じて embeddings 併用）を観測入力として利用します。
- `StepE` は **PPO 専用**です。reward 定義・train/test 境界・StepE/StepF の責務分離は維持したまま、既定値は短時間学習向けに調整しています。
- `StepF` は upstream の `cluster_id_stable` / `cluster_id_raw20` / `rare_flag_raw20` を consume して StepE 日次ログを束ねる最終 router 層です。StepF 自身は cluster/regime を再学習しません。

### 3ヶ月窓は2種類（用途が異なる）

1. **固定長圧縮の3ヶ月窓**: `DPRIME` / Phase2 側で使う圧縮窓
2. **lookback の3ヶ月窓**: `StepB` 側で使う観測窓

この2つは同じ「3ヶ月」でも役割が異なるため、混同しないでください。

```bash
python tools/run_pipeline.py --symbol SOXL --test-start 2022-01-03
```

StepE の wall time 調整が必要な場合は、たとえば以下のように PPO timesteps と agent 並列数を上書きできます。

```bash
python tools/run_pipeline.py --symbol SOXL --steps E --stepE-ppo-total-timesteps 80000 --stepE-max-parallel-agents 2
```

- 推奨運用値は **sim=2 / live=1** です。live で `2` を使うのは GPU/CPU 余力を確認した後にしてください。
- 実行時は StepE summary / audit / log に `requested_parallel_agents` / `max_parallel_agents` / `effective_parallel_agents` / `parallelism_warning` が残るため、実際に有効化された並列数を追跡できます。
- 同一 run 内で複数 agent を回す場合、StepE は merge 済み入力を共有し、summary / audit / log に `merge_inputs_cache_hit` と `merge_cache_key` を残します。

- 既存の `run_*.py` は後方互換ラッパで、内部的に同じパイプライン実行に委譲します。
- 相対パス（`config/`, `data/`, `output/`）はリポジトリルート基準で解決されるため、実行時 CWD に依存しません。
- GitHub Actions 実行時は `GITHUB_WORKSPACE` を repo ルートとして使います。ローカル手動実行は OneDrive 外の clone（推奨: `C:\work\apex-trader-ai`）を前提にしてください。

## データ準備（clone直後向け）

`data/` はリポジトリに含まれますが、`data/*.csv` は Git 管理外です。必要な価格CSVが無い場合は以下のどちらかで生成できます。

```bash
pip install -r requirements.txt
python tools/prepare_data.py --symbols SOXL,SOXS --start 2014-01-01 --end 2022-03-31
```

または `run_pipeline.py` 実行時に自動生成できます（デフォルトON）。

```bash
python tools/run_pipeline.py --symbol SOXL --steps A --auto-prepare-data 1
```

- 自動生成期間は `--data-start` / `--data-end` で指定できます（既定: `2010-01-01` 〜 当日）。
- `--auto-prepare-data 0` を指定すると自動生成を無効化できます。
- 取得は `yfinance`（Yahoo Finance の公開データ）を利用します。利用規約・ライセンスを確認し、研究/個人利用の範囲で利用してください。

## CI

GitHub Actions (`.github/workflows/ci.yml`) で以下を実行します。

1. `pip install -r requirements.txt`
2. 全 `.py` ファイルに対する `python -m py_compile`
3. 主要モジュールの import smoke check
