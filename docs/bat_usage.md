# Windows BAT 運用ガイド（ローカル計算 → 完了後 OneDrive コピー）

## 追加されたスクリプト

- `scripts\run_all_local_then_copy.bat`
  - メイン実行。`run_id` 付きのローカル作業ディレクトリに `data/logs` を作成し、canonical output へ直接出力した後、必要時のみ OneDrive 向け ZIP を作成します。
- `scripts\doctor.bat`
  - 実行前診断。`git hash`、`python --version`、`torch/cuda`、`data\prices_*.csv` の存在と更新日時をログ化します。
- `scripts\copy_run_to_onedrive.bat`
  - canonical output または既存 run フォルダを入力に取り、canonical output 由来の ZIP だけを OneDrive に再エクスポートします。
- `scripts\bat_config.bat`
  - `SYMBOLS/日付範囲/WORK_ROOT` などの既定値を集約しています。

## 事前準備

1. OneDrive 外のローカル clone（推奨: `C:\work\apex-trader-ai`）で作業する
2. リポジトリを pull
3. `pip install -r requirements.txt`
4. 必要なら `scripts\bat_config.bat` の既定値を編集

## 実行方法

```bat
scripts\run_all_local_then_copy.bat
```

## ローカル生成物

既定では以下に出力されます。

- `C:\work\apex_work\runs\<run_id>\data`
- `C:\work\apex_work\runs\<run_id>\logs\run_<run_id>.log`
- `C:\work\apex_work\output\<mode>\<symbol>\<test_start>_<YYYYMMDD>_<NNN>`（canonical output 正本）
- ローカル ZIP は生成しません

## OneDrive コピー先

優先順:

1. `%ONE_DRIVE_RUNS_ROOT%\export`
2. `%OneDrive%\ApexTraderAI\runs\export`

保存されるのは canonical output を ZIP 化した `output_YYYYMMDD_NNN.zip` のみです。同日内の連番は既存 ZIP を走査して最大番号+1で採番します。

## 失敗時（エラーパケット）

`logs\run_<run_id>.log` に以下が残ります。

- 実行コマンド（`[CMD] ...`）
- 終了コード（`[RC] ...`）
- 失敗したコマンド（`[FAILED] command=...`）
- 失敗終了コード（`[FAILED] exit_code=...`）
- コミットID（`git rev-parse --short HEAD` の結果）

GPT/Codex へは次を貼ると再現しやすいです。

1. `run_<run_id>.log` の末尾 200〜400 行
2. `run_id`
3. 実行した BAT コマンド（例: `scripts\run_all_local_then_copy.bat`）
4. `git rev-parse --short HEAD` の値

## 補助コマンド

```bat
scripts\doctor.bat
scripts\copy_run_to_onedrive.bat C:\work\apex_work\output\sim\SOXL\2022-01-03_20260322_001
```
