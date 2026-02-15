# Windows BAT 運用ガイド（ローカル計算 → 完了後 OneDrive コピー）

## 追加されたスクリプト

- `scripts\run_all_local_then_copy.bat`
  - メイン実行。`run_id` 付きのローカル作業ディレクトリに `data/output/logs` を作成し、StepA〜F 完了後に OneDrive へまとめてコピーします。
- `scripts\doctor.bat`
  - 実行前診断。`git hash`、`python --version`、`torch/cuda`、`data\prices_*.csv` の存在と更新日時をログ化します。
- `scripts\copy_run_to_onedrive.bat`
  - ローカルの既存 run フォルダを OneDrive に再コピーします。
- `scripts\bat_config.bat`
  - `SYMBOLS/日付範囲/WORK_ROOT` などの既定値を集約しています。

## 事前準備

1. リポジトリを pull
2. `pip install -r requirements.txt`
3. 必要なら `scripts\bat_config.bat` の既定値を編集

## 実行方法

```bat
scripts\run_all_local_then_copy.bat
```

## ローカル生成物

既定では以下に出力されます。

- `C:\work\apex_work\runs\<run_id>\data`
- `C:\work\apex_work\runs\<run_id>\output`
- `C:\work\apex_work\runs\<run_id>\logs\run_<run_id>.log`
- （既定有効）`C:\work\apex_work\runs\<run_id>\run_<run_id>.zip`

## OneDrive コピー先

優先順:

1. `%ONE_DRIVE_RUNS_ROOT%\<run_id>`
2. `%OneDrive%\ApexTraderAI\runs\<run_id>`

コピーは `robocopy /E /Z /R:2 /W:2` を使い、戻り値 `0..7` を成功、`8+` を失敗として扱います。

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
scripts\copy_run_to_onedrive.bat C:\work\apex_work\runs\<run_id>
```
