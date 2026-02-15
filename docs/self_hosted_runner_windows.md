# Windows self-hosted runner セットアップ手順

## 1. Runner を追加

1. GitHub リポジトリの **Settings** → **Actions** → **Runners** を開く。
2. **New self-hosted runner** を選ぶ。
3. OS は **Windows** を選択し、表示された手順どおりに runner を登録する。
4. ラベルは最低限 `self-hosted`, `windows` を付与する（必要なら専用ラベルも追加）。

## 2. OneDrive 前提（重要）

このパイプラインは成果物を OneDrive にコピーするため、runner は次を満たす必要があります。

- **OneDrive にサインイン済みユーザー**で実行する。
- `%OneDrive%` が利用できない運用（サービス化、別ユーザー等）の場合は、
  `ONE_DRIVE_RUNS_ROOT` を固定パスで設定する。
  - 例: `C:\Users\<user>\OneDrive\ApexTraderAI\runs`

## 3. Runner 側の前提

- デスクトップ PC が起動しており、runner が **Online** であること
- リポジトリを checkout 可能であること
- Python 実行環境があること（`requirements.txt` 相当）
- `scripts\bat_config.bat` の既定値で問題ないこと
  - とくに `WORK_ROOT`（既定: `C:\work\apex_work\runs`）
  - 必要に応じて `PYTHON_EXE` やシンボル設定

## 4. 動作確認

1. Actions の **Run Desktop Pipeline** を手動実行。
2. runner がジョブを拾うことを確認。
3. 完了後、`C:\work\apex_work\runs\<run_id>` と OneDrive 側 `runs\<run_id>` を確認。
4. 失敗時も Actions Artifacts にログ収集されることを確認。

## 運用・セキュリティ注意

- self-hosted runner は任意コードを実行され得るため、公開リポジトリ運用では権限とブランチ保護を厳格化する。
- この運用は `workflow_dispatch`（手動実行）前提にし、不要な自動トリガーを増やさない。
- 同時実行防止のため workflow 側で `concurrency` を有効化して運用する。
