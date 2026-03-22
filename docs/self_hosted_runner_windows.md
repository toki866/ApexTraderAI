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
- `actions/checkout` で `%GITHUB_WORKSPACE%` に repo を展開できること（OneDrive 配下の固定 clone は不要）
- Python 実行環境があること（`requirements.txt` 相当）
- `scripts\bat_config.bat` の既定値で問題ないこと
  - とくに `WORK_ROOT`（既定: `C:\work\apex_work\runs`）
  - 必要に応じて `PYTHON_EXE` やシンボル設定

## 4. 動作確認

1. Actions の **Run Desktop Pipeline** を手動実行。
2. runner がジョブを拾うことを確認。
3. ログ先頭の debug step で `whoami` / `%GITHUB_WORKSPACE%` / `REPO_ROOT` / `git config --global --get-all safe.directory` / `git rev-parse --show-toplevel` が期待どおり出ていることを確認。
4. 完了後、`C:\work\apex_work\runs\<run_id>` と OneDrive 側 `runs\<run_id>` を確認。
5. 失敗時も Actions Artifacts にログ収集されることを確認。

## 5. service account 変更後の注意

- runner service の実行ユーザーを変更したあと（例: `NETWORK SERVICE` → `.\becky`）、`C:\work\actions-runner\_work` 配下に旧所有者の workspace が残ると、checkout 後の `git` が `fatal: detected dubious ownership in repository` で止まることがあります。
- 現行 workflow は checkout 直後に `%GITHUB_WORKSPACE%` を `git config --global --add safe.directory ...` へ追加し、`whoami` と safe.directory 一覧を diagnostics に残します。
- それでも ownership mismatch が続く場合は、runner を停止して該当 workspace (`C:\work\actions-runner\_work\ApexTraderAI\ApexTraderAI`) を削除し、checkout で新規生成させてください。
- `_work` を掃除した後に rerun し、Actions ログで current user と safe.directory 一覧が新しい service account 前提で出ていることを確認してください。

## 運用・セキュリティ注意

- self-hosted runner は任意コードを実行され得るため、公開リポジトリ運用では権限とブランチ保護を厳格化する。
- この運用は `workflow_dispatch`（手動実行）前提にし、不要な自動トリガーを増やさない。
- 同時実行防止のため workflow 側で `concurrency` を有効化して運用する。
