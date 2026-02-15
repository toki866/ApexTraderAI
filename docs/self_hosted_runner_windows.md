# Windows self-hosted runner セットアップ手順

## 1. Runner を追加

1. GitHub リポジトリの **Settings** → **Actions** → **Runners** を開く。
2. **New self-hosted runner** を選択。
3. OS は **Windows** を選び、表示された手順どおりに runner を登録。
4. ラベルは最低限 `self-hosted`, `windows` を付与（必要なら専用ラベルも追加）。

## 2. 重要: OneDrive サインイン済みユーザーで動かす

このパイプラインは最終成果物を OneDrive へコピーします。

- runner は **OneDrive にサインイン済みユーザーのセッション**で実行すること。
- サービス実行だと `OneDrive` 環境変数が無い／同期先が見えない場合があります。
- 必要に応じて `ONE_DRIVE_RUNS_ROOT` を固定パス（例: `C:\Users\<user>\OneDrive\ApexTraderAI\runs`）で設定してください。

## 3. Runner 側の前提

- リポジトリを実行可能な場所に checkout できること
- Python 実行環境があること（`requirements.txt` 相当が導入済み）
- `scripts\bat_config.bat` の既定値で問題ないこと
  - とくに `WORK_ROOT`（既定: `C:\work\apex_work\runs`）
  - 必要なら `PYTHON_EXE` やシンボル設定

## 4. 動作確認

1. Actions の **Run Desktop Pipeline** を手動実行。
2. runner がジョブを拾うことを確認。
3. 完了後、`C:\work\apex_work\runs\<run_id>` と OneDrive 側 `runs\<run_id>` を確認。
