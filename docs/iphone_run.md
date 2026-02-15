# iPhone からデスクトップ実行する手順（GitHub Actions 手動実行）

## 前提（必須）

- デスクトップ PC（Windows）が起動中で、self-hosted runner が **Online** であること
- runner 実行ユーザーが OneDrive にサインイン済みであること
- 本ワークフローは `workflow_dispatch` 専用（手動実行のみ）

## 1. iPhone で Workflow を起動

1. GitHub（アプリまたはブラウザ）で対象リポジトリを開く。
2. **Actions** タブを開く。
3. **Run Desktop Pipeline** ワークフローを選ぶ。
4. 右上の **Run workflow** を押す。
5. 必要に応じて入力値（`mode`, `symbols`, `start`, `end`, `test_start`, `train_years`, `test_months`, `copy_to_onedrive`）を調整して実行する。
   - 現時点では入力値は将来連携用に保持しており、既存 BAT 実行を優先する。

## 2. 実行中ログの見方

- Actions 実行詳細で `self-hosted, windows` runner のジョブを開く。
- `Run desktop BAT` ステップで、`scripts\run_all_local_then_copy.bat` の実行ログを確認。
- 正常終了時はログに `[OK] run_id=...` が出る。

## 3. 完了後の成果物の見方

1. **OneDrive（本命）**
   - `runs/<run_id>/output`
   - `runs/<run_id>/logs`
   - `runs/<run_id>/run_<run_id>.zip`（ZIP 有効時）
2. **Actions Artifacts（保険）**
   - `desktop-run-<run_id>` にコンソールログと run ログ（可能なら zip）を保存。
   - ワークフロー失敗時でも `if: always()` で収集を試行。

## 4. 失敗時の確認ポイント

1. `Run desktop BAT` の失敗コマンドを確認。
2. Artifacts から `run_*.log` をダウンロードして確認。
3. OneDrive 側に `logs` があれば同一 run のログを確認。

## 運用・セキュリティ注意

- self-hosted runner は任意コードを実行し得るため、公開リポジトリ運用時は特に注意する。
- 本ワークフローは手動実行のみ（PR などから自動起動しない）。
- `concurrency` で同時実行を防止（`cancel-in-progress: false`）している。
