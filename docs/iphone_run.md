# iPhone からデスクトップ実行する手順（GitHub Actions 手動実行）

## 1. iPhone で Workflow を起動

1. GitHub（アプリまたはブラウザ）で対象リポジトリを開く。
2. **Actions** タブを開く。
3. **Run Desktop Pipeline** ワークフローを選ぶ。
4. **Run workflow** を押して実行する（通常は `main` ブランチ）。

## 2. 実行中の確認

- Actions の実行詳細で、`self-hosted, windows` runner 上でジョブが動いていることを確認。
- `Run desktop BAT` ステップのログに、最終的に `[OK] run_id=...` が出ることを確認。

## 3. 成果物の確認先

優先度順で以下を確認します。

1. **OneDrive（本命）**
   - `runs/<run_id>/output`
   - `runs/<run_id>/logs`
   - `runs/<run_id>/run_<run_id>.zip`（ZIP有効時）
2. **Actions Artifacts（保険）**
   - `desktop-run-<run_id>` に実行ログと ZIP（存在時）が保存されます。
3. **Actions のステップログ（速報）**
   - BAT の標準出力・エラーを即時確認できます。

## 4. 失敗時の見方

1. Actions の `Run desktop BAT` で失敗コマンドを確認。
2. Artifacts から `run_<run_id>.log`（取得できた場合）をダウンロード。
3. OneDrive 側に `logs` が出ていれば同じログを参照。
