# Windows Self-hosted Runner Runbook

## 目的
`workflow_dispatch` を押しても GitHub Actions が「処理待ち」のまま進まない事象を、
Windows self-hosted runner 側の運用で再発防止するための手順書です。

## 典型的なオフライン原因
- **PC の電源 OFF**: runner プロセスが起動しておらず、ジョブを受信できません。
- **PC がスリープ状態**: ネットワーク接続が停止し、GitHub から offline 扱いになります。
- **runner 停止**: `actions-runner` のサービスまたは `run.cmd` が停止しています。

## まず確認すること
1. GitHub リポジトリの **Settings → Actions → Runners** で対象 runner が `Idle` か `Offline` か確認。
2. Windows 側で以下を確認。
   - `hostname`
   - `Get-Date`
   - `Get-Process | ? { $_.Name -like '*Runner*' }`
3. runner が見つからない場合は、以下の復旧手順を実施。

## 復旧手順（優先順）

### A. サービス化済み runner を起動
`actions-runner` ディレクトリで:

```powershell
.\svc start
```

状態確認:

```powershell
Get-Service | ? { $_.Name -like 'actions.runner*' }
Get-Process | ? { $_.Name -like '*Runner*' }
```

### B. 未サービス化なら service install + start
`actions-runner` ディレクトリで:

```powershell
.\svc install
.\svc start
```

> `svc` が使える構成では、この方式を第一候補にしてください（OS 起動時に自動復旧しやすいため）。

### C. `svc` が無い/使えない場合（代替）
`run.cmd` を **タスク スケジューラ** に登録し、**「コンピューターの起動時」** に実行。

推奨設定:
- トリガー: 「コンピューターの起動時」
- 操作: `actions-runner\run.cmd`
- 実行ユーザー: runner 用ユーザー
- 「最上位の特権で実行する」を有効化
- 「タスクが失敗した場合、再試行」を有効化

## 電源設定の注意（再発防止の要点）
- **スリープを無効化**（AC 接続時は「スリープしない」）。
- ディスプレイ OFF のみ許可（画面消灯は可、PC 本体は稼働維持）。
- 必要に応じて NIC の省電力設定を調整し、ネットワーク断を避ける。

## BAT 失敗時の一次調査ポイント
現在の失敗症状（`Run desktop BAT preflight` 後に `[NG] failed. log=...`）では、
実原因は通常以下に記録されます。

- `C:\work\apex_work\runs\<run_id>\logs\run_<run_id>.log`

GitHub Actions ワークフロー側で、失敗時にこのログ末尾を表示し、
同ログを artifact として回収する実装を併用してください。
