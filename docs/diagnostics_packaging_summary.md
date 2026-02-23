# package_diagnostics_to_onedrive.ps1 要約

対象: `scripts/package_diagnostics_to_onedrive.ps1`

## diag zip の出力先

- ZIP名は `"diag_${runId}_${attempt}_${timestamp}.zip"` で生成。`$runId` / `$attempt` / `$timestamp` を含む。`[L127]`
- ZIPのフルパスは `"$zipPath = Join-Path $DiagDir $zipName"` で決定。`[L128]`
- `DiagDir` 自体は `"$DiagDir = [System.IO.Path]::GetFullPath($DiagDir)"` で絶対化し、`"New-Item -Path $DiagDir -ItemType Directory -Force"` で作成保証。`[L60-L61]`

## zip 作成処理の成否判定（try/catch/exit code）

- 主系ZIP作成は `try { Compress-Archive -Path $zipSources -DestinationPath $zipPath -Force -ErrorAction Stop }`。`[L204-L205]`
- 主系失敗時は `catch` で `Write-Warning` と `Add-PublishSummaryWarning` を実行し、フォールバック素材で再度 `Compress-Archive`。`[L206-L230]`
- フォールバックも失敗した場合は `throw "Fallback diagnostics archive failed: ..."` で終了。`[L229-L230]`
- 作成後の最終成否は `if (!(Test-Path $zipPath)) { ...; throw "zip not created: $zipPath" }` で判定。`[L234-L237]`
- ※このスクリプト内で ZIP 作成成否に `Compress-Archive` の **exit code (`$LASTEXITCODE`) 判定は未使用**。`$LASTEXITCODE` は `robocopy` の戻り値 (`$diagSnapshotRc`) 記録用途のみ。`[L139-L141]`

## KeepLatest / _to_delete の移動条件

- `scripts/package_diagnostics_to_onedrive.ps1` 内には `KeepLatest` / `_to_delete` に関する変数・条件式・移動処理（`Move-Item` 等）は存在しない（該当ロジックなし）。
