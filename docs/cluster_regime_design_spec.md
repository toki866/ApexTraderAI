# クラスタ局面設計仕様（DPrimeCluster）

## 1. 目的
本仕様は、ApexTraderAI のクラスタを「当日1点のラベル」ではなく**連続窓で定義される局面（regime）**として扱うための正式設計を定義する。  
主目的は以下。
- StepE / StepF へ局面文脈を渡す
- rare 局面を監視する
- live 運用の慎重モード判定を補助する
- expert/router 判断の補助信号を提供する

> クラスタは価格予測器ではない。将来予測そのものではなく、過去〜現在の推移の要約ラベルである。

## 2. 全体位置づけ（責務フロー）
外部互換フローは維持する。

`StepA -> DPrimeCluster -> cluster_id(raw20/stable) -> StepB -> StepC -> DPrimeRL -> StepE -> StepF`

- 外部ステップ名は従来どおり `DPRIME`。
- 内部責務として `DPrimeCluster`（クラスタ文脈）と `DPrimeRL`（RL state）を分離する。

## 3. なぜ StepB をクラスタに使わないか
クラスタは**局面要約**であり、予測器出力を混ぜるとリークしやすい。  
従って、クラスタ生成入力は StepA 由来のみとする。

- 使用可: `prices`, `periodic`, `tech`, および StepA 由来派生特徴
- 使用不可: StepB予測、StepC校正予測、StepE/StepF reward 系、未来情報

## 4. 多時間軸窓の思想
クラスタ特徴は「単日1行特徴」ではなく、複数窓の連続要約で作る。
- 短期: 日足窓 (`cluster_short_window_days`)
- 中期: 週足相当窓 (`cluster_mid_window_weeks`)
- 長期: 月足相当窓 (`cluster_long_window_months`)
- 超長期: 8年背景スカラー

## 5. 8年背景スカラー
8年系列を巨大テンソルとして直接投入しない。軽量スカラー群で超長期文脈を保持する。

代表例:
- `ctx_8y_high_distance`
- `ctx_8y_low_distance`
- `ctx_8y_range_position`
- `ctx_8y_drawdown`
- `ctx_8y_vol_percentile`
- `ctx_8y_return_rank`
- `ctx_8y_trend_strength_scalar`

## 6. backend 方針
本命 backend は `TICC`。理由は連続状態分割（regime segmentation）との整合性。  
ただし実装構造は backend 差し替え可能にする。

- `cluster_backend: ticc`（正式本命）
- 将来差し替え候補: hdbscan / gmm / spectral / custom
- 現在は placeholder backend を使用（`status: placeholder`, `not yet wired` を明記）

## 7. raw20 / stable 二系統
正式に二系統を採用。

1. `raw20`
   - `k=20` 固定（`cluster_raw_k`）
   - 高解像度
   - rare 局面検知を保持
2. `stable`
   - raw20 の small cluster 整理後に再学習/再構成
   - 主採用ラベル

## 8. small cluster 判定
small cluster 判定は両条件AND。
- `share < cluster_small_share_threshold`（既定 0.01）
- `mean_run_length < cluster_small_mean_run_threshold`（既定 3 営業日）

## 9. k_eff の決め方
- `k_valid`: raw20 で small でないクラスタ数
- `k_eff = max(cluster_k_eff_min, k_valid)`
- 既定の下限は `cluster_k_eff_min = 12`

## 10. 月1再学習 / 日次 assign
- 月末引け後に再学習（raw20 -> small判定 -> stable）
- 月中は日次 assign のみ
- 日次付与: `cluster_id_raw20`, `cluster_id_stable`, `rare_flag_raw20`

## 11. stable 主採用 / raw20 rare監視
- 主採用: `cluster_id_stable`
- 補助: `cluster_id_raw20`, `rare_flag_raw20`
- rare が立っても main label を raw20 へ切替しない

## 12. rare_flag_raw20 の意味
最小実装として、raw20 側で small/rare 扱い cluster へ所属した日を `rare_flag_raw20=1` とする。  
将来拡張で以下を追加可能。
- raw20/stable 対応の不安定性
- assign confidence / distance 悪化

## 13. sim / live の使い分け
- `sim`: 学習評価時の cluster 文脈付与
- `live`: 月末再学習 + 月中日次assign
- 両モードとも外部 API/step 名は `DPRIME` 互換を維持

## 14. StepE / StepF 受け渡し
DPrimeRL state に以下を渡せる境界を維持。
- `cluster_id_stable`（main）
- `rare_flag_raw20`（補助）
- 将来: transition/confidence/distance 系

StepF 側の正式責務もここで固定する。
- StepF は `DPrimeCluster` が出した `cluster_id_stable` / `cluster_id_raw20` / `rare_flag_raw20` を **consume** する。
- StepF は cluster/regime を新規学習しない。
- StepF は StepE expert 本体ではなく `stepE_daily_log_*` の日次 `ratio` / `ret` を使って expert 群を束ねる。
- sim / live ともに cluster 正本は `DPrimeCluster` の月1再学習 + 月中日次 assign を使い、StepF はその文脈を apply するだけに留める。

## 15. artifact / 保存仕様
cluster 成果物の正規配置は以下。

- `stepDprime/cluster/<mode>/models/raw20/`
- `stepDprime/cluster/<mode>/models/stable/`
- `stepDprime/cluster/<mode>/cluster_assignments_<SYMBOL>.csv`
- `stepDprime/cluster/<mode>/cluster_summary_<SYMBOL>.json`
- `stepDprime/cluster/<mode>/cluster_mapping_raw20_to_stable_<SYMBOL>.json`
- `stepDprime/cluster/<mode>/cluster_feature_manifest_<SYMBOL>.json`

## 16. logging 方針
少なくとも以下を出力する。
- `DPrimeCluster start/end`
- `cluster_backend`
- `raw20 training start/end`
- `stable training start/end`
- `k_raw, k_valid, k_eff`
- `small cluster list`
- `cluster_id_raw20, cluster_id_stable, rare_flag_raw20`
- `cluster artifact paths`

未接続項目は `planned` / `placeholder` / `not yet wired` と明示。

## 17. 未実装 / 将来拡張
- TICC 本実装（現在は placeholder backend）
- 月次 refit の厳密スケジューラ連携
- rare 判定の confidence/distance 連動
- stable 再学習ロジックの本実装
