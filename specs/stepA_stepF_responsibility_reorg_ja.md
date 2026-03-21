# StepA〜StepF 処理整理仕様（責務分離・互換維持版）

## 目的
本仕様は、ApexTraderAI の StepA〜StepF を **実装責務として明確に分離**しつつ、既存の実行インターフェース（workflow / CLI / 生成物契約）を壊さないための正式な整理方針を定義する。

- 外部互換: 既存フロー `A,B,C,DPRIME,E,F` を維持する。
- 内部整理: DPRIME を `DPrimeCluster` と `DPrimeRL` に責務分離して扱う。
- StepD は deprecated のまま維持し、標準運用では使用しない。

---

## 全体フロー（正式採用）

内部責務としての正規フローは次のとおり。

1. `StepA`
2. `DPrimeCluster`（= `D'_cluster`）
3. `cluster_id`（`raw20` / `stable`）
4. `StepB`
5. `StepC`
6. `DPrimeRL`（= `D'_rl`）
7. `StepE`
8. `StepF`

> ただし外部公開インターフェース（workflow の steps 指定・run_stepDPRIME など）は従来通り `DPRIME` 名で扱う。

---

## 各ステップ責務

## StepA（基盤データ準備）
- 価格・周期・テクニカルの基盤データを生成する。
- train/test 分割済みの基礎材料を作る。
- 予測・意思決定は行わない（判断ロジック非担当）。

## DPrimeCluster（クラスタ判定専用）
- クラスタ判定用 state を作る。
- 使用入力は **StepA 由来のみ**。
- **StepB 予測を絶対に使用しない**（因果逆流防止）。
- 多時間軸（日足/週足/月足/8年背景スカラー）特徴を扱う責務を持つ。
- 出力は、cluster 用 state と daily assign に必要な中間成果物。

## cluster_id（raw20 / stable）
- 本命方式は TICC ベース設計。
- `raw20`: `k=20` の監視系クラスタ。
- `stable`: 小クラスタ統合後の運用系クラスタ。
- small cluster 判定条件:
  - `share < 1%`
  - かつ `mean_run < 3 営業日`
- stable の実効クラスタ数:
  - `k_eff = max(12, k_valid)`
- 月1回の更新（monthly refit）を基本とし、月中は最新窓で daily assign を行う。
- 主採用は `stable`、`raw20` は rare 局面監視に使う。
- `rare_flag_raw20` は、monthly 確定 rare cluster 集合に属する日を 1 とする。

## StepB（予測）
- 将来見通し（予測）を生成する。
- 実行順は cluster_id の後に置く。
- 予測結果はクラスタ作成へ逆流させない。

## StepC（予測整形）
- StepB 出力の整合・校正・再整形を担当。
- DPrimeRL へ入力しやすい形式へ変換する。

## DPrimeRL（RL state 統合）
- RL 入力用 state を構築する。
- 主な入力:
  - StepA の past 特徴
  - `cluster_id_stable`
  - 必要に応じ `rare_flag_raw20`
  - StepB/StepC 由来の予測要約
  - Gap / ATR_norm / pos / action など
- `DPrimeCluster` で作ったクラスタ情報を統合し、StepE 以降へ渡す。

## StepE（candidate generation）
- expert 群（10本前提）の学習・評価。
- 担当は価格予測ではなく action / ratio 専門家。
- 当面は cluster ごと完全分離せず、state に cluster 情報を入れて利用する。

## StepF（candidate selection / integration）
- StepE の expert 群を束ねる最終判断層。
- router / MARL を含む上位統合層。
- StepE が生成した候補を選別・統合する。
- live での最終判断直前レイヤとして扱う。
- **cluster/regime の新規学習は行わない。**
- `DPrimeCluster` が月1更新・月中日次 assign した `cluster_id_stable` を主入力として consume する。
- 補助入力として `cluster_id_raw20` / `rare_flag_raw20` を受け取れる。
- StepF が読むのは expert 本体ではなく、各 expert の `stepE_daily_log_*` にある日次 `ratio` / `ret` である。
- StepF は `chosen_expert` / `selected_expert` と `final_ratio` を最終出力する router 層である。

---

## raw20 / stable / rare_flag の運用意図
- `cluster_id_raw20`:
  - 高分解能（k=20）で異常・希少局面を監視。
  - モデル主経路より「監視・検知」に比重。
- `cluster_id_stable`:
  - 小クラスタを統合した安定運用向け。
  - StepE/StepF の主入力に採用。
- `rare_flag_raw20`:
  - monthly 確定 rare cluster 集合への所属フラグ。
  - rare regime の警戒信号として RL state に連結可能。

---

## sim と live の役割差
- sim:
  - 学習・比較・検証を実施する主環境。
  - 月次更新や cluster 統合ロジックの妥当性検証を重視。
- live:
  - sim で確定した設定を運用に適用。
  - daily assign と最終判断（StepF）を重視。
  - 未配線機能は `planned` / `placeholder` と明示し誤解を避ける。

---

## StepE と StepF の違い（要点）
- StepE: 候補を作る層（candidate generation）。
- StepF: 候補を統合し最終採択する層（selection/integration）。
- よって責務境界は「個別 expert の性能最適化」と「市場最終出力の統合最適化」。
- StepF 内で Phase2 / PCA / HDBSCAN / TICC による再クラスタリングをしてはならず、cluster 正本は常に `DPrimeCluster` 側に置く。

---

## 互換方針
- workflow / CLI の step 名は `DPRIME` を維持。
- `run_stepDPRIME`, `steps`, `resume_from`, `reuse_output`, `force_rebuild` などの既存入力は維持。
- 既存の reuse / manifest / output_root / canonical path 仕組みは維持。
- 既存 StepA〜F の主要出力ファイル名は維持し、必要な補助成果物を追加する方針。
