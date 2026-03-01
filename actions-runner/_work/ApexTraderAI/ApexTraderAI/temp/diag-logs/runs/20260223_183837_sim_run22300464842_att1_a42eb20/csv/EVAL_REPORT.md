# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260223_183812\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## D' (stepD_prime) artifacts
- status: **WARN**
- summary: D' missing: state, embeddings
- state_count: 0
- embeddings_count: 0

## StepA table
| status | summary | test_rows | test_date_start | test_date_end | missing_ohlcv_count |
|---|---|---:|---|---|---:|
| OK | prices_test evaluated | 62 | 2022-01-03 | 2022-03-31 | 0 |

## StepB table
- SKIP: no stepB prediction files found

## StepE table
- SKIP: stepE_daily_log missing

## StepF table
- SKIP: stepF_equity_marl missing

## Diversity
- status: **SKIP**
- summary: not evaluated
- max_corr: NA
- max_match_ratio: NA
- pairs_over_0_9999: NA / NA
- identical_all_agents: NA

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
  - note: PLOT StepE topN: stepE_daily_log files were not found
  - note: PLOT StepF equity: stepF_equity_marl file not found
  - note: PLOT StepE return bar: no numeric StepE rows
  - note: PLOT StepE scatter: no numeric StepE DD/return pairs
  - note: StepF_vs_best_StepE: 比較に必要な equity_multiple が不足

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260223_183812\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260223_183812\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
      "test_rows": 62,
      "test_date_start": "2022-01-03",
      "test_date_end": "2022-03-31",
      "missing_ohlcv_count": 0,
      "ohlcv_missing": {
        "Open": 0,
        "High": 0,
        "Low": 0,
        "Close": 0,
        "Volume": 0
      }
    }
  },
  "stepB": {
    "status": "SKIP",
    "summary": "no stepB prediction files found",
    "rows": [],
    "files": []
  },
  "dprime": {
    "status": "WARN",
    "summary": "D' missing: state, embeddings",
    "details": {
      "state_count": 0,
      "embeddings_count": 0,
      "state_files": [],
      "embeddings_files": [],
      "searched": [
        "C:\\work\\apex_work\\runs\\20260223_183812\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\20260223_183812\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\20260223_183812\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
      ]
    }
  },
  "stepE": {
    "status": "SKIP",
    "summary": "stepE_daily_log missing",
    "rows": []
  },
  "stepF": {
    "status": "SKIP",
    "summary": "stepF_equity_marl missing",
    "rows": []
  },
  "diversity": {
    "status": "SKIP",
    "summary": "not evaluated"
  },
  "overall_status": "WARN",
  "plots": {
    "items": [
      {
        "name": "equity_stepE_topN.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepE_topN.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepE_return.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepE_return.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "scatter_stepE_dd_vs_ret.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\scatter_stepE_dd_vs_ret.png",
        "exists": true,
        "reason": null
      }
    ],
    "notes": [
      "PLOT StepE topN: stepE_daily_log files were not found",
      "PLOT StepF equity: stepF_equity_marl file not found",
      "PLOT StepE return bar: no numeric StepE rows",
      "PLOT StepE scatter: no numeric StepE DD/return pairs",
      "StepF_vs_best_StepE: 比較に必要な equity_multiple が不足"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.