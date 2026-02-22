# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260223_003634\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## StepE agents table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 62 | 0.5880 | -0.5831 | -0.0053 | 0.0782 | -1.0757 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 62 | 0.8054 | -0.3145 | -0.0029 | 0.0361 | -1.2983 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 62 | 0.4278 | -0.6008 | -0.0119 | 0.0626 | -3.0176 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 62 | 0.4599 | -0.5650 | -0.0108 | 0.0603 | -2.8341 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 62 | 0.4360 | -0.5865 | -0.0112 | 0.0683 | -2.6078 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 62 | 0.7899 | -0.3626 | -0.0030 | 0.0439 | -1.0767 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 62 | 0.5266 | -0.5028 | -0.0088 | 0.0592 | -2.3478 |  | OK |
| dprime_bnf_h03 | stepE_daily_log_dprime_bnf_h03_SOXL.csv | 62 | 0.4506 | -0.5724 | -0.0111 | 0.0612 | -2.8878 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 62 | 0.6075 | -0.4434 | -0.0060 | 0.0665 | -1.4317 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 62 | 0.6282 | -0.4210 | -0.0054 | 0.0663 | -1.2905 |  | OK |

## StepF router summary
- SKIP: stepF_equity_marl missing

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9891
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260223_003634\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260223_003634\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
    "status": "OK",
    "summary": "stepB files evaluated",
    "rows": [
      {
        "file": "stepB_pred_close_mamba_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.4166717529296875,
        "corr": 0.911602446398402,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.4166717529296875,
        "corr": 0.911602446398402,
        "status": "OK"
      }
    ],
    "files": [
      "stepB_pred_close_mamba_SOXL.csv",
      "stepB_pred_path_mamba_SOXL.csv",
      "stepB_pred_time_all_SOXL.csv"
    ]
  },
  "stepE": {
    "status": "OK",
    "summary": "stepE daily logs evaluated",
    "rows": [
      {
        "file": "stepE_daily_log_dprime_all_features_3scale_SOXL.csv",
        "agent": "dprime_all_features_3scale",
        "test_days": 62,
        "equity_multiple": 0.5880120187507911,
        "max_dd": -0.5831221333679317,
        "mean_ret": -0.00530200700591048,
        "std_ret": 0.0782457643789618,
        "sharpe": -1.075671668505011,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 62,
        "equity_multiple": 0.8054043976341367,
        "max_dd": -0.3145022324543466,
        "mean_ret": -0.002948351286777907,
        "std_ret": 0.03605123327148149,
        "sharpe": -1.2982531094672416,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 62,
        "equity_multiple": 0.42784987360184296,
        "max_dd": -0.6008009382859487,
        "mean_ret": -0.011896782803946293,
        "std_ret": 0.06258522805668042,
        "sharpe": -3.0175742434766257,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 62,
        "equity_multiple": 0.45993156092544063,
        "max_dd": -0.5649565973325714,
        "mean_ret": -0.01077060971874743,
        "std_ret": 0.06032874222548871,
        "sharpe": -2.834107299419355,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 62,
        "equity_multiple": 0.4359505479329219,
        "max_dd": -0.5865355867778294,
        "mean_ret": -0.011219414191850033,
        "std_ret": 0.06829542396883694,
        "sharpe": -2.607827413532879,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 62,
        "equity_multiple": 0.7898746258885672,
        "max_dd": -0.3626475886064804,
        "mean_ret": -0.0029797942822164416,
        "std_ret": 0.043932650596704405,
        "sharpe": -1.0767109912737642,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 62,
        "equity_multiple": 0.5266337897937974,
        "max_dd": -0.5028074855695497,
        "mean_ret": -0.008754537011085512,
        "std_ret": 0.05919335626193107,
        "sharpe": -2.3477967026248177,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h03_SOXL.csv",
        "agent": "dprime_bnf_h03",
        "test_days": 62,
        "equity_multiple": 0.4505946437658416,
        "max_dd": -0.5723632963555936,
        "mean_ret": -0.0111415613416372,
        "std_ret": 0.061247243330083426,
        "sharpe": -2.887751244712522,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 62,
        "equity_multiple": 0.6075292110388293,
        "max_dd": -0.44341291843269914,
        "mean_ret": -0.006000824925121696,
        "std_ret": 0.06653440590713981,
        "sharpe": -1.431742587611148,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 62,
        "equity_multiple": 0.6281902619186107,
        "max_dd": -0.42096846056261916,
        "mean_ret": -0.005391248734667893,
        "std_ret": 0.06631726445724208,
        "sharpe": -1.2905149382830334,
        "status": "OK"
      }
    ]
  },
  "stepF": {
    "status": "SKIP",
    "summary": "stepF_equity_marl missing",
    "rows": []
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.9890901996370417,
    "max_match_ratio": 0.0,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "WARN"
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.