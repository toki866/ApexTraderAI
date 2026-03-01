# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260222_210216\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **OK**

## StepE agents table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 62 | 0.9132 | -0.3416 | 0.0007 | 0.0645 | 0.1842 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 62 | 1.0520 | -0.4190 | 0.0028 | 0.0622 | 0.7254 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 62 | 0.7208 | -0.4637 | -0.0025 | 0.0725 | -0.5419 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 62 | 0.7863 | -0.3664 | -0.0011 | 0.0727 | -0.2442 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 62 | 1.0190 | -0.3920 | 0.0024 | 0.0630 | 0.6153 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 62 | 1.0506 | -0.4175 | 0.0029 | 0.0630 | 0.7263 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 62 | 0.7489 | -0.4844 | -0.0015 | 0.0772 | -0.3090 |  | OK |
| dprime_bnf_h03 | stepE_daily_log_dprime_bnf_h03_SOXL.csv | 62 | 0.9192 | -0.3779 | 0.0018 | 0.0773 | 0.3701 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 62 | 1.0386 | -0.2923 | 0.0024 | 0.0573 | 0.6660 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 62 | 0.9335 | -0.3478 | 0.0012 | 0.0653 | 0.2873 |  | OK |

## StepF router summary
| file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 62 | 0.9842 | -0.2990 | 0.0014 | 0.0547 | 0.4027 | Split missing: evaluated all rows as test | OK |

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9991
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260222_210216\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260222_210216\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "mae": 4.506098778017106,
        "corr": 0.8972053476265942,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.506098778017106,
        "corr": 0.8972053476265942,
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
        "equity_multiple": 0.9131901799659078,
        "max_dd": -0.34162290557548536,
        "mean_ret": 0.0007486272210841119,
        "std_ret": 0.06451383057347476,
        "sharpe": 0.184209937690133,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 62,
        "equity_multiple": 1.0520303076853452,
        "max_dd": -0.4189983272503861,
        "mean_ret": 0.002841558513717881,
        "std_ret": 0.06218400024899346,
        "sharpe": 0.7254011127974217,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 62,
        "equity_multiple": 0.7208161460896721,
        "max_dd": -0.4636831611656064,
        "mean_ret": -0.0024746311208864156,
        "std_ret": 0.07249122641846022,
        "sharpe": -0.5419076643586085,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 62,
        "equity_multiple": 0.7863011669874284,
        "max_dd": -0.3664141179013286,
        "mean_ret": -0.0011186088189377578,
        "std_ret": 0.07271445561488994,
        "sharpe": -0.24420679967246692,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 62,
        "equity_multiple": 1.0190254031095036,
        "max_dd": -0.3919809616480382,
        "mean_ret": 0.002441126730709874,
        "std_ret": 0.0629766820816022,
        "sharpe": 0.6153338697534145,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 62,
        "equity_multiple": 1.050609818921143,
        "max_dd": -0.4175065277711755,
        "mean_ret": 0.002883387078210001,
        "std_ret": 0.06301967185549244,
        "sharpe": 0.7263184575102734,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 62,
        "equity_multiple": 0.7489097652879553,
        "max_dd": -0.4843750028224374,
        "mean_ret": -0.0015019745840090177,
        "std_ret": 0.07716147068950265,
        "sharpe": -0.30900275922564835,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h03_SOXL.csv",
        "agent": "dprime_bnf_h03",
        "test_days": 62,
        "equity_multiple": 0.9191727059625769,
        "max_dd": -0.37792055554781356,
        "mean_ret": 0.0018021333737358839,
        "std_ret": 0.07729557111842794,
        "sharpe": 0.37011150837889883,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 62,
        "equity_multiple": 1.038644050574431,
        "max_dd": -0.29230154658400664,
        "mean_ret": 0.0024047773010440125,
        "std_ret": 0.057319214143049885,
        "sharpe": 0.6660010391465244,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 62,
        "equity_multiple": 0.9335136965706857,
        "max_dd": -0.3478136672533573,
        "mean_ret": 0.0011828020815887765,
        "std_ret": 0.06534882263053048,
        "sharpe": 0.2873257725654632,
        "status": "OK"
      }
    ]
  },
  "stepF": {
    "status": "OK",
    "summary": "stepF equity logs evaluated",
    "rows": [
      {
        "file": "stepF_equity_marl_SOXL.csv",
        "test_days": 62,
        "equity_multiple": 0.9841802524055157,
        "max_dd": -0.29900598952680346,
        "mean_ret": 0.0013875035945074383,
        "std_ret": 0.05469689583105594,
        "sharpe": 0.402690799742668,
        "status": "OK",
        "note": "Split missing: evaluated all rows as test"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.9990872379244699,
    "max_match_ratio": 0.0,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK"
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.