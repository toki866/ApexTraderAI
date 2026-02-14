# ApexTraderAI

## 実行方法（GitHub / ローカル共通）

実行エントリーポイントは `tools/run_pipeline.py` に統一しています。

```bash
python tools/run_pipeline.py --symbol SOXL --test-start 2022-01-03
```

- 既存の `run_*.py` は後方互換ラッパで、内部的に同じパイプライン実行に委譲します。
- 相対パス（`config/`, `data/`, `output/`）はリポジトリルート基準で解決されるため、実行時 CWD に依存しません。

## CI

GitHub Actions (`.github/workflows/ci.yml`) で以下を実行します。

1. `pip install -r requirements.txt`
2. 全 `.py` ファイルに対する `python -m py_compile`
3. 主要モジュールの import smoke check
