"""
auto_trader.py

ApexTraderAI live 自動売買エントリーポイント。

引け10分前（NY時間 15:50）に Windows Task Scheduler から起動される。
config/live_config.yaml を読み込んで LiveClosePreRunner を実行する。

使用例:
    python auto_trader.py
    python auto_trader.py --date 2026-03-27
    python auto_trader.py --dry-run
    python auto_trader.py --config config/live_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import yaml

from ai_core.live.gateway_state import (
    GatewayState,
    GATEWAY_STATE_FILENAME,
    clear_flag,
    detect_state,
    read_flag,
    write_flag,
)
from engine.broker_client import IBKRBrokerClient, IBKRSettings
from engine.discord_notifier import build_notifier
from engine.live_close_pre_runner import LiveClosePreConfig, LiveClosePreRunner
from engine.pnl_calculator import PnLCalculator
from engine.trade_logger import TradeLogger, TradeLoggerConfig

MARKET_TZ = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)


# =========================================================
# 設定ファイル読み込み
# =========================================================

def load_live_config(config_path: Path) -> Dict[str, Any]:
    """config/live_config.yaml を読み込んで dict を返す。"""
    if not config_path.exists():
        raise FileNotFoundError(
            f"live_config.yaml が見つかりません: {config_path}\n"
            "config/live_config.yaml を確認してください。"
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_runner_config(cfg: Dict[str, Any], dry_run_override: bool = False) -> LiveClosePreConfig:
    """yaml dict から LiveClosePreConfig を生成する。"""
    trading = cfg.get("trading", {})
    paths = cfg.get("paths", {})
    execution = cfg.get("execution", {})

    dry_run = dry_run_override or bool(execution.get("dry_run", False))

    return LiveClosePreConfig(
        symbol_long=str(trading.get("symbol_long", "SOXL")),
        symbol_inverse=str(trading.get("symbol_inverse", "SOXS")),
        sim_output_root=str(paths.get("sim_output_root", "output/sim")),
        live_output_root=str(paths.get("live_output_root", "output/live")),
        neutral_threshold=float(trading.get("neutral_threshold", 0.05)),
        max_position_shares=int(trading.get("max_position_shares", 100)),
        order_type=str(trading.get("order_type", "SAFE_LIMIT")),
        safe_band_entry_pct=float(trading.get("safe_band_entry_pct", 0.01)),
        safe_band_exit_pct=float(trading.get("safe_band_exit_pct", 0.01)),
        audit_log_path=str(paths.get("audit_log", "output/live/live_audit.jsonl")),
        flags_path=str(paths.get("flags_file", "output/live/live_flags.txt")),
        dry_run=dry_run,
        order_wait_sec=int(execution.get("order_wait_sec", 5)),
    )


def build_ibkr_broker(cfg: Dict[str, Any]) -> IBKRBrokerClient:
    """yaml dict から IBKRBrokerClient を生成する。"""
    ibkr = cfg.get("ibkr", {})
    account_id = str(ibkr.get("account_id", ""))
    if not account_id:
        raise ValueError(
            "config/live_config.yaml の ibkr.account_id が未設定です。"
            "DU****** (paper) または U****** (real) を設定してください。"
        )
    settings = IBKRSettings(
        host=str(ibkr.get("host", "127.0.0.1")),
        port=int(ibkr.get("port", 7497)),
        client_id=int(ibkr.get("client_id", 10)),
        account_id=account_id,
        exchange=str(ibkr.get("exchange", "SMART")),
        currency=str(ibkr.get("currency", "USD")),
    )
    return IBKRBrokerClient(settings=settings, logger=logger)


# =========================================================
# Gateway プリフライトチェック
# =========================================================

_GATEWAY_WAIT_INTERVAL_SEC = 30  # 再チェック間隔


def check_gateway_preflight(
    cfg: Dict[str, Any],
    trading_date: "date",
    notifier: Any,
) -> bool:
    """
    取引実行前に IBKR Gateway の接続状態を確認する。

    - CONNECTED     : フラグクリア → True（取引続行）
    - LOGIN_WAIT    : IBGateway 起動中だがポート未開放。
                      gateway_login_wait_timeout_sec 秒まで待機してから再判定。
    - GATEWAY_DOWN  : IBGateway プロセスなし → 即スキップ
    - NETWORK_ERROR : 接続異常 → 即スキップ

    アラート送信の失敗は WARNING ログに留め、スキップ判定には影響しない。
    """
    ibkr = cfg.get("ibkr", {})
    paths = cfg.get("paths", {})
    execution = cfg.get("execution", {})
    host = str(ibkr.get("host", "127.0.0.1"))
    port = int(ibkr.get("port", 7497))
    live_output_root = Path(str(paths.get("live_output_root", "output/live")))
    state_file = live_output_root / GATEWAY_STATE_FILENAME
    login_wait_timeout_sec = int(
        execution.get("gateway_login_wait_timeout_sec", 0)
    )

    # 前回フラグをログ出力（診断用、処理には影響しない）
    prev_flag = read_flag(state_file)
    if prev_flag:
        logger.info("[GatewayPreflight] previous flag: %s", prev_flag)

    state = detect_state(host, port)
    logger.info("[GatewayPreflight] current state: %s", state.value)

    # LOGIN_WAIT のとき、設定秒数まで待機ループ
    if state == GatewayState.LOGIN_WAIT and login_wait_timeout_sec > 0:
        logger.info(
            "[GatewayPreflight] LOGIN_WAIT detected. Waiting up to %d sec for gateway "
            "(interval=%ds) ...",
            login_wait_timeout_sec,
            _GATEWAY_WAIT_INTERVAL_SEC,
        )
        wait_start = time.time()
        while True:
            elapsed = time.time() - wait_start
            remaining = login_wait_timeout_sec - elapsed
            if remaining <= 0:
                logger.warning(
                    "[GatewayPreflight] login wait timeout (%ds). Giving up.",
                    login_wait_timeout_sec,
                )
                break
            sleep_sec = min(_GATEWAY_WAIT_INTERVAL_SEC, remaining)
            logger.info(
                "[GatewayPreflight] waiting... elapsed=%.0fs remaining=%.0fs",
                elapsed,
                remaining,
            )
            time.sleep(sleep_sec)
            state = detect_state(host, port)
            logger.info("[GatewayPreflight] re-check state: %s", state.value)
            if state != GatewayState.LOGIN_WAIT:
                # CONNECTED / GATEWAY_DOWN / NETWORK_ERROR のどれかに変化
                break

    if state == GatewayState.CONNECTED:
        clear_flag(state_file)
        return True

    # CONNECTED 以外 → 取引スキップ
    write_flag(state_file, state)
    logger.warning(
        "[GatewayPreflight] trading SKIPPED: state=%s host=%s:%d",
        state.value, host, port,
    )

    if notifier is not None:
        try:
            notifier.notify_gateway_alert(
                state=state.value,
                trading_date=trading_date,
                host=host,
                port=port,
            )
        except Exception as e:
            logger.warning("[GatewayPreflight] alert send failed: %s", e)

    return False


# =========================================================
# ロギング設定
# =========================================================

def setup_logging(cfg: Dict[str, Any], trading_date: date) -> None:
    """ファイル + コンソール のロギングを設定する。"""
    log_cfg = cfg.get("logging", {})
    level_str = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    log_dir_str = log_cfg.get("log_dir", "output/live/logs")
    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"live_close_pre_{trading_date.isoformat()}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logger.info("Log file: %s", log_file)


# =========================================================
# 取引日決定
# =========================================================

class MarketClosedError(Exception):
    """NYSE 休場日（祝日・土日）に実行された場合に raise する。"""
    pass


def _is_nyse_trading_day(d: date) -> bool:
    """pandas_market_calendars で NYSE の取引日かどうかを判定する。"""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=d.isoformat(),
            end_date=d.isoformat(),
        )
        return not schedule.empty
    except Exception as e:
        logger.warning("NYSE calendar check failed (%s). Assuming trading day.", e)
        return True  # チェック失敗時は実行を止めない


def decide_trading_date(
    date_arg: Optional[date],
    skip_market_check: bool,
) -> date:
    """
    取引日を決定する。

    - date_arg が指定されていればそれを使う（テスト・手動実行用）
    - 指定がなければ NY 時間の現在日付を使う
    - 土日・NYSE 祝日で skip_market_check=False なら MarketClosedError
    """
    now_ny = datetime.now(MARKET_TZ)

    if date_arg is not None:
        trading_date = date_arg
    else:
        trading_date = now_ny.date()

    if not skip_market_check:
        if not _is_nyse_trading_day(trading_date):
            raise MarketClosedError(
                f"{trading_date} は NYSE 休場日です（祝日または土日）。"
                "スキップします。"
            )

    return trading_date


# =========================================================
# CLI 引数パーサ
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ApexTraderAI live 自動売買 — 引け10分前に1回実行する。\n"
            "config/live_config.yaml に IBKR 接続情報を設定してから使用すること。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/live_config.yaml"),
        help="設定ファイルのパス（デフォルト: config/live_config.yaml）",
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="取引日（YYYY-MM-DD）。省略時は NY 時間の今日。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="発注をスキップして動作確認のみ行う（config の dry_run を上書き）。",
    )
    parser.add_argument(
        "--skip-market-check",
        action="store_true",
        help="土日・時間外でも強制実行する（テスト用）。",
    )
    return parser


# =========================================================
# メイン
# =========================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # 設定ファイル読み込み
    cfg = load_live_config(args.config)
    execution = cfg.get("execution", {})
    skip_market_check = args.skip_market_check or bool(
        execution.get("skip_market_check", False)
    )

    # 取引日決定（ロギング前に決める必要がある）
    try:
        trading_date = decide_trading_date(
            date_arg=args.date,
            skip_market_check=skip_market_check,
        )
    except MarketClosedError as e:
        # 休場日は通知なしで正常終了
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logger.info("auto_trader.py: %s スキップして終了します。", e)
        sys.exit(0)

    # ロギング設定
    setup_logging(cfg, trading_date)
    logger.info("auto_trader.py start | trading_date=%s dry_run=%s", trading_date, args.dry_run)

    # notifier を先に生成（Gateway アラートにも使うため setup より前）
    notifier = build_notifier(cfg)

    # Gateway プリフライトチェック
    # CONNECTED 以外の場合はアラートを送信して取引をスキップする。
    # ログインは手動操作のみ（自動化しない）。
    if not check_gateway_preflight(cfg, trading_date, notifier):
        logger.info(
            "auto_trader.py: Gateway not ready. Trading skipped for %s.", trading_date
        )
        sys.exit(0)

    # セットアップ（IBKR クライアント・runner 生成）
    try:
        broker = build_ibkr_broker(cfg)
        runner_config = build_runner_config(cfg, dry_run_override=args.dry_run)

        trade_logger_config = TradeLoggerConfig(
            log_root=Path(runner_config.live_output_root),
            log_file_pattern="live_daily_log_{symbol}.csv",
        )
        trade_logger = TradeLogger(symbol=runner_config.symbol_long, config=trade_logger_config)
        pnl_calculator = PnLCalculator()

        runner = LiveClosePreRunner(
            config=runner_config,
            broker=broker,
            trade_logger=trade_logger,
            pnl_calculator=pnl_calculator,
            notifier=notifier,
        )
    except Exception as setup_exc:
        logger.exception("auto_trader.py setup FAILED: %s", setup_exc)
        if notifier:
            try:
                notifier.notify_failure(
                    trading_date=trading_date,
                    error=f"[setup error] {setup_exc}",
                )
            except Exception:
                pass
        sys.exit(1)

    # t_eff フロー実行
    # （runner.run() 内の finally が成功/失敗を通知する。main() 側は二重通知を避けるため通知しない）
    # ModuleNotFoundError（cloudpickle/numpy 初期化タイミング起因の間欠障害）は
    # broker/runner を再生成して 1 回リトライする。
    _MAX_RETRIES = 1
    _RETRY_WAIT_SEC = 10

    run_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            result = runner.run(trading_date)
            logger.info(
                "auto_trader.py DONE | status=%s ratio=%.4f elapsed=%.1fs",
                result.get("status"),
                float(result.get("ratio_final", 0.0)),
                float(result.get("elapsed_sec", 0.0)),
            )
            run_exc = None
            break

        except ModuleNotFoundError as exc:
            run_exc = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "[Retry %d/%d] ModuleNotFoundError: %s — %d 秒後にリトライします",
                    attempt + 1, _MAX_RETRIES, exc, _RETRY_WAIT_SEC,
                )
                try:
                    broker.close()
                except Exception:
                    pass
                time.sleep(_RETRY_WAIT_SEC)
                try:
                    broker = build_ibkr_broker(cfg)
                    runner = LiveClosePreRunner(
                        config=runner_config,
                        broker=broker,
                        trade_logger=trade_logger,
                        pnl_calculator=pnl_calculator,
                        notifier=notifier,
                    )
                except Exception as rebuild_exc:
                    logger.exception("auto_trader.py retry setup FAILED: %s", rebuild_exc)
                    run_exc = rebuild_exc
                    break
            else:
                logger.exception("auto_trader.py FAILED: %s", exc)

        except Exception as exc:
            run_exc = exc
            logger.exception("auto_trader.py FAILED: %s", exc)
            break

    try:
        broker.close()
    except Exception:
        pass

    sys.exit(1 if run_exc else 0)


if __name__ == "__main__":
    main()
