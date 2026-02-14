from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

from ai_core.utils.paths import resolve_repo_path
from typing import Dict, Tuple, Optional

from zoneinfo import ZoneInfo  # Python 3.11 なら標準ライブラリ

from engine.live_policy_runner import LivePolicyConfig, LivePolicyRunner
from engine.daily_trading_orchestrator import DailyTradingOrchestrator

# 以下は「実装済み前提」のクラス
# ユーザー側で engine/state_builder.py, engine/broker_client.py などとして用意してください。
from engine.state_builder import StateBuilder           # StateBuilderProtocol 実装クラス
from engine.broker_client import BrokerClient           # BrokerClientProtocol 実装クラス
from engine.pnl_calculator import PnLCalculator         # PnLCalculatorProtocol 実装クラス
from engine.trade_logger import TradeLogger             # TradeLoggerProtocol 実装クラス


logger = logging.getLogger(__name__)


# =========================================================
# 市場時間・タイムゾーン関連（サマータイム吸収）
# =========================================================

MARKET_TZ = ZoneInfo("America/New_York")

# 市場の標準的なオープン / クローズ（NY時間）
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)

# 各フローの実行ウィンドウ幅（分）
MORNING_WINDOW_MINUTES = 30    # 9:30〜10:00
CLOSE_WINDOW_MINUTES = 30      # 15:30〜16:00

# 実行フラグ保存先
FLAGS_FILE_PATH = resolve_repo_path("output") / "auto_trader_flags.txt"


# =========================================================
# ユーティリティ
# =========================================================

def parse_date(value: str) -> date:
    """
    YYYY-MM-DD 形式の文字列を datetime.date に変換する。
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date format (expected YYYY-MM-DD): {value}") from e


def parse_policy_map(value: str) -> Dict[str, Path]:
    """
    "xsr=path/to/xsr.npz;mamba=path/to/mamba.npz" のような文字列を
    { "xsr": Path(...), "mamba": Path(...) } 辞書に変換する。
    """
    result: Dict[str, Path] = {}
    if not value:
        return result

    entries = value.split(";")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"Invalid policy-map entry (expected name=path): {entry}"
            )
        name, path_str = entry.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip()).resolve()
        if not name:
            raise argparse.ArgumentTypeError(f"Invalid agent name in policy-map: {entry}")
        result[name] = path
    if not result:
        raise argparse.ArgumentTypeError("policy-map is empty.")
    return result


def parse_weights_map(value: str) -> Dict[str, float]:
    """
    "xsr=0.5;mamba=0.3;fed=0.2" のような文字列を
    { "xsr": 0.5, "mamba": 0.3, "fed": 0.2 } に変換する。
    """
    result: Dict[str, float] = {}
    if not value:
        return result

    entries = value.split(";")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"Invalid marl-weights entry (expected name=weight): {entry}"
            )
        name, weight_str = entry.split("=", 1)
        name = name.strip()
        try:
            w = float(weight_str.strip())
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid float weight in marl-weights: {entry}"
            ) from e
        result[name] = w
    if not result:
        raise argparse.ArgumentTypeError("marl-weights is empty.")
    return result


# =========================================================
# 引数パーサ
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live auto-trading launcher (auto / morning / close / full_day)."
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "morning", "close", "full_day"],
        default="auto",
        help=(
            "Which flow to run: "
            "auto (NY時間から自動判定), morning (open), close (pre-close), or full_day (both). "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--date",
        type=parse_date,
        default=None,
        help=(
            "Trading date in YYYY-MM-DD. "
            "If omitted in auto mode, NY時間の現在日付を使用。"
        ),
    )

    # LivePolicyRunner / RL 関連
    parser.add_argument(
        "--symbol",
        type=str,
        default="SOXL",
        help="Base symbol for the RL policy (default: SOXL).",
    )
    parser.add_argument(
        "--mode-rl",
        choices=["single", "marl"],
        default="single",
        help="RL mode: single or marl (default: single).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help=(
            "Policy file path for single mode (.npz or .zip). "
            "If --mode-rl=single and --policy-map is not given, this is required."
        ),
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="xsr",
        help="Agent name used in single mode (default: xsr).",
    )
    parser.add_argument(
        "--policy-map",
        type=parse_policy_map,
        default=None,
        help=(
            "Policy map for MARL or advanced usage in single mode. "
            'Format: "xsr=path/to/xsr.npz;mamba=path/to/mamba.npz;fed=path/to/fed.npz"'
        ),
    )
    parser.add_argument(
        "--marl-weights",
        type=parse_weights_map,
        default=None,
        help=(
            "Weights for MARL mode. "
            'Format: "xsr=0.5;mamba=0.3;fed=0.2". '
            "If omitted in marl mode, equal weights will be used."
        ),
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=24,
        help="Dimension of observation vector for RL agent (default: 24).",
    )
    parser.add_argument(
        "--log-decisions",
        action="store_true",
        help="If set, LivePolicyRunner will log decisions to CSV.",
    )
    parser.add_argument(
        "--decisions-log-path",
        type=Path,
        default=resolve_repo_path("output/live_decisions.csv"),
        help="Path to log RL decisions if --log-decisions is set.",
    )

    # トレード運用パラメータ
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=1.0,
        help="Max leverage used when ratio=±1.0 (default: 1.0 = 100% of equity).",
    )

    # ログ設定
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )

    return parser


# =========================================================
# ログ・フラグ関連
# =========================================================

def setup_logging(level: str) -> None:
    """
    ロギング設定を簡単に行う。
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_flags(path: Path) -> Dict[str, str]:
    """
    実行済みフラグを "key=value" 形式のテキストから読み込む。
    key: "YYYY-MM-DD_morning" / "YYYY-MM-DD_close" など
    value: "done"
    """
    flags: Dict[str, str] = {}
    if not path.exists():
        return flags

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        flags[key.strip()] = value.strip()
    return flags


def _save_flags(path: Path, flags: Dict[str, str]) -> None:
    """
    実行済みフラグ辞書をテキストファイルに保存する。
    """
    lines = [f"{k}={v}" for k, v in flags.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def mark_flow_done(trading_date: date, flow: str) -> None:
    """
    指定日の指定フロー(morning/close)を実行済みとしてフラグ保存。
    """
    key = f"{trading_date.isoformat()}_{flow}"
    flags = _load_flags(FLAGS_FILE_PATH)
    flags[key] = "done"
    _save_flags(FLAGS_FILE_PATH, flags)


def is_flow_done(trading_date: date, flow: str) -> bool:
    """
    指定日の指定フロー(morning/close)が実行済みかどうか。
    """
    key = f"{trading_date.isoformat()}_{flow}"
    flags = _load_flags(FLAGS_FILE_PATH)
    return flags.get(key) == "done"


# =========================================================
# auto モード用: NY時間からどのフローを走らせるか判定
# =========================================================

def _build_time_range(base: datetime, center_time: time, window_minutes: int) -> Tuple[datetime, datetime]:
    """
    base（NY時間の現在日時）を元に、当日の center_time を中心とした
    [center - window/2, center + window/2] の時間帯を作る。
    """
    center = base.replace(
        hour=center_time.hour,
        minute=center_time.minute,
        second=center_time.second,
        microsecond=center_time.microsecond,
    )
    half = timedelta(minutes=window_minutes / 2)
    return center - half, center + half


def decide_auto_flow(now_ny: datetime) -> Tuple[Optional[str], date]:
    """
    現在の NY 時間から、どのフローを実行すべきかを判定する。
    戻り値:
        (flow, trading_date)
        flow: "morning" / "close" / None
        trading_date: そのフローに対する取引日
    """
    trading_date = now_ny.date()

    # 土日なら何もしない
    if now_ny.weekday() >= 5:  # 5=土, 6=日
        logger.info("NY weekday=%d (weekend). No trading.", now_ny.weekday())
        return None, trading_date

    # 9:30 周辺 = 朝フロー
    morning_start, morning_end = _build_time_range(
        now_ny, MARKET_OPEN_TIME, MORNING_WINDOW_MINUTES
    )
    # 16:00 周辺 = 引け前フロー
    close_start, close_end = _build_time_range(
        now_ny, MARKET_CLOSE_TIME, CLOSE_WINDOW_MINUTES
    )

    if morning_start <= now_ny <= morning_end:
        return "morning", trading_date
    if close_start <= now_ny <= close_end:
        return "close", trading_date

    return None, trading_date


# =========================================================
# LivePolicyConfig 構築
# =========================================================

def build_live_policy_config_from_args(args: argparse.Namespace) -> LivePolicyConfig:
    """
    CLI引数から LivePolicyConfig を構築する。
    """
    # mode-rl = single / marl
    mode_rl = args.mode_rl

    if mode_rl == "single":
        # single の場合:
        # 1. policy-map が指定されていれば、それをそのまま使う（agent_names/paths）
        # 2. 無ければ --policy と --agent-name の組を使う
        if args.policy_map:
            policy_paths = args.policy_map
            agent_names = list(policy_paths.keys())
        else:
            if args.policy is None:
                raise ValueError(
                    "--mode-rl=single の場合、--policy または --policy-map のどちらかは必須です。"
                )
            policy_paths = {args.agent_name: args.policy.resolve()}
            agent_names = [args.agent_name]

        live_cfg = LivePolicyConfig(
            symbol=args.symbol,
            mode="single",
            policy_paths=policy_paths,
            agent_names=agent_names,
            marl_weights=None,  # single では未使用
            action_type="ratio",  # v1 は ratio を前提
            obs_dim=args.obs_dim,
            device="cpu",
            log_decisions=args.log_decisions,
            decisions_log_path=args.decisions_log_path if args.log_decisions else None,
        )
        return live_cfg

    # mode_rl == "marl"
    if not args.policy_map:
        raise ValueError(
            "--mode-rl=marl の場合、--policy-map は必須です。"
        )
    policy_paths = args.policy_map
    agent_names = list(policy_paths.keys())
    marl_weights = args.marl_weights or None

    live_cfg = LivePolicyConfig(
        symbol=args.symbol,
        mode="marl",
        policy_paths=policy_paths,
        agent_names=agent_names,
        marl_weights=marl_weights,
        action_type="ratio",  # v1 は ratio を前提
        obs_dim=args.obs_dim,
        device="cpu",
        log_decisions=args.log_decisions,
        decisions_log_path=args.decisions_log_path if args.log_decisions else None,
    )
    return live_cfg


# =========================================================
# メイン
# =========================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ログ設定
    setup_logging(args.log_level)
    logger.info("auto_trader.py started with args: %s", vars(args))

    # NY時間の現在時刻
    now_ny = datetime.now(MARKET_TZ)
    logger.info("Current NY time: %s", now_ny.isoformat())

    # LivePolicyConfig 構築
    live_cfg = build_live_policy_config_from_args(args)
    logger.info(
        "LivePolicyConfig prepared: symbol=%s, mode=%s, agents=%s",
        live_cfg.symbol,
        live_cfg.mode,
        live_cfg.agent_names,
    )

    # LivePolicyRunner 構築
    live_runner = LivePolicyRunner.from_config(live_cfg)

    # 依存コンポーネントの生成
    state_builder = StateBuilder()               # StateBuilderProtocol 実装
    broker = BrokerClient()                      # BrokerClientProtocol 実装
    pnl_calculator = PnLCalculator()             # PnLCalculatorProtocol 実装
    trade_logger = TradeLogger()                 # TradeLoggerProtocol 実装

    # DailyTradingOrchestrator 構築
    orchestrator = DailyTradingOrchestrator(
        symbol=live_cfg.symbol,
        live_runner=live_runner,
        state_builder=state_builder,
        broker=broker,
        pnl_calculator=pnl_calculator,
        trade_logger=trade_logger,
        max_leverage=args.max_leverage,
    )

    mode = args.mode

    # ==============================
    # auto モード: NY時間から実行フローを自動判定
    # ==============================
    if mode == "auto":
        flow, trading_date = decide_auto_flow(now_ny)

        if flow is None:
            logger.info("No trading flow to run at this time (auto mode). Exiting.")
            return

        # autoモードでは --date が指定されていても、基本は NY 日付を優先。
        # （必要ならここをカスタマイズ可）
        logger.info(
            "Auto mode selected flow=%s, trading_date=%s (NY date).",
            flow,
            trading_date,
        )

        # 1日1回だけ実行したいのでフラグを確認
        if is_flow_done(trading_date, flow):
            logger.info(
                "Flow '%s' for %s is already done. Skipping.",
                flow,
                trading_date,
            )
            return

        if flow == "morning":
            logger.info("Running morning open flow (auto) for %s", trading_date)
            orchestrator.run_morning_open_flow(trading_date)
            mark_flow_done(trading_date, "morning")
        elif flow == "close":
            logger.info("Running close pre flow (auto) for %s", trading_date)
            orchestrator.run_close_pre_flow(trading_date)
            mark_flow_done(trading_date, "close")
        else:
            logger.warning("Unknown auto-selected flow: %s", flow)

        logger.info("auto_trader.py finished successfully (auto mode).")
        return

    # ==============================
    # 手動モード: morning / close / full_day
    # ==============================
    if args.date is None:
        # date指定がない場合は NY 日付を使う
        trading_date = now_ny.date()
    else:
        trading_date = args.date

    if mode == "morning":
        logger.info("Running morning open flow (manual) for %s", trading_date)
        orchestrator.run_morning_open_flow(trading_date)
    elif mode == "close":
        logger.info("Running close pre flow (manual) for %s", trading_date)
        orchestrator.run_close_pre_flow(trading_date)
    elif mode == "full_day":
        logger.info("Running full day flow (manual) for %s", trading_date)
        orchestrator.run_full_day(trading_date)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info("auto_trader.py finished successfully.")


if __name__ == "__main__":
    main()
