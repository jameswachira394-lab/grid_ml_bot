"""
train_model.py
──────────────────────────────────────────────────────────────────
Standalone training script for the Grid ML Scalper.

Usage:
    python train_model.py
    python train_model.py --symbol GBPUSD --timeframe M5 --bars 10000
    python train_model.py --symbol EURUSD --timeframe M1 --bars 20000 --export-csv

What it does:
    1. Connects to MT5 using your credentials
    2. Downloads historical OHLCV bars
    3. Engineers features + labels
    4. Trains the Random Forest model
    5. Saves the model to models/ (ready for the live bot)
    6. Prints a full training report
    7. Optionally exports predictions to CSV
    8. Optionally backtests the signals
──────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ── logging ──────────────────────────────────────────────────────
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "train.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────── CLI ARGS ─────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Grid ML Scalper model from MT5 history")
    p.add_argument("--login",     type=int,   default=None,     help="MT5 account login")
    p.add_argument("--password",  type=str,   default=None,     help="MT5 password")
    p.add_argument("--server",    type=str,   default=None,     help="MT5 broker server")
    p.add_argument("--path",      type=str,   default=None,     help="Path to terminal64.exe (optional)")
    p.add_argument("--symbol",    type=str,   default="EURUSD", help="Symbol to train on")
    p.add_argument("--timeframe", type=str,   default="M1",     help="Timeframe: M1 M5 M15 H1 H4 D1")
    p.add_argument("--bars",      type=int,   default=10000,    help="Number of historical bars to fetch")
    p.add_argument("--tp-pips",   type=float, default=4.0,      help="Take-profit pips for labelling")
    p.add_argument("--sl-pips",   type=float, default=6.0,      help="Stop-loss pips for labelling")
    p.add_argument("--horizon",   type=int,   default=10,       help="Bars ahead to look for TP/SL")
    p.add_argument("--export-csv",action="store_true",          help="Export predictions to CSV")
    p.add_argument("--backtest",  action="store_true",          help="Run simple backtest on signals")
    p.add_argument("--config",    type=str,   default=None,     help="Path to JSON config file")
    return p.parse_args()


# ─────────────────────────── CONFIG FILE ──────────────────────────

def load_config(path: str) -> dict:
    """Load credentials from a JSON config file."""
    with open(path) as f:
        return json.load(f)


def get_credentials(args) -> dict:
    """
    Resolve MT5 credentials in order of priority:
      1. CLI args
      2. JSON config file (--config path)
      3. mt5_config.json in the same directory
      4. Interactive prompt
    """
    creds = {"login": None, "password": None, "server": None, "path": None}

    # Try config file
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mt5_config.json"
    )
    if os.path.exists(config_path):
        logger.info(f"Loading credentials from {config_path}")
        cfg = load_config(config_path)
        creds.update(cfg)

    # CLI overrides config
    if args.login:    creds["login"]    = args.login
    if args.password: creds["password"] = args.password
    if args.server:   creds["server"]   = args.server
    if args.path:     creds["path"]     = args.path

    # Interactive fallback
    if not creds["login"]:
        print("\n── MT5 Credentials ─────────────────────────")
        creds["login"]    = int(input("  Account login  : ").strip())
        creds["password"] = input("  Password       : ").strip()
        creds["server"]   = input("  Broker server  : ").strip()
        mt5_path = input("  terminal64.exe path (leave blank if MT5 is running): ").strip()
        if mt5_path:
            creds["path"] = mt5_path
        print("─────────────────────────────────────────────\n")

    return creds


# ─────────────────────────── BACKTEST ─────────────────────────────

def simple_backtest(df, predictions: list, tp_pips: float, sl_pips: float,
                    pip_size: float = 0.0001, lot: float = 0.1) -> dict:
    """
    Walk-forward backtest on the predicted signals.
    Uses the same TP/SL as the labeller — a fair sanity check.
    """
    pip_val = 10.0  # USD per pip per standard lot
    equity  = 0.0
    trades  = []
    wins = losses = neutrals = 0

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(predictions)

    for i, sig in enumerate(predictions):
        if sig == "NEUTRAL" or i >= n - 10:
            neutrals += 1
            continue

        entry = closes[i]
        tp_pts = tp_pips * pip_size
        sl_pts = sl_pips * pip_size

        if sig == "BUY":
            tp_price = entry + tp_pts
            sl_price = entry - sl_pts
        else:  # SELL
            tp_price = entry - tp_pts
            sl_price = entry + sl_pts

        result = "open"
        pnl    = 0.0
        for j in range(i + 1, min(i + 11, n)):
            if sig == "BUY":
                if highs[j] >= tp_price:
                    result = "win"; pnl = tp_pips * pip_val * lot; break
                if lows[j]  <= sl_price:
                    result = "loss"; pnl = -sl_pips * pip_val * lot; break
            else:
                if lows[j]  <= tp_price:
                    result = "win"; pnl = tp_pips * pip_val * lot; break
                if highs[j] >= sl_price:
                    result = "loss"; pnl = -sl_pips * pip_val * lot; break

        if result == "open":
            continue

        equity += pnl
        if result == "win":  wins   += 1
        else:                losses += 1
        trades.append({"idx": i, "signal": sig, "entry": entry, "pnl": round(pnl, 2),
                        "result": result, "equity": round(equity, 2)})

    total  = wins + losses
    wr     = wins / total * 100 if total else 0
    profit_factor = (wins * tp_pips) / (losses * sl_pips + 1e-9)

    return {
        "total_trades":   total,
        "wins":           wins,
        "losses":         losses,
        "neutrals":       neutrals,
        "win_rate":       round(wr, 1),
        "profit_factor":  round(profit_factor, 2),
        "net_pnl_usd":    round(equity, 2),
        "trades":         trades,
    }


# ─────────────────────────── REPORT ───────────────────────────────

def print_report(metrics: dict, backtest_result: dict = None):
    W = 56
    LINE  = "═" * W
    THIN  = "─" * W

    def row(label, value, color=""):
        reset = "\033[0m"
        return f"  {color}{label:<30}{str(value):>20}{reset}"

    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    print(f"\n{BOLD}{CYAN}╔{LINE}╗{RESET}")
    print(f"{BOLD}{CYAN}║{'  GRID ML SCALPER — TRAINING REPORT':^{W}}║{RESET}")
    print(f"{BOLD}{CYAN}╚{LINE}╝{RESET}")

    print(f"\n{BOLD}  MODEL PERFORMANCE{RESET}")
    print(f"  {THIN}")
    acc = metrics['accuracy']
    acc_color = GREEN if acc >= 60 else YELLOW
    print(row("Accuracy", f"{acc}%", acc_color))
    print(row("Features used", metrics['n_features']))
    print(row("Training samples", metrics['n_train']))
    print(row("Test samples", metrics['n_test']))

    print(f"\n{BOLD}  CLASS BREAKDOWN{RESET}")
    print(f"  {THIN}")
    report = metrics.get("report", {})
    for cls in ["BUY", "NEUTRAL", "SELL"]:
        if cls in report:
            r = report[cls]
            print(row(f"  {cls} precision",  f"{r['precision']*100:.1f}%"))
            print(row(f"  {cls} recall",     f"{r['recall']*100:.1f}%"))
            print(row(f"  {cls} F1",         f"{r['f1-score']*100:.1f}%"))
            print(row(f"  {cls} support",    int(r['support'])))
        print()

    print(f"\n{BOLD}  TOP 10 FEATURES{RESET}")
    print(f"  {THIN}")
    for feat in metrics.get("importances", [])[:10]:
        bar_len = int(feat["importance"] * 300)
        bar = "█" * bar_len
        print(f"  {feat['name']:<25} {feat['importance']:.4f}  {CYAN}{bar}{RESET}")

    if backtest_result:
        print(f"\n{BOLD}  BACKTEST RESULTS{RESET}")
        print(f"  {THIN}")
        wr_color = GREEN if backtest_result['win_rate'] >= 55 else YELLOW
        pf_color = GREEN if backtest_result['profit_factor'] >= 1.2 else YELLOW
        pnl_color = GREEN if backtest_result['net_pnl_usd'] >= 0 else "\033[91m"
        print(row("Total trades",    backtest_result['total_trades']))
        print(row("Wins",            backtest_result['wins']))
        print(row("Losses",          backtest_result['losses']))
        print(row("Win rate",        f"{backtest_result['win_rate']}%", wr_color))
        print(row("Profit factor",   backtest_result['profit_factor'], pf_color))
        print(row("Net P&L (0.1 lot)", f"${backtest_result['net_pnl_usd']}", pnl_color))

    print(f"\n  {THIN}")
    print(f"  {GREEN}✔ Model saved to models/rf_model.pkl{RESET}")
    print(f"  {GREEN}✔ Ready for live bot{RESET}\n")


# ─────────────────────────── MAIN ─────────────────────────────────

def main():
    args  = parse_args()
    creds = get_credentials(args)

    # ── 1. Connect to MT5 ──────────────────────────────────────────
    print(f"\n⟳  Connecting to MT5 ({creds['server']})...")
    try:
        from mt5_connector import MT5Connector
    except ImportError:
        logger.error("mt5_connector.py not found. Run this script from the grid_ml_bot folder.")
        sys.exit(1)

    mt5 = MT5Connector(
        login    = int(creds["login"]),
        password = str(creds["password"]),
        server   = str(creds["server"]),
        path     = creds.get("path"),
    )

    if not mt5.connect():
        logger.error("MT5 connection failed. Check your credentials and that MT5 is running.")
        sys.exit(1)

    acc = mt5.get_account_info()
    print(f"✔  Connected: Login={acc['login']} | Balance={acc['currency']} {acc['balance']}")

    # ── 2. Fetch historical data ───────────────────────────────────
    symbol    = args.symbol
    timeframe = args.timeframe
    bars      = args.bars

    print(f"\n⟳  Fetching {bars:,} bars of {symbol} {timeframe}...")
    df = mt5.get_ohlcv(symbol, timeframe, count=bars)

    if df.empty:
        logger.error(f"No data returned for {symbol} {timeframe}. "
                     "Check the symbol name and that market history is available.")
        mt5.disconnect()
        sys.exit(1)

    actual_bars = len(df)
    date_from   = str(df["time"].iloc[0])[:10]
    date_to     = str(df["time"].iloc[-1])[:10]
    print(f"✔  Got {actual_bars:,} bars  ({date_from} → {date_to})")

    # ── 3. Train ───────────────────────────────────────────────────
    try:
        from ml_engine import GridMLEngine, label_data, engineer_features
    except ImportError:
        logger.error("ml_engine.py not found. Run this script from the grid_ml_bot folder.")
        mt5.disconnect()
        sys.exit(1)

    print(f"\n⟳  Engineering features...")
    ml = GridMLEngine()

    # Override labelling params from CLI if provided
    import ml_engine as _ml
    _orig_label = _ml.label_data
    def _patched_label(df, tp_pips=None, sl_pips=None, pip_size=0.0001, horizon=None):
        return _orig_label(
            df,
            tp_pips  = args.tp_pips,
            sl_pips  = args.sl_pips,
            pip_size = pip_size,
            horizon  = args.horizon,
        )
    _ml.label_data = _patched_label

    print(f"⟳  Training model  (tp={args.tp_pips}pip  sl={args.sl_pips}pip  horizon={args.horizon})...")
    t0      = time.time()
    metrics = ml.train(df)
    elapsed = time.time() - t0
    print(f"✔  Training complete in {elapsed:.1f}s")

    # Restore original
    _ml.label_data = _orig_label

    # ── 4. Backtest ────────────────────────────────────────────────
    backtest_result = None
    if args.backtest:
        print(f"\n⟳  Running backtest on {actual_bars:,} bars...")
        # Generate predictions for the whole dataset
        feats = engineer_features(df)
        preds = []
        label_map = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        for col in ml.feature_names:
            if col not in feats.columns:
                feats[col] = 0.0
        feats_aligned = feats[ml.feature_names].dropna()
        X_scaled = ml.scaler.transform(feats_aligned.values)
        raw_preds = ml.model.predict(X_scaled)
        preds = [label_map[p] for p in raw_preds]

        df_bt = df.iloc[feats_aligned.index].reset_index(drop=True)
        backtest_result = simple_backtest(
            df_bt, preds,
            tp_pips  = args.tp_pips,
            sl_pips  = args.sl_pips,
        )
        print(f"✔  Backtest done — {backtest_result['total_trades']} trades")

    # ── 5. Export predictions CSV ──────────────────────────────────
    if args.export_csv:
        import pandas as pd
        print(f"\n⟳  Exporting predictions to CSV...")
        feats = engineer_features(df)
        for col in ml.feature_names:
            if col not in feats.columns:
                feats[col] = 0.0
        feats_aligned = feats[ml.feature_names].dropna()

        import numpy as np
        X_scaled = ml.scaler.transform(feats_aligned.values)
        raw_preds = ml.model.predict(X_scaled)
        proba     = ml.model.predict_proba(X_scaled)
        label_map = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}

        out_df = df.iloc[feats_aligned.index].copy().reset_index(drop=True)
        out_df["signal"]     = [label_map[p] for p in raw_preds]
        out_df["confidence"] = (proba.max(axis=1) * 100).round(1)

        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"predictions_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        out_df.to_csv(csv_path, index=False)
        print(f"✔  Saved to {csv_path}")

    # ── 6. Save MT5 credentials for next time ──────────────────────
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mt5_config.json")
    if not os.path.exists(config_path):
        save = input("\nSave credentials to mt5_config.json for future runs? (y/n): ").strip().lower()
        if save == "y":
            with open(config_path, "w") as f:
                json.dump({
                    "login":    creds["login"],
                    "password": creds["password"],
                    "server":   creds["server"],
                    "path":     creds.get("path"),
                }, f, indent=2)
            print(f"✔  Saved to {config_path}")

    # ── 7. Print report ────────────────────────────────────────────
    print_report(metrics, backtest_result)

    mt5.disconnect()


if __name__ == "__main__":
    main()
