"""
bot_engine.py
Main orchestrator:
  - Connects to MT5
  - Runs ML prediction every tick
  - Builds and manages grid levels
  - Enforces risk rules
  - Emits live state to SocketIO dashboard
"""

import time
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from mt5_connector import MT5Connector
from ml_engine     import GridMLEngine
from risk_manager  import RiskManager, RiskConfig

logger = logging.getLogger(__name__)


class BotEngine:
    def __init__(
        self,
        mt5:         MT5Connector,
        ml:          GridMLEngine,
        risk:        RiskManager,
        symbol:      str     = "XAUUSD",
        timeframe:   str     = "M1",
        on_update:   Callable = None,   # callback(state_dict) → push to dashboard
        tick_interval: float  = 1.0,    # seconds between ticks
    ):
        self.mt5           = mt5
        self.ml            = ml
        self.risk          = risk
        self.symbol        = symbol
        self.timeframe     = timeframe
        self.on_update     = on_update
        self.tick_interval = tick_interval

        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._lock         = threading.Lock()

        # State
        self.state = {
            "running":      False,
            "symbol":       symbol,
            "timeframe":    timeframe,
            "tick":         {},
            "account":      {},
            "signal":       {"signal":"NEUTRAL","confidence":0,"probabilities":{"BUY":0,"NEUTRAL":100,"SELL":0}},
            "risk":         {},
            "candles":      [],
            "log":          [],
            "last_retrain": None,
            "bars_since_retrain": 0,
            "error":        None,
        }
        self._bars_loaded   = 0
        self._retrain_every = risk.cfg.ml_retrain_bars

    # ─────────────────────────── CONTROL ──────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self.state["running"] = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log("INFO", "Bot started")
        logger.info("Bot engine started")

    def stop(self):
        self._running = False
        self.state["running"] = False
        self._log("WARN", "Bot stopped")
        logger.info("Bot engine stopped")

    def emergency_stop(self):
        self._running = False
        self.state["running"] = False
        try:
            closed = self.mt5.close_all_positions(self.symbol)
            self._log("WARN", f"EMERGENCY STOP — {len(closed)} positions closed")
        except Exception as e:
            self._log("ERROR", f"Emergency stop error: {e}")
        self.risk.clear_grid()
        self.state["risk"] = self.risk.get_summary(0)

    def retrain_now(self):
        """Force an immediate model retrain."""
        threading.Thread(target=self._retrain, daemon=True).start()

    # ─────────────────────────── MAIN LOOP ────────────────────────────

    def _loop(self):
        # Initial training
        self._retrain()

        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.exception(f"Tick error: {e}")
                self.state["error"] = str(e)
                self._log("ERROR", str(e))
                # Try reconnect
                if "MT5" in str(e) or "terminal" in str(e).lower():
                    self._log("WARN", "Attempting MT5 reconnect...")
                    self.mt5.reconnect()
            time.sleep(self.tick_interval)

    def _tick(self):
        # 1. Get live tick
        tick = self.mt5.get_tick(self.symbol)
        if not tick:
            return
        self.state["tick"] = tick
        price = tick["mid"]

        # 2. Account info
        account = self.mt5.get_account_info()
        self.state["account"] = account
        equity  = account.get("equity", 0)
        balance = account.get("balance", 0)

        # 3. Daily state / drawdown check
        self.risk.update_daily_state(balance)
        if self.risk.is_daily_dd_breached(equity):
            self._log("WARN", "Daily drawdown limit hit — pausing")
            self.stop()
            return

        # 4. Get OHLCV + ML prediction
        df = self.mt5.get_ohlcv(self.symbol, self.timeframe, count=300)
        if df.empty:
            return

        new_bars = len(df) - self._bars_loaded
        self._bars_loaded = len(df)
        self.state["bars_since_retrain"] += max(0, new_bars)

        # 5. Retrain check
        if self.state["bars_since_retrain"] >= self._retrain_every:
            self.state["bars_since_retrain"] = 0
            threading.Thread(target=self._retrain, daemon=True).start()

        # 6. ML prediction
        signal_data = self.ml.predict(df)
        self.state["signal"] = signal_data
        signal     = signal_data["signal"]
        confidence = signal_data["confidence"]

        # 7. Candle snapshot for chart (last 80)
        self.state["candles"] = df.tail(80)[["time","open","high","low","close","volume"]].copy()
        self.state["candles"]["time"] = self.state["candles"]["time"].astype(str)
        self.state["candles"] = self.state["candles"].to_dict(orient="records")

        # 8. Update open position profits
        positions = self.mt5.get_open_positions(self.symbol)
        for pos in positions:
            self.risk.mark_level_closed  # keep profits live in grid state

        # 9. Grid logic
        self._manage_grid(price, signal, confidence, balance, equity)

        # 10. Risk summary
        self.state["risk"] = self.risk.get_summary(equity)

        # 11. Push to dashboard
        if self.on_update:
            self.on_update(self._safe_state())

    # ─────────────────────────── GRID LOGIC ───────────────────────────

    def _manage_grid(self, price: float, signal: str, confidence: float,
                     balance: float, equity: float):
        """
        Core grid management:
        1. If no grid → build grid when ML confidence >= threshold
        2. Check if any pending levels should be activated (price reached level)
        3. Place new limit orders for unfilled levels
        4. Check filled levels for TP/SL hits
        """
        cfg = self.risk.cfg
        threshold = cfg.ml_min_confidence

        # Don't trade if confidence too low
        if confidence < threshold and not self.risk.get_active_levels():
            return

        # If no grid exists and we have a valid signal, build one
        if not self.risk.grid_levels and confidence >= threshold:
            levels = self.risk.build_grid(price, signal, balance)
            self._log("INFO", f"Grid built: {signal} | {len(levels)} levels | confidence={confidence}%")
            # Place limit orders for all levels
            for level in levels:
                self._place_grid_order(level)
            return

        # Check existing positions for TP/SL status
        open_positions = self.mt5.get_open_positions(self.symbol)
        open_tickets   = {p["ticket"] for p in open_positions}

        for g in self.risk.grid_levels:
            if not g.active:
                continue
            if g.filled and g.ticket not in open_tickets:
                # Position was closed (TP or SL hit)
                history = self.mt5.get_trade_history(self.symbol, days=1)
                profit  = next((h["profit"] for h in history if h.get("ticket") == g.ticket or
                                abs(h["price"]-g.tp) < cfg.pip_size*2), 0.0)
                self.risk.mark_level_closed(g.ticket, profit)
                verb = "TP HIT" if profit > 0 else "SL HIT"
                self._log("TRADE", f"{verb} {g.type.upper()} @ {g.price:.5f} profit={profit:.2f}")

        # If all grid levels are closed → rebuild if signal still valid
        active = self.risk.get_active_levels()
        if not active and confidence >= threshold:
            self.risk.clear_grid()
            levels = self.risk.build_grid(price, signal, balance)
            self._log("INFO", f"Grid rebuilt: {signal} | {len(levels)} levels")
            for level in levels:
                self._place_grid_order(level)

    def _place_grid_order(self, level):
        """Place a limit order for a grid level."""
        if level.filled:
            return
        order_type = f"{level.type}_limit"
        result = self.mt5.place_order(
            symbol     = self.symbol,
            order_type = order_type,
            volume     = level.volume,
            price      = level.price,
            sl         = level.sl,
            tp         = level.tp,
            comment    = f"GridML_{level.type.upper()}",
        )
        if result.get("success"):
            self.risk.mark_level_filled(level.price, level.type, result["ticket"])
            self._log("TRADE", f"LIMIT {level.type.upper()} placed @ {level.price:.5f} "
                               f"lot={level.volume} TP={level.tp:.5f} SL={level.sl:.5f}")
        else:
            self._log("ERROR", f"Order failed: {result.get('error','unknown')}")

    # ─────────────────────────── RETRAIN ──────────────────────────────

    def _retrain(self):
        try:
            self._log("INFO", "Retraining ML model...")
            # Fetch more bars for training
            df = self.mt5.get_ohlcv(self.symbol, self.timeframe, count=5000)
            if df.empty or len(df) < 200:
                self._log("WARN", "Not enough bars to train")
                return
            metrics = self.ml.train(df)
            self.state["last_retrain"]   = datetime.now(timezone.utc).isoformat()
            self.state["ml_metrics"]     = metrics
            self._log("INFO",
                f"Retrain done — accuracy={metrics['accuracy']}% | "
                f"n_features={metrics['n_features']}")
        except Exception as e:
            logger.exception(f"Retrain error: {e}")
            self._log("ERROR", f"Retrain failed: {e}")

    # ─────────────────────────── LOG ──────────────────────────────────

    def _log(self, level: str, msg: str):
        entry = {
            "time":  datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level": level,
            "msg":   msg,
        }
        self.state["log"].insert(0, entry)
        self.state["log"] = self.state["log"][:100]
        logger.info(f"[{level}] {msg}")

    # ─────────────────────────── SAFE STATE ───────────────────────────

    def _safe_state(self) -> dict:
        """Return JSON-serialisable state snapshot."""
        import copy
        s = copy.deepcopy(self.state)
        return s
