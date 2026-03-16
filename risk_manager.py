"""
risk_manager.py
Handles all risk logic:
- Position sizing (% risk per trade)
- Daily drawdown limit enforcement
- Grid level management
- SL/TP calculation in pips
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, date

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    # Symbol
    symbol:          str   = "EURUSD"
    pip_size:        float = 0.0001       # 0.001 for JPY pairs
    digits:          int   = 5

    # Grid settings
    grid_step_pips:  float = 5.0          # distance between grid levels
    max_levels:      int   = 6            # max open grid positions per side
    tp_pips:         float = 40.0          # take-profit per level
    sl_pips:         float = 15.0         # hard stop (grid collapse SL)

    # Money management
    risk_pct:        float = 0.4          # % of balance risked per trade
    min_lot:         float = 0.01
    max_lot:         float = 1.0
    lot_step:        float = 0.01

    # Daily limits
    max_daily_dd_pct: float = 20.0         # stop bot if daily DD exceeds this
    max_trades_day:  int   = 100

    # ML thresholds
    ml_min_confidence: float = 55.0       # minimum ML confidence to trade
    ml_retrain_bars:   int   = 500        # retrain every N new bars


@dataclass
class GridLevel:
    price:    float
    type:     str           # "buy" or "sell"
    ticket:   int   = 0
    volume:   float = 0.01
    tp:       float = 0.0
    sl:       float = 0.0
    profit:   float = 0.0
    active:   bool  = True
    filled:   bool  = False
    open_time: Optional[datetime] = None


class RiskManager:
    def __init__(self, config: RiskConfig = None):
        self.cfg            = config or RiskConfig()
        self.grid_levels:   List[GridLevel] = []
        self.daily_start_balance: float     = 0.0
        self.day_opened:    Optional[date]  = None
        self.trades_today:  int             = 0
        self.wins_today:    int             = 0
        self.losses_today:  int             = 0
        self.daily_pnl:     float           = 0.0

    # ─────────────────────────── SIZING ───────────────────────────────

    def calculate_lot_size(self, balance: float, sl_pips: float = None) -> float:
        """
        Risk-based position sizing:
        lot = (balance * risk_pct%) / (sl_pips * pip_value_per_lot)
        Assumes standard lot = 100,000 units, pip value ≈ $10 per standard lot for USD pairs.
        """
        if sl_pips is None:
            sl_pips = self.cfg.sl_pips

        risk_amount    = balance * (self.cfg.risk_pct / 100)
        pip_value_lot  = 10.0   # USD per pip per standard lot (EURUSD approx)
        lot_raw        = risk_amount / (sl_pips * pip_value_lot)
        lot_stepped    = round(round(lot_raw / self.cfg.lot_step) * self.cfg.lot_step, 2)
        lot_final      = max(self.cfg.min_lot, min(self.cfg.max_lot, lot_stepped))
        return lot_final

    # ─────────────────────────── DRAWDOWN ─────────────────────────────

    def update_daily_state(self, current_balance: float):
        today = date.today()
        if self.day_opened != today:
            self.daily_start_balance = current_balance
            self.day_opened          = today
            self.trades_today        = 0
            self.wins_today          = 0
            self.losses_today        = 0
            self.daily_pnl           = 0.0
            logger.info(f"New trading day — start balance: {current_balance:.2f}")

        self.daily_pnl = current_balance - self.daily_start_balance

    def is_daily_dd_breached(self, current_equity: float) -> bool:
        if self.daily_start_balance == 0:
            return False
        dd_pct = (self.daily_start_balance - current_equity) / self.daily_start_balance * 100
        if dd_pct >= self.cfg.max_daily_dd_pct:
            logger.warning(f"Daily DD breached: {dd_pct:.2f}% >= {self.cfg.max_daily_dd_pct}%")
            return True
        return False

    def is_trade_limit_reached(self) -> bool:
        return self.trades_today >= self.cfg.max_trades_day

    def get_daily_dd_pct(self, current_equity: float) -> float:
        if self.daily_start_balance == 0:
            return 0.0
        return max(0, (self.daily_start_balance - current_equity) / self.daily_start_balance * 100)

    # ─────────────────────────── GRID ─────────────────────────────────

    def build_grid(self, current_price: float, signal: str, balance: float) -> List[GridLevel]:
        """
        Build grid levels around current price based on ML signal.
        signal = "BUY"  → place buy levels below price
        signal = "SELL" → place sell levels above price
        signal = "NEUTRAL" → place both sides (market-neutral grid)
        """
        step = self.cfg.grid_step_pips * self.cfg.pip_size
        tp   = self.cfg.tp_pips        * self.cfg.pip_size
        sl   = self.cfg.sl_pips        * self.cfg.pip_size
        lot  = self.calculate_lot_size(balance)
        levels: List[GridLevel] = []

        if signal in ("BUY", "NEUTRAL"):
            for i in range(1, self.cfg.max_levels // 2 + 1):
                lp = round(current_price - i * step, 5)
                levels.append(GridLevel(
                    price    = lp,
                    type     = "buy",
                    volume   = lot,
                    tp       = round(lp + tp, 5),
                    sl       = round(lp - sl, 5),
                    active   = True,
                ))

        if signal in ("SELL", "NEUTRAL"):
            for i in range(1, self.cfg.max_levels // 2 + 1):
                lp = round(current_price + i * step, 5)
                levels.append(GridLevel(
                    price    = lp,
                    type     = "sell",
                    volume   = lot,
                    tp       = round(lp - tp, 5),
                    sl       = round(lp + sl, 5),
                    active   = True,
                ))

        self.grid_levels = levels
        logger.info(f"Grid built: {len(levels)} levels | signal={signal} | lot={lot}")
        return levels

    def get_active_levels(self) -> List[GridLevel]:
        return [g for g in self.grid_levels if g.active and not g.filled]

    def mark_level_filled(self, price: float, order_type: str, ticket: int):
        for g in self.grid_levels:
            if g.type == order_type and abs(g.price - price) < self.cfg.pip_size * 3:
                g.filled   = True
                g.ticket   = ticket
                g.open_time = datetime.utcnow()
                logger.info(f"Level filled: {order_type.upper()} @ {price} ticket={ticket}")
                break

    def mark_level_closed(self, ticket: int, profit: float):
        for g in self.grid_levels:
            if g.ticket == ticket:
                g.active = False
                g.profit = profit
                if profit > 0:
                    self.wins_today += 1
                else:
                    self.losses_today += 1
                self.trades_today += 1
                logger.info(f"Level closed: ticket={ticket} profit={profit:.2f}")
                break

    def clear_grid(self):
        self.grid_levels = []
        logger.info("Grid cleared")

    # ─────────────────────────── SUMMARY ──────────────────────────────

    def get_summary(self, current_equity: float) -> dict:
        active  = [g for g in self.grid_levels if g.active]
        filled  = [g for g in self.grid_levels if g.filled and g.active]
        wr      = self.wins_today/(self.trades_today or 1) * 100

        return {
            "grid_step_pips":    self.cfg.grid_step_pips,
            "max_levels":        self.cfg.max_levels,
            "tp_pips":           self.cfg.tp_pips,
            "sl_pips":           self.cfg.sl_pips,
            "risk_pct":          self.cfg.risk_pct,
            "max_daily_dd_pct":  self.cfg.max_daily_dd_pct,
            "ml_min_confidence": self.cfg.ml_min_confidence,
            "active_levels":     len(active),
            "filled_levels":     len(filled),
            "trades_today":      self.trades_today,
            "wins_today":        self.wins_today,
            "losses_today":      self.losses_today,
            "daily_pnl":         round(self.daily_pnl, 2),
            "daily_dd_pct":      round(self.get_daily_dd_pct(current_equity), 2),
            "win_rate":          round(wr, 1),
            "levels": [
                {
                    "price":   g.price,
                    "type":    g.type,
                    "tp":      g.tp,
                    "sl":      g.sl,
                    "volume":  g.volume,
                    "ticket":  g.ticket,
                    "profit":  g.profit,
                    "filled":  g.filled,
                    "active":  g.active,
                }
                for g in self.grid_levels
            ]
        }
