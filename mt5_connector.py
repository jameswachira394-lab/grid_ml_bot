"""
mt5_connector.py
Real MetaTrader 5 connection layer.
Handles: login, live tick feed, OHLCV history, order placement, position management.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import logging

logger = logging.getLogger(__name__)


class MT5Connector:
    def __init__(self, login: int, password: str, server: str, path: str = None):
        self.login    = login
        self.password = password
        self.server   = server
        self.path     = path          # e.g. "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.connected = False

    # ─────────────────────────── CONNECTION ───────────────────────────

    def connect(self) -> bool:
        kwargs = dict(login=self.login, password=self.password, server=self.server)
        if self.path:
            kwargs["path"] = self.path

        if not mt5.initialize(**kwargs):
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        info = mt5.account_info()
        if info is None:
            logger.error(f"Account info failed: {mt5.last_error()}")
            return False

        self.connected = True
        logger.info(f"Connected: {info.login} | {info.server} | Balance: {info.balance}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def reconnect(self) -> bool:
        self.disconnect()
        time.sleep(2)
        return self.connect()

    # ─────────────────────────── ACCOUNT ──────────────────────────────

    def get_account_info(self) -> dict:
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "login":    info.login,
            "balance":  round(info.balance, 2),
            "equity":   round(info.equity, 2),
            "margin":   round(info.margin, 2),
            "free_margin": round(info.margin_free, 2),
            "profit":   round(info.profit, 2),
            "leverage": info.leverage,
            "currency": info.currency,
        }

    # ─────────────────────────── PRICE DATA ───────────────────────────

    def get_tick(self, symbol: str) -> dict:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        return {
            "symbol": symbol,
            "bid":    round(tick.bid, 5),
            "ask":    round(tick.ask, 5),
            "mid":    round((tick.bid + tick.ask) / 2, 5),
            "spread": round(tick.ask - tick.bid, 5),
            "time":   datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
        }

    def get_ohlcv(self, symbol: str, timeframe_str: str = "M1", count: int = 500) -> pd.DataFrame:
        """Returns OHLCV DataFrame sorted oldest→newest."""
        tf_map = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No OHLCV data for {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time","open","high","low","close","volume"]].copy()
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def get_symbol_info(self, symbol: str) -> dict:
        info = mt5.symbol_info(symbol)
        if info is None:
            return {}
        return {
            "symbol":     symbol,
            "digits":     info.digits,
            "point":      info.point,
            "spread":     info.spread,
            "trade_tick_size": info.trade_tick_size,
            "volume_min": info.volume_min,
            "volume_step":info.volume_step,
            "volume_max": info.volume_max,
        }

    # ─────────────────────────── ORDERS ───────────────────────────────

    def place_order(
        self,
        symbol:     str,
        order_type: str,           # "buy" or "sell"
        volume:     float,
        price:      float = 0.0,   # 0 = market
        sl:         float = 0.0,
        tp:         float = 0.0,
        comment:    str   = "GridML",
        magic:      int   = 20240101,
    ) -> dict:
        """
        Places a market or limit order.
        order_type: "buy" | "sell" | "buy_limit" | "sell_limit"
        """
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return {"success": False, "error": f"Symbol not found: {symbol}"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": "Cannot get tick"}

        type_map = {
            "buy":        mt5.ORDER_TYPE_BUY,
            "sell":       mt5.ORDER_TYPE_SELL,
            "buy_limit":  mt5.ORDER_TYPE_BUY_LIMIT,
            "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
            "buy_stop":   mt5.ORDER_TYPE_BUY_STOP,
            "sell_stop":  mt5.ORDER_TYPE_SELL_STOP,
        }
        mt5_type = type_map.get(order_type.lower(), mt5.ORDER_TYPE_BUY)

        # Use market price if not specified
        if price == 0.0:
            price = tick.ask if "buy" in order_type else tick.bid

        request = {
            "action":    mt5.TRADE_ACTION_DEAL if order_type in ("buy","sell")
                         else mt5.TRADE_ACTION_PENDING,
            "symbol":    symbol,
            "volume":    round(volume, 2),
            "type":      mt5_type,
            "price":     round(price, sym_info.digits),
            "sl":        round(sl, sym_info.digits) if sl else 0.0,
            "tp":        round(tp, sym_info.digits) if tp else 0.0,
            "deviation": 20,
            "magic":     magic,
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "error": str(mt5.last_error())}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "retcode": result.retcode,
                "error":   result.comment,
            }

        return {
            "success":  True,
            "ticket":   result.order,
            "volume":   result.volume,
            "price":    result.price,
            "comment":  result.comment,
        }

    def close_position(self, ticket: int) -> dict:
        """Close an open position by ticket."""
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return {"success": False, "error": "Position not found"}
        p = pos[0]
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    p.symbol,
            "volume":    p.volume,
            "type":      close_type,
            "position":  ticket,
            "price":     price,
            "deviation": 20,
            "magic":     p.magic,
            "comment":   "GridML_Close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": result.comment if result else str(mt5.last_error())}
        return {"success": True, "ticket": ticket, "price": price}

    def close_all_positions(self, symbol: str = None, magic: int = 20240101) -> list:
        """Close all open positions (optionally filtered by symbol and magic)."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol
        positions = mt5.positions_get(**kwargs)
        if positions is None:
            return []
        results = []
        for p in positions:
            if p.magic == magic:
                r = self.close_position(p.ticket)
                results.append(r)
        return results

    def get_open_positions(self, symbol: str = None, magic: int = 20240101) -> list:
        """Return list of open positions."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol
        positions = mt5.positions_get(**kwargs)
        if positions is None:
            return []
        out = []
        for p in positions:
            if p.magic != magic:
                continue
            out.append({
                "ticket":    p.ticket,
                "symbol":    p.symbol,
                "type":      "buy" if p.type == 0 else "sell",
                "volume":    p.volume,
                "open_price":round(p.price_open, 5),
                "sl":        round(p.sl, 5),
                "tp":        round(p.tp, 5),
                "profit":    round(p.profit, 2),
                "comment":   p.comment,
                "time":      datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
            })
        return out

    def get_trade_history(self, symbol: str, days: int = 1) -> list:
        """Return closed deals for the past N days."""
        from_date = datetime.now(tz=timezone.utc) - pd.Timedelta(days=days)
        to_date   = datetime.now(tz=timezone.utc)
        deals = mt5.history_deals_get(from_date, to_date, group=f"*{symbol}*")
        if deals is None:
            return []
        out = []
        for d in deals:
            if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                out.append({
                    "ticket": d.ticket,
                    "symbol": d.symbol,
                    "type":   "buy" if d.type == mt5.DEAL_TYPE_BUY else "sell",
                    "volume": d.volume,
                    "price":  round(d.price, 5),
                    "profit": round(d.profit, 2),
                    "time":   datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                    "comment":d.comment,
                })
        return out
