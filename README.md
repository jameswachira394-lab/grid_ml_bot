# GRID·ML Scalper — Real MT5 Trading Bot

A complete ML-powered grid scalping bot for MetaTrader 5.
Trades EUR/USD (or any Forex pair) on the 1-minute timeframe.

---

## SYSTEM REQUIREMENTS

- **Windows 10/11** (MetaTrader5 Python library only works on Windows)
- **Python 3.10+**
- **MetaTrader 5 terminal** installed and logged in
- A live or **demo** MT5 account (any broker)

---

## INSTALLATION

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Make sure MT5 terminal is running
Open MetaTrader 5, log into your account. The Python library connects
to the already-running terminal.

### 3. Start the bot server

```bash
python server.py
```

### 4. Open the dashboard
Navigate to: **http://localhost:5000**

---

## CONNECTING

Fill in the connect form:
| Field | Example |
|---|---|
| MT5 Login | 12345678 |
| Password | your_password |
| Server | ICMarkets-Demo01 |
| MT5 Path | C:/Program Files/MetaTrader 5/terminal64.exe |
| Symbol | EURUSD |
| Timeframe | M1 |

Click **CONNECT & START BOT** — the bot will:
1. Connect to MT5
2. Fetch 5,000 1M candles for initial training
3. Train the Random Forest model
4. Start the live trading loop

---

## HOW IT WORKS

### ML Engine (Random Forest Classifier)
- **47 features** engineered from raw OHLCV:
  RSI, EMA cross, Bollinger Width, ATR, Volume Z-Score,
  MACD, Stoch RSI, CCI, candle body%, price returns, etc.
- **Labels**: forward-looking (BUY / SELL / NEUTRAL) based on
  whether price hits TP before SL in next 10 candles
- **Retrains automatically** every 500 new bars
- Minimum confidence threshold: 55% (configurable)

### Grid Logic
- When ML fires BUY signal → places 3 BUY LIMIT orders below price
- When ML fires SELL signal → places 3 SELL LIMIT orders above price
- Each level has its own TP and SL
- Hard 30-pip grid stop loss protects against trending moves
- Grid rebuilds when all levels are closed

### Risk Manager
- **% risk per trade** position sizing (default 0.4%)
- **Max daily drawdown** guard (default 2.0%) — bot pauses if breached
- **Max trades per day** limit
- All values configurable live without restart

---

## RISK WARNING

This bot places **REAL ORDERS** on your MT5 account.

- Always test on a **DEMO account** first
- Understand the grid system — in a strong trend, all levels can hit SL
- The ML model is not a crystal ball — past accuracy ≠ future returns
- Start with minimum lot size (0.01)
- Monitor the daily drawdown meter

---

## FILE STRUCTURE

```
grid_ml_bot/
├── server.py           ← Flask + SocketIO server (entry point)
├── bot_engine.py       ← Main orchestration loop
├── mt5_connector.py    ← All MT5 API calls
├── ml_engine.py        ← Feature engineering + Random Forest
├── risk_manager.py     ← Position sizing, DD guard, grid state
├── requirements.txt
├── models/             ← Saved model files (auto-created)
│   ├── rf_model.pkl
│   └── scaler.pkl
├── logs/
│   └── bot.log
└── templates/
    └── dashboard.html  ← Live trading dashboard
```

---

## CONFIGURATION (Live)

You can update settings without restarting via the API:

```bash
curl -X POST http://localhost:5000/api/update_config \
  -H "Content-Type: application/json" \
  -d '{"risk_pct": 0.3, "grid_step": 4, "tp_pips": 5}'
```

Or just click **⚙ CONFIG** on the dashboard to reconnect with new settings.

---

## API ENDPOINTS

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/connect | Connect to MT5 + configure |
| POST | /api/start | Start bot loop |
| POST | /api/stop | Pause bot loop |
| POST | /api/emergency_stop | Close all + stop |
| POST | /api/retrain | Force ML retrain |
| POST | /api/update_config | Update risk settings live |
| GET  | /api/state | Full bot state JSON |
| GET  | /api/account | Account info |
| GET  | /api/positions | Open positions |
| GET  | /api/history?days=1 | Trade history |
