"""
server.py
Flask + Flask-SocketIO backend for GRID·ML Scalper dashboard.
  - Serves the dashboard HTML from templates/dashboard.html
  - REST endpoints: /api/connect, start, stop, emergency_stop,
                    retrain, state, account, positions, history,
                    update_config
  - WebSocket push of live bot state on every tick
  - Auto-increments port if the default (5000) is already in use

Project layout expected:
  project/
  ├── server.py              ← this file
  ├── mt5_connector.py
  ├── ml_engine.py
  ├── risk_manager.py
  ├── bot_engine.py
  ├── templates/
  │   └── dashboard.html     ← rename gridml-dashboard.html → this
  ├── static/                ← (optional) CSS / JS assets
  └── logs/                  ← created automatically on startup
"""

import logging
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from mt5_connector import MT5Connector
from ml_engine     import GridMLEngine
from risk_manager  import RiskManager, RiskConfig
from bot_engine    import BotEngine

# ─────────────────────────── LOGGING ──────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)          # create logs/ if missing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log")),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────── FLASK SETUP ──────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",   # templates/dashboard.html
    static_folder="static",        # static/  (CSS, JS, images)
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gridml-secret-2024")
CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,           # set True for verbose WS debug output
    engineio_logger=False,
)

# ─────────────────────────── GLOBALS ──────────────────────────────────
mt5_conn: MT5Connector | None = None
ml_eng:   GridMLEngine        = GridMLEngine()
risk_mgr: RiskManager         = RiskManager()
bot:      BotEngine | None    = None


def push_state(state: dict) -> None:
    """Called by BotEngine on every tick → broadcast to all dashboards."""
    socketio.emit("state", state)


# ─────────────────────────── HELPERS ──────────────────────────────────

def _bot_required():
    """Return a 400 JSON response if bot is not initialised, else None."""
    if bot is None:
        return jsonify({"success": False, "error": "Not connected — call /api/connect first"}), 400
    return None


def _mt5_required():
    if mt5_conn is None:
        return jsonify({"success": False, "error": "MT5 not connected"}), 400
    return None


# ─────────────────────────── SERVE DASHBOARD ──────────────────────────

@app.route("/")
def index():
    """Render the main dashboard (templates/dashboard.html)."""
    return render_template("dashboard.html")


# Optional: serve arbitrary files from static/ (logo, icons, etc.)
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ─────────────────────────── REST: CONNECTION ─────────────────────────

@app.route("/api/connect", methods=["POST"])
def api_connect():
    """
    Connect to MT5 and initialise the bot engine.

    Expected JSON body:
      login, password, server          (required)
      path, symbol, timeframe          (optional)
      grid_step, max_levels, tp_pips,
      sl_pips, risk_pct, max_dd,
      min_conf, tick_interval          (optional, all have defaults)
    """
    global mt5_conn, bot

    data = request.get_json(force=True) or {}

    # Validate required fields
    missing = [k for k in ("login", "password", "server") if not data.get(k)]
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    # --- MT5 connection ---
    mt5_conn = MT5Connector(
        login    = int(data["login"]),
        password = data["password"],
        server   = data["server"],
        path     = data.get("path") or None,   # None → use default MT5 install
    )

    if not mt5_conn.connect():
        return jsonify({"success": False, "error": "MT5 connection failed — check credentials and server name."}), 400

    # --- Risk / grid config ---
    cfg = RiskConfig(
        symbol            = data.get("symbol",       "EURUSD"),
        grid_step_pips    = float(data.get("grid_step",   5.0)),
        max_levels        = int(data.get("max_levels",    6)),
        tp_pips           = float(data.get("tp_pips",     4.0)),
        sl_pips           = float(data.get("sl_pips",     30.0)),
        risk_pct          = float(data.get("risk_pct",    0.4)),
        max_daily_dd_pct  = float(data.get("max_dd",      2.0)),
        ml_min_confidence = float(data.get("min_conf",    55.0)),
    )
    risk_mgr.__init__(cfg)

    # --- ML model (load saved weights if available) ---
    ml_eng.load()

    # --- Build BotEngine ---
    bot = BotEngine(
        mt5           = mt5_conn,
        ml            = ml_eng,
        risk          = risk_mgr,
        symbol        = cfg.symbol,
        timeframe     = data.get("timeframe", "M1"),
        on_update     = push_state,
        tick_interval = float(data.get("tick_interval", 1.0)),
    )

    acc = mt5_conn.get_account_info()
    logger.info(f"Connected: {cfg.symbol} on {data['server']}  account={data['login']}")
    return jsonify({"success": True, "account": acc})


# ─────────────────────────── REST: BOT CONTROL ────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    """Start (or resume) the bot loop."""
    err = _bot_required()
    if err:
        return err
    bot.start()
    logger.info("Bot started")
    return jsonify({"success": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Pause the bot loop without closing positions."""
    err = _bot_required()
    if err:
        return err
    bot.stop()
    logger.info("Bot paused")
    return jsonify({"success": True})


@app.route("/api/emergency_stop", methods=["POST"])
def api_emergency_stop():
    """Close ALL open positions immediately and halt the bot."""
    err = _bot_required()
    if err:
        return err
    bot.emergency_stop()
    logger.warning("EMERGENCY STOP triggered via API")
    return jsonify({"success": True})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Kick off a background ML retrain without pausing live trading."""
    err = _bot_required()
    if err:
        return err
    bot.retrain_now()
    logger.info("Retrain requested")
    return jsonify({"success": True, "message": "Retraining started in background"})


# ─────────────────────────── REST: DATA ───────────────────────────────

@app.route("/api/state", methods=["GET"])
def api_state():
    """Snapshot of the full bot state (same dict pushed over WS)."""
    if bot is None:
        return jsonify({"connected": False})
    return jsonify(bot._safe_state())


@app.route("/api/account", methods=["GET"])
def api_account():
    """Current MT5 account info."""
    err = _mt5_required()
    if err:
        return err
    return jsonify(mt5_conn.get_account_info())


@app.route("/api/positions", methods=["GET"])
def api_positions():
    """All currently open MT5 positions for the active symbol."""
    err = _mt5_required()
    if err:
        return err
    symbol = bot.symbol if bot else None
    return jsonify(mt5_conn.get_open_positions(symbol))


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Closed trade history.
    Query param:  ?days=N  (default 1)
    """
    err = _mt5_required()
    if err:
        return err
    days = int(request.args.get("days", 1))
    sym  = bot.symbol if bot else "EURUSD"
    return jsonify(mt5_conn.get_trade_history(sym, days=days))


@app.route("/api/update_config", methods=["POST"])
def api_update_config():
    """
    Hot-update risk/grid parameters without restarting the bot.

    Accepts any subset of:
      grid_step, tp_pips, sl_pips, risk_pct,
      max_dd, min_conf, max_levels
    """
    if risk_mgr is None:
        return jsonify({"success": False, "error": "Risk manager not initialised"}), 400

    data = request.get_json(force=True) or {}
    cfg  = risk_mgr.cfg

    mapping = {
        "grid_step":  ("grid_step_pips",   float),
        "tp_pips":    ("tp_pips",          float),
        "sl_pips":    ("sl_pips",          float),
        "risk_pct":   ("risk_pct",         float),
        "max_dd":     ("max_daily_dd_pct", float),
        "min_conf":   ("ml_min_confidence",float),
        "max_levels": ("max_levels",       int),
    }

    updated = {}
    for key, (attr, cast) in mapping.items():
        if key in data:
            setattr(cfg, attr, cast(data[key]))
            updated[attr] = getattr(cfg, attr)

    logger.info(f"Config updated live: {updated}")
    return jsonify({"success": True, "updated": updated})


# ─────────────────────────── SOCKET EVENTS ────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info(f"Dashboard connected (sid={request.sid})")
    # Push current state immediately so the UI doesn't wait for the next tick
    if bot is not None:
        emit("state", bot._safe_state())


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Dashboard disconnected (sid={request.sid})")


@socketio.on("ping")
def on_ping(data):
    """Latency check — dashboard sends {ts: Date.now()} and expects it back."""
    emit("pong", {"ts": data.get("ts")})


@socketio.on("request_state")
def on_request_state(data):
    """Explicit state pull from the dashboard (e.g. after page focus)."""
    if bot is not None:
        emit("state", bot._safe_state())


# ─────────────────────────── ENTRY POINT ──────────────────────────────

if __name__ == "__main__":
    import socket as _socket

    default_port = int(os.environ.get("PORT", 5000))
    port = default_port

    # Auto-increment until a free port is found
    while True:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                break   # port is free
            logger.warning(f"Port {port} in use, trying {port + 1}")
            port += 1

    logger.info(f"Starting GRID·ML Scalper → http://localhost:{port}")
    socketio.run(
        app,
        host  = "0.0.0.0",
        port  = port,
        debug = os.environ.get("DEBUG", "false").lower() == "true",
        use_reloader = False,   # reloader breaks the bot background thread
    )