"""
server.py
Flask + Flask-SocketIO backend.
  - REST endpoints for config, start/stop, retrain
  - WebSocket push of live bot state to dashboard
"""

import logging
import os
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from mt5_connector import MT5Connector
from ml_engine     import GridMLEngine
from risk_manager  import RiskManager, RiskConfig
from bot_engine    import BotEngine

# ─────────────────────────── LOGGING ──────────────────────────────────
# FIX 1: Create logs directory before FileHandler tries to open it
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_log_dir, "bot.log")),
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────── FLASK SETUP ──────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "gridml-secret-2024"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ─────────────────────────── GLOBALS ──────────────────────────────────
mt5_conn: MT5Connector = None
ml_eng:   GridMLEngine  = GridMLEngine()
risk_mgr: RiskManager   = RiskManager()
bot:      BotEngine     = None


def push_state(state: dict):
    """Called by BotEngine every tick → push to all connected dashboards."""
    socketio.emit("state", state)


# ─────────────────────────── FIX 2: JSON error handlers ───────────────
# Ensures Flask never returns an HTML error page to API calls

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": f"Endpoint not found: {request.path}"}), 404
    return render_template("dashboard.html"), 404   # fallback for SPA

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({"success": False, "error": "Internal server error", "detail": str(e)}), 500


# ─────────────────────────── REST API ─────────────────────────────────

@app.route("/")
def index():
    # FIX 3: Graceful fallback if template is missing
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "dashboard.html")
    if not os.path.exists(template_path):
        return (
            "<h2>Grid ML Scalper — Server Running</h2>"
            "<p>Dashboard template not found. Place <code>dashboard.html</code> in the <code>templates/</code> folder.</p>"
            "<p>API is available at <code>/api/state</code></p>"
        ), 200
    return render_template("dashboard.html")


@app.route("/api/connect", methods=["POST"])
def api_connect():
    """Connect to MT5 and initialise bot."""
    global mt5_conn, bot
    data = request.get_json(silent=True)  # FIX 4: silent=True avoids 400 on bad JSON body

    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    required = ["login", "password", "server"]
    if not all(k in data for k in required):
        return jsonify({"success": False, "error": f"Missing required fields: {required}"}), 400

    try:
        mt5_conn = MT5Connector(
            login    = int(data["login"]),
            password = data["password"],
            server   = data["server"],
            path     = data.get("path"),
        )

        if not mt5_conn.connect():
            return jsonify({"success": False, "error": "MT5 connection failed. Check credentials and terminal."}), 400

        cfg = RiskConfig(
            symbol            = data.get("symbol",         "EURUSD"),
            grid_step_pips    = float(data.get("grid_step",  5.0)),
            max_levels        = int(data.get("max_levels",   6)),
            tp_pips           = float(data.get("tp_pips",    4.0)),
            sl_pips           = float(data.get("sl_pips",    30.0)),
            risk_pct          = float(data.get("risk_pct",   0.4)),
            max_daily_dd_pct  = float(data.get("max_dd",     2.0)),
            ml_min_confidence = float(data.get("min_conf",   55.0)),
        )
        risk_mgr.__init__(cfg)

        ml_eng.load()

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
        return jsonify({"success": True, "account": acc})

    except Exception as e:
        logger.exception("Error in /api/connect")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def api_start():
    if bot is None:
        return jsonify({"success": False, "error": "Not connected. Call /api/connect first."}), 400
    bot.start()
    return jsonify({"success": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    if bot is None:
        return jsonify({"success": False, "error": "Not connected"}), 400
    bot.stop()
    return jsonify({"success": True})


@app.route("/api/emergency_stop", methods=["POST"])
def api_emergency_stop():
    if bot is None:
        return jsonify({"success": False, "error": "Not connected"}), 400
    bot.emergency_stop()
    return jsonify({"success": True})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    if bot is None:
        return jsonify({"success": False, "error": "Not connected"}), 400
    bot.retrain_now()
    return jsonify({"success": True, "message": "Retraining started in background"})


@app.route("/api/state", methods=["GET"])
def api_state():
    if bot is None:
        return jsonify({"connected": False})
    return jsonify(bot._safe_state())


@app.route("/api/account", methods=["GET"])
def api_account():
    if mt5_conn is None:
        return jsonify({"success": False, "error": "Not connected"}), 400
    return jsonify(mt5_conn.get_account_info())


@app.route("/api/positions", methods=["GET"])
def api_positions():
    if mt5_conn is None:
        return jsonify([]), 400
    return jsonify(mt5_conn.get_open_positions(bot.symbol if bot else None))


@app.route("/api/history", methods=["GET"])
def api_history():
    if mt5_conn is None:
        return jsonify([]), 400
    days = int(request.args.get("days", 1))
    sym  = bot.symbol if bot else "EURUSD"
    return jsonify(mt5_conn.get_trade_history(sym, days=days))


@app.route("/api/update_config", methods=["POST"])
def api_update_config():
    """Update risk config on the fly (no restart needed)."""
    if risk_mgr is None:
        return jsonify({"success": False, "error": "Not connected"}), 400
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400
    cfg = risk_mgr.cfg
    if "grid_step"  in data: cfg.grid_step_pips    = float(data["grid_step"])
    if "tp_pips"    in data: cfg.tp_pips            = float(data["tp_pips"])
    if "sl_pips"    in data: cfg.sl_pips            = float(data["sl_pips"])
    if "risk_pct"   in data: cfg.risk_pct           = float(data["risk_pct"])
    if "max_dd"     in data: cfg.max_daily_dd_pct   = float(data["max_dd"])
    if "min_conf"   in data: cfg.ml_min_confidence  = float(data["min_conf"])
    if "max_levels" in data: cfg.max_levels         = int(data["max_levels"])
    return jsonify({"success": True})


# ─────────────────────────── SOCKET EVENTS ────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("Dashboard client connected")
    if bot:
        emit("state", bot._safe_state())


@socketio.on("disconnect")
def on_disconnect():
    logger.info("Dashboard client disconnected")


@socketio.on("ping")
def on_ping(data):
    emit("pong", {"ts": data.get("ts")})


# ─────────────────────────── ENTRY POINT ──────────────────────────────

if __name__ == "__main__":
    import eventlet
    import eventlet.wsgi
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Grid ML Scalper server starting on http://localhost:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)