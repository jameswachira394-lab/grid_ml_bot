"""
ml_engine.py
Feature engineering + Random Forest classifier.
Trains on historical OHLCV from MT5 and predicts: BUY / NEUTRAL / SELL
for the next 1M candle — tuned for grid scalping.
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")


# ─────────────────────────── FEATURE ENGINEERING ──────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 47 features from raw OHLCV.
    Input: DataFrame with columns [time, open, high, low, close, volume]
    Output: DataFrame with feature columns (NaN rows dropped)
    """
    d = df.copy()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    o = d["open"]
    v = d["volume"]

    # ── Price action ──
    d["body_pct"]     = (c - o).abs() / (h - l + 1e-9)
    d["upper_wick"]   = (h - c.clip(lower=o)) / (h - l + 1e-9)
    d["lower_wick"]   = (c.clip(upper=o) - l) / (h - l + 1e-9)
    d["candle_range"] = h - l
    d["close_pos"]    = (c - l) / (h - l + 1e-9)   # 0=bottom 1=top of candle

    # ── Returns ──
    for n in [1, 2, 3, 5, 10]:
        d[f"ret_{n}"]  = c.pct_change(n)
        d[f"logr_{n}"] = np.log(c / c.shift(n))

    # ── Moving averages ──
    for n in [5, 9, 14, 21, 50]:
        d[f"ema_{n}"]   = c.ewm(span=n, adjust=False).mean()
        d[f"ema_d_{n}"] = c - d[f"ema_{n}"]    # price vs EMA distance

    d["ema_cross_9_21"]  = d["ema_9"]  - d["ema_21"]
    d["ema_cross_21_50"] = d["ema_21"] - d["ema_50"]

    # ── Volatility ──
    d["atr_14"]      = _atr(h, l, c, 14)
    d["atr_norm"]    = d["atr_14"] / c            # normalised ATR
    d["volatility"]  = c.rolling(14).std()

    # ── Bollinger Bands ──
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std
    d["bb_width"]    = (bb_upper - bb_lower) / bb_mid
    d["bb_position"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── RSI ──
    d["rsi_14"] = _rsi(c, 14)
    d["rsi_7"]  = _rsi(c, 7)

    # ── MACD ──
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d["macd"]        = macd
    d["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # ── Stochastic RSI ──
    rsi = d["rsi_14"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    d["stoch_rsi"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # ── CCI ──
    tp = (h + l + c) / 3
    d["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)

    # ── Volume ──
    d["vol_zscore"]  = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    d["vol_ratio"]   = v / v.rolling(20).mean()

    # ── Hour of day (session proxy) ──
    if "time" in d.columns:
        d["hour"] = pd.to_datetime(d["time"]).dt.hour
    else:
        d["hour"] = 0

    # Keep only feature columns
    drop_cols = ["time","open","high","low","close","volume",
                 "ema_5","ema_9","ema_14","ema_21","ema_50"]
    feat_cols = [c for c in d.columns if c not in drop_cols]
    out = d[feat_cols].copy()
    return out


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_g / (avg_l + 1e-9)
    return 100 - 100/(1+rs)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────── LABELLING ────────────────────────────────

def label_data(df: pd.DataFrame, tp_pips: float = 4.0, sl_pips: float = 6.0,
               pip_size: float = 0.0001, horizon: int = 10) -> pd.Series:
    """
    Forward-looking label:
      1  = BUY  (price hits TP before SL in next `horizon` candles)
     -1  = SELL (price hits SL before TP → short opportunity)
      0  = NEUTRAL
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(closes)
    labels = np.zeros(n, dtype=int)
    tp_pts = tp_pips * pip_size
    sl_pts = sl_pips * pip_size

    for i in range(n - horizon):
        entry = closes[i]
        tp_b  = entry + tp_pts
        sl_b  = entry - sl_pts
        tp_s  = entry - tp_pts
        sl_s  = entry + sl_pts
        hit_buy_tp = hit_buy_sl = hit_sell_tp = hit_sell_sl = False
        for j in range(i+1, i+horizon+1):
            if highs[j] >= tp_b and not hit_buy_sl:
                hit_buy_tp = True; break
            if lows[j]  <= sl_b:
                hit_buy_sl = True; break
        for j in range(i+1, i+horizon+1):
            if lows[j]  <= tp_s and not hit_sell_sl:
                hit_sell_tp = True; break
            if highs[j] >= sl_s:
                hit_sell_sl = True; break

        if hit_buy_tp and not hit_sell_tp:
            labels[i] = 1
        elif hit_sell_tp and not hit_buy_tp:
            labels[i] = -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="label")


# ─────────────────────────── MODEL ────────────────────────────────────

class GridMLEngine:
    def __init__(self):
        self.model:   RandomForestClassifier = None
        self.scaler:  StandardScaler          = None
        self.feature_names: list              = []
        self.is_trained = False
        self.last_metrics: dict               = {}

    # ── TRAIN ──
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train on OHLCV DataFrame from MT5.
        Returns dict with accuracy, classification report, feature importances.
        """
        logger.info(f"Training on {len(df)} candles...")

        feats  = engineer_features(df)
        labels = label_data(df, tp_pips=4.0, sl_pips=6.0, horizon=10)

        # Align
        combined = pd.concat([feats, labels], axis=1).dropna()
        X = combined.drop("label", axis=1)
        y = combined["label"]

        self.feature_names = list(X.columns)
        logger.info(f"Features: {len(self.feature_names)}, Samples: {len(X)}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        # Time-series split
        tss = TimeSeriesSplit(n_splits=5)
        X_arr = X.values
        y_arr = y.values

        # Scaler
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X_arr)

        # Train on last split
        train_idx, test_idx = list(tss.split(X_scaled))[-1]
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y_arr[train_idx],    y_arr[test_idx]

        self.model = RandomForestClassifier(
            n_estimators    = 200,
            max_depth       = 12,
            min_samples_leaf= 10,
            class_weight    = "balanced",
            n_jobs          = -1,
            random_state    = 42,
        )
        self.model.fit(X_tr, y_tr)

        # Evaluate
        y_pred = self.model.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        report = classification_report(y_te, y_pred, target_names=["SELL","NEUTRAL","BUY"],
                                       zero_division=0, output_dict=True)

        # Feature importances
        importances = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:15]

        self.is_trained = True
        self.last_metrics = {
            "accuracy":    round(acc*100, 1),
            "report":      report,
            "n_features":  len(self.feature_names),
            "n_train":     len(X_tr),
            "n_test":      len(X_te),
            "importances": [{"name": n, "importance": round(float(v),4)} for n,v in importances],
        }

        self.save()
        logger.info(f"Training done — accuracy: {acc*100:.1f}%")
        return self.last_metrics

    # ── PREDICT ──
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict signal from the latest rows of OHLCV.
        Returns: { signal, confidence, probabilities, features }
        """
        if not self.is_trained:
            return {"signal": "NEUTRAL", "confidence": 0.0,
                    "probabilities": {"BUY":0,"NEUTRAL":100,"SELL":0}}

        feats = engineer_features(df).tail(1)
        if feats.empty or feats.isnull().any().any():
            return {"signal": "NEUTRAL", "confidence": 0.0,
                    "probabilities": {"BUY":0,"NEUTRAL":100,"SELL":0}}

        # Align to training features
        for col in self.feature_names:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[self.feature_names]

        X_scaled = self.scaler.transform(feats.values)
        proba    = self.model.predict_proba(X_scaled)[0]
        classes  = self.model.classes_

        # Map class labels to names
        label_map = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}
        proba_dict = {label_map[c]: round(float(p)*100, 1) for c, p in zip(classes, proba)}
        for k in ["BUY","NEUTRAL","SELL"]:
            proba_dict.setdefault(k, 0.0)

        signal_class = classes[np.argmax(proba)]
        signal       = label_map[signal_class]
        confidence   = round(float(np.max(proba)) * 100, 1)

        return {
            "signal":        signal,
            "confidence":    confidence,
            "probabilities": proba_dict,
            "raw_features":  feats.iloc[0].to_dict(),
        }

    # ── SAVE / LOAD ──
    def save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH,  "wb") as f: pickle.dump(self.model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        # Save feature names
        import json
        fn_path = MODEL_PATH.replace(".pkl","_features.json")
        with open(fn_path, "w") as f:
            json.dump(self.feature_names, f)
        logger.info("Model saved")

    def load(self) -> bool:
        try:
            with open(MODEL_PATH,  "rb") as f: self.model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            import json
            fn_path = MODEL_PATH.replace(".pkl","_features.json")
            if os.path.exists(fn_path):
                with open(fn_path) as f:
                    self.feature_names = json.load(f)
            self.is_trained = True
            logger.info("Model loaded from disk")
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
