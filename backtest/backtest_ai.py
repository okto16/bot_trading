import os
import re
import json
import math
import time
import pandas as pd
import numpy as np
import psycopg2
import subprocess
from datetime import datetime, timezone

# ========= CONFIG =========
CSV_FILE = "./data_backtest/XAUUSD_M15_202509010100_202511062345.csv"   # file utama M15
H1_FILE  = "./data_backtest/XAUUSD_H1_202509010100_202511062300.csv"
H4_FILE  = "./data_backtest/XAUUSD_H4_202509010000_202511070000.csv"
AI_PREDICT_SCRIPT = "predict_multitf.py"
DB_CFG = dict(host="localhost", port=54321, database="ML", user="postgres", password="postgres")
INITIAL_BALANCE = 50.0
CONF_THRESHOLD = 0.60

# Faktor realistik tambahan
SPREAD_POINTS = 20
SLIPPAGE_POINTS = 5
COMMISSION_RATE = 0.0002
TP_PIPS = 200
SL_PIPS = 100
LATENCY_MIN = 0.05
LATENCY_MAX = 0.15
# ==========================

# ---------- util indikator ----------
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi14(close):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def adx14(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(14).mean()

# ---------- Multi-TF loader ----------
def autodetect_sep(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    if "\t" in head: return "\t"
    if ";" in head: return ";"
    return ","

def load_tf_csv(path):
    sep = autodetect_sep(path)
    df = pd.read_csv(
        path, sep=sep, skiprows=1,
        names=["date","time","open","high","low","close","tickvol","vol","spread"],
        comment="<", skip_blank_lines=True
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi14(df["close"])
    df["adx14"] = adx14(df[["high","low","close"]].rename(columns=str))
    return df.fillna(method="bfill").fillna(method="ffill")

def load_h1_h4():
    if not os.path.exists(H1_FILE) or not os.path.exists(H4_FILE):
        raise FileNotFoundError("❌ file H1 atau H4 tidak ditemukan.")
    h1 = load_tf_csv(H1_FILE)
    h4 = load_tf_csv(H4_FILE)
    return h1, h4

# ---------- DB ----------
def db():
    conn = psycopg2.connect(**DB_CFG)
    conn.autocommit = True
    return conn

def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            candle_id BIGINT PRIMARY KEY,
            ema_20 DOUBLE PRECISION,
            ema_50 DOUBLE PRECISION,
            ema_100 DOUBLE PRECISION,
            rsi_14 DOUBLE PRECISION,
            macd DOUBLE PRECISION,
            macd_signal DOUBLE PRECISION,
            adx_14 DOUBLE PRECISION,
            atr_14 DOUBLE PRECISION,
            stochastic_k DOUBLE PRECISION,
            stochastic_d DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_predictions (
            candle_id BIGINT PRIMARY KEY,
            model_version TEXT,
            prediction DOUBLE PRECISION,
            action TEXT,
            confidence DOUBLE PRECISION,
            reasoning TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            timeframe TEXT,
            entry_time TIMESTAMPTZ,
            exit_time TIMESTAMPTZ,
            entry_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            action TEXT,
            profit_loss DOUBLE PRECISION,
            balance_after DOUBLE PRECISION,
            ai_confidence DOUBLE PRECISION,
            model_version TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

def upsert_candle(conn, row, symbol, timeframe):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO candles (symbol, timeframe, time, open, high, low, close, volume, spread, is_closed, created_at, tick_volume)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),%s)
        ON CONFLICT (symbol, timeframe, time)
        DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
                      volume=EXCLUDED.volume, spread=EXCLUDED.spread, is_closed=EXCLUDED.is_closed, tick_volume=EXCLUDED.tick_volume
        RETURNING id;
        """, (
            symbol, timeframe, row["datetime"], float(row["open"]), float(row["high"]), float(row["low"]),
            float(row["close"]), int(row.get("vol", 0)), int(row.get("spread", 0)), True, int(row["tickvol"])
        ))
        return cur.fetchone()[0]

def upsert_indicators(conn, candle_id, feat):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO indicators (candle_id, ema_20, ema_50, ema_100, rsi_14, macd, macd_signal, adx_14, atr_14, stochastic_k, stochastic_d)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (candle_id) DO UPDATE SET
          ema_20=EXCLUDED.ema_20, ema_50=EXCLUDED.ema_50, ema_100=EXCLUDED.ema_100, rsi_14=EXCLUDED.rsi_14,
          macd=EXCLUDED.macd, macd_signal=EXCLUDED.macd_signal, adx_14=EXCLUDED.adx_14, atr_14=EXCLUDED.atr_14,
          stochastic_k=EXCLUDED.stochastic_k, stochastic_d=EXCLUDED.stochastic_d;
        """, (
            candle_id,
            feat.get("ema20"), feat.get("ema50"), feat.get("ema100"),
            feat.get("rsi14"), feat.get("macd"), feat.get("macd_signal"),
            feat.get("adx14"), feat.get("atr14"), feat.get("sto_k"), feat.get("sto_d")
        ))

def upsert_prediction(conn, candle_id, action, confidence, reasoning="backtest"):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO ai_predictions (candle_id, model_version, prediction, action, confidence, reasoning)
        VALUES (%s, 'latest', %s, %s, %s, %s)
        ON CONFLICT (candle_id) DO UPDATE SET
            prediction=EXCLUDED.prediction, action=EXCLUDED.action, confidence=EXCLUDED.confidence, reasoning=EXCLUDED.reasoning, created_at=NOW();
        """, (candle_id, float(confidence), action, float(confidence), reasoning))

def insert_trade(conn, t):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO backtest_trades (symbol, timeframe, entry_time, exit_time, entry_price, exit_price, action,
                                     profit_loss, balance_after, ai_confidence, model_version)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'latest');
        """, (
            t["symbol"], t["timeframe"], t["entry_time"], t["exit_time"],
            t["entry_price"], t["exit_price"], t["action"], t["profit_loss"],
            t["balance_after"], t["ai_confidence"]
        ))

# ---------- AI ----------
def call_ai(payload):
    proc = subprocess.Popen(
        ["python", AI_PREDICT_SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate(json.dumps(payload))
    if err and err.strip():
        print("⚠️ AI stderr:", err.strip())
    try:
        return json.loads(out)
    except Exception:
        return {"action": "HOLD", "confidence": 0.0}

# ---------- CSV loader ----------
def parse_symbol_tf_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]+)_(M1|M5|M15|M30|H1|H4|D1|W1|MN)\.csv", name, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    return "XAUUSD", "M15"

def load_mt5_csv(path):
    sep = autodetect_sep(path)
    df = pd.read_csv(
        path, sep=sep, comment="<",
        names=["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"],
        skiprows=1, skip_blank_lines=True
    )
    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str),
                                    format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "tickvol", "vol", "spread"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df

# ---------- Backtest utama ----------
def run_backtest():
    symbol, timeframe = parse_symbol_tf_from_filename(CSV_FILE)
    df = load_mt5_csv(CSV_FILE)
    h1, h4 = load_h1_h4()

    feat = pd.DataFrame({
        "close": df["close"],
        "high": df["high"],
        "low": df["low"]
    })
    feat["ema20"] = ema(df["close"], 20)
    feat["ema50"] = ema(df["close"], 50)
    feat["ema100"] = ema(df["close"], 100)
    feat["rsi14"] = rsi14(df["close"])
    feat["adx14"] = adx14(df[["high","low","close"]].rename(columns=str))

    # --- merge multi-TF ---
    feat["datetime"] = df["datetime"]
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    h4["datetime"] = pd.to_datetime(h4["datetime"])

    feat = pd.merge_asof(
        feat,
        h1[["datetime","ema50","rsi14","adx14"]].rename(columns={
            "ema50":"ema50_h1","rsi14":"rsi_h1","adx14":"adx_h1"}),
        on="datetime", direction="backward"
    )
    feat = pd.merge_asof(
        feat,
        h4[["datetime","adx14"]].rename(columns={"adx14":"adx_h4"}),
        on="datetime", direction="backward"
    )
    feat = feat.fillna(method="bfill").fillna(method="ffill")

    conn = db()
    ensure_tables(conn)

    balance = INITIAL_BALANCE
    position = None
    trades = []

    print(f"🚀 Backtest (Realistic) {symbol} {timeframe} | candles: {len(df)}")

    for i, row in df.iterrows():
        candle_id = upsert_candle(conn, row, symbol, timeframe)

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "time": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "close": float(row["close"]),
            "volume": float(row.get("tickvol", 0)),
            "indicators": {
                "ema_20": float(feat.loc[i,"ema20"]),
                "ema_50": float(feat.loc[i,"ema50"]),
                "ema_100": float(feat.loc[i,"ema100"]),
                "rsi_14": float(feat.loc[i,"rsi14"]),
                "adx_14": float(feat.loc[i,"adx14"]),
                "volume": float(row.get("tickvol",0)),
                "ema_50_h1": float(feat.loc[i,"ema50_h1"]),
                "rsi_14_h1": float(feat.loc[i,"rsi_h1"]),
                "adx_14_h1": float(feat.loc[i,"adx_h1"]),
                "adx_14_h4": float(feat.loc[i,"adx_h4"])
            }
        }

        upsert_indicators(conn, candle_id, {
            "ema20": payload["indicators"]["ema_20"],
            "ema50": payload["indicators"]["ema_50"],
            "ema100": payload["indicators"]["ema_100"],
            "rsi14": payload["indicators"]["rsi_14"],
            "macd": None, "macd_signal": None,
            "adx14": payload["indicators"]["adx_14"],
            "atr14": None, "sto_k": None, "sto_d": None
        })

        time.sleep(np.random.uniform(LATENCY_MIN, LATENCY_MAX))
        ai = call_ai(payload)
        action = ai.get("action","HOLD")
        conf = float(ai.get("confidence",0) or 0)
        upsert_prediction(conn, candle_id, action, conf, "backtest")

        price = float(row["close"])
        spread = SPREAD_POINTS * 0.0001
        slip = SLIPPAGE_POINTS * 0.0001

        if position is None and conf >= CONF_THRESHOLD:
            if action in ["BUY","SELL"]:
                position = {"type":action,"entry":price,"time":row["datetime"]}
                print(f"✅ {row['datetime']} | {action} @ {price:.2f} (conf={conf:.2f}) OPEN")
        elif position is not None:
            stop_loss = position["entry"] - SL_PIPS*0.0001 if position["type"]=="BUY" else position["entry"] + SL_PIPS*0.0001
            take_profit = position["entry"] + TP_PIPS*0.0001 if position["type"]=="BUY" else position["entry"] - TP_PIPS*0.0001
            exit_price = price - spread - slip if position["type"]=="BUY" else price + spread + slip
            reason=None
            if (position["type"]=="BUY" and row["low"]<=stop_loss) or (position["type"]=="SELL" and row["high"]>=stop_loss):
                reason="STOPLOSS"
            elif (position["type"]=="BUY" and row["high"]>=take_profit) or (position["type"]=="SELL" and row["low"]<=take_profit):
                reason="TAKEPROFIT"
            elif conf>=CONF_THRESHOLD and action!=position["type"]:
                reason="SIGNAL_REVERSAL"
            if reason:
                pl=(exit_price-position["entry"]) if position["type"]=="BUY" else (position["entry"]-exit_price)
                pl-=abs(pl)*COMMISSION_RATE
                balance+=pl
                trade={
                    "symbol":symbol,"timeframe":timeframe,
                    "entry_time":position["time"],"exit_time":row["datetime"],
                    "entry_price":position["entry"],"exit_price":exit_price,
                    "action":position["type"],"profit_loss":pl,
                    "balance_after":balance,"ai_confidence":conf
                }
                insert_trade(conn,trade)
                trades.append(trade)
                print(f"💰 {reason} | {position['type']} | P/L={pl:.4f} | Bal={balance:.2f}")
                position=None
            else:
                print(f"📊 {row['datetime']} | Posisi aktif {position['type']} @ {position['entry']:.2f}")
        candle_time=row["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{i+1}/{len(df)}] 🕒 {candle_time} | AI={action} ({conf:.2f}) | Bal={balance:.2f}")

    if position is not None:
        last=df.iloc[-1]
        price=float(last["close"])
        exit_price=price-SPREAD_POINTS*0.0001 if position["type"]=="BUY" else price+SPREAD_POINTS*0.0001
        pl=(exit_price-position["entry"]) if position["type"]=="BUY" else (position["entry"]-exit_price)
        pl-=abs(pl)*COMMISSION_RATE
        balance+=pl
        trade={
            "symbol":symbol,"timeframe":timeframe,
            "entry_time":position["time"],"exit_time":last["datetime"],
            "entry_price":position["entry"],"exit_price":exit_price,
            "action":position["type"],"profit_loss":pl,
            "balance_after":balance,"ai_confidence":1.0
        }
        insert_trade(conn,trade)
        trades.append(trade)
        print(f"⚙️ FORCE CLOSE {position['type']} @ {exit_price:.2f} | Final Bal={balance:.2f}")

    pd.DataFrame(trades).to_csv("./logs/trade_history.csv",index=False)
    print(f"✅ Selesai. Trades: {len(trades)} | Final balance: {balance:.2f}")
    conn.close()
    return trades

def retrain():
    print("🔁 ai_feedback.py ...")
    subprocess.call(["python","ai_feedback.py"])

if __name__=="__main__":
    t=run_backtest()
    if t:
        retrain()
