import os
import re
import json
import time
import math
import pandas as pd
import numpy as np
import psycopg2
import requests
import pika
from datetime import datetime
import subprocess

# ========= CONFIG =========
CSV_FILE = "./data_backtest/XAUUSD_M15_202509010100_202511062345.csv"
H1_FILE  = "./data_backtest/XAUUSD_H1_202509010100_202511062300.csv"
H4_FILE  = "./data_backtest/XAUUSD_H4_202509010000_202511070000.csv"
NODE_URL = "http://127.0.0.1:3000/candle"
RABBIT_URL = "amqp://localhost"
DB_CFG = dict(host="localhost", port=54321, database="ML", user="postgres", password="postgres")

INITIAL_BALANCE = 50.0
CONF_THRESHOLD = 0.6

# Faktor realistik tambahan
SPREAD_POINTS = 20
SLIPPAGE_POINTS = 5
COMMISSION_RATE = 0.0002
TP_PIPS = 200
SL_PIPS = 100
LATENCY_MIN = 0.05
LATENCY_MAX = 0.15
# ==========================

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
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi14(df["close"])
    df["adx14"] = adx14(df[["high","low","close"]])
    return df.fillna(method="bfill").fillna(method="ffill")

def load_h1_h4():
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

# ---------- Komunikasi Node + MQ ----------
def request_ai_via_node(payload, timeout=10):
    """Kirim candle ke Node.js lalu tunggu hasil prediksi dari MQ"""
    import uuid
    response = None
    req_id = str(uuid.uuid4())
    payload = {**payload, "req_id": req_id}

    print(f"🆔 [REQ_ID] {req_id} - Mulai kirim ke Node.js")

    connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URL))
    channel = connection.channel()

    # pastikan queue hasil sudah ada dan ter-bind ke exchange "ai" routing "predict.result"
    channel.exchange_declare(exchange="ai", exchange_type="topic", durable=True)
    reply_queue = f"ai_results_{req_id[:8]}"
    channel.queue_declare(queue=reply_queue, durable=False, exclusive=True, auto_delete=True)
    channel.queue_bind(exchange="ai", queue=reply_queue, routing_key=f"predict.result.{req_id}")


    def on_response(ch, method, props, body):
        nonlocal response
        try:
            msg = json.loads(body)
            rid = msg.get("req_id")
            print(f"📨 [MQ RECEIVE] Dapat pesan req_id={rid}")
            if rid == req_id:
                print(f"✅ [MATCH] Balasan cocok req_id={req_id}")
                response = msg
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ch.stop_consuming()
            else:
                print(f"⚠️ [SKIP] req_id tidak cocok ({rid} ≠ {req_id})")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        except Exception as e:
            print(f"❌ [ERROR MQ] {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=reply_queue, on_message_callback=on_response, auto_ack=False)

    # kirim ke Node.js
    try:
        print(f"📤 [HTTP SEND] Mengirim ke Node.js: {NODE_URL}")
        r = requests.post(NODE_URL, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Node.js error {r.status_code}")
        else:
            print(f"📬 [HTTP OK] req_id={req_id} terkirim ke Node.js")
    except Exception as e:
        print(f"❌ Gagal kirim ke Node: {e}")

    start = time.time()
    while not response and time.time() - start < timeout:
        channel.connection.process_data_events(time_limit=0.5)

    connection.close()

    if not response:
        print(f"⏰ [TIMEOUT] Tidak ada respons untuk req_id={req_id}")
        return {"action": "HOLD", "confidence": 0.0}

    print("🔥 RESPONSE MASUK DARI MQ:")
    print(json.dumps(response, indent=2))

    return response.get("result", {"action": "HOLD", "confidence": 0.0})

# ---------- Main backtest ----------
def run_backtest():
    symbol = "XAUUSD"
    timeframe = "M15"
    df = load_tf_csv(CSV_FILE)
    h1, h4 = load_h1_h4()

    feat = df.copy()
    feat["ema20"] = ema(df["close"], 20)
    feat["ema50"] = ema(df["close"], 50)
    feat["ema100"] = ema(df["close"], 100)
    feat["rsi14"] = rsi14(df["close"])
    feat["adx14"] = adx14(df)

    feat = pd.merge_asof(
        feat,
        h1[["datetime","ema50","rsi14","adx14"]].rename(columns={
            "ema50":"ema50_h1","rsi14":"rsi_h1","adx14":"adx_h1"
        }),
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

    print(f"🚀 Simulasi Node+MQ | candles: {len(df)}")

    for i, row in df.iterrows():
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "time": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["tickvol"]),
            "is_closed": True,
            "indicators": {
                "ema_20": float(feat.loc[i,"ema20"]),
                "ema_50": float(feat.loc[i,"ema50"]),
                "ema_100": float(feat.loc[i,"ema100"]),
                "rsi_14": float(feat.loc[i,"rsi14"]),
                "adx_14": float(feat.loc[i,"adx14"]),
                "ema_50_h1": float(feat.loc[i,"ema50_h1"]),
                "rsi_14_h1": float(feat.loc[i,"rsi_h1"]),
                "adx_14_h1": float(feat.loc[i,"adx_h1"]),
                "adx_14_h4": float(feat.loc[i,"adx_h4"])
            }
        }

        time.sleep(np.random.uniform(LATENCY_MIN, LATENCY_MAX))
        print(f"\n📤 [SEND] {row['datetime']} | {symbol} {timeframe}")
        print(json.dumps(payload, indent=2))
        ai = request_ai_via_node(payload)
        print(f"📥 [RECV] {ai}\n")
        action = ai.get("action", "HOLD")
        conf = float(ai.get("confidence", 0))
        price = float(row["close"])

        spread = SPREAD_POINTS * 0.0001
        slip = SLIPPAGE_POINTS * 0.0001

        if position is None and conf >= CONF_THRESHOLD:
            if action in ["BUY","SELL"]:
                position = {"type": action, "entry": price, "time": row["datetime"]}
                print(f"✅ {row['datetime']} | {action} @ {price:.2f} (conf={conf:.2f}) OPEN")
        elif position:
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
                insert_trade(conn, trade)
                trades.append(trade)
                print(f"💰 {reason} | {position['type']} | P/L={pl:.4f} | Bal={balance:.2f}")
                position=None

        print(f"[{i+1}/{len(df)}] 🕒 {row['datetime']} | AI={action} ({conf:.2f}) | Bal={balance:.2f}")

    conn.close()
    pd.DataFrame(trades).to_csv("./logs/trade_history_node.csv", index=False)
    print(f"✅ Done | Trades={len(trades)} | Final balance={balance:.2f}")

if __name__ == "__main__":
    run_backtest()
