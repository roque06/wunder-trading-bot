import os, time, requests, pandas as pd, ta
from datetime import datetime, UTC

# ==============================
# CONFIG GENERAL
# ==============================
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
WUNDER_WEBHOOK = "https://wtalerts.com/bot/custom"
POLL_SECONDS = 20
LOG_CSV = "trades_log.csv"

INITIAL_CAPITAL = 100.0
capital = INITIAL_CAPITAL

# ==============================
# ESTRATEGIA
# ==============================
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 45, 60
RSI_SHORT_MIN, RSI_SHORT_MAX = 40, 55
ATR_PERIOD = 14
ATR_MULT_RANGE_BLOCK = 0.15
ATR_ACTIVE_FACTOR = 1.0

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
ENTER_LONG  = "ENTER-LONG_Binance_BTCUSDT_Bot-Test_5M_d9b7a795e987b1d1faa1095a"
ENTER_SHORT = "ENTER-SHORT_Binance_BTCUSDT_Bot-Test_5M_d9b7a795e987b1d1faa1095a"
EXIT_ALL    = "EXIT-ALL_Binance_BTCUSDT_Bot-Test_5M_d9b7a795e987b1d1faa1095a"

# ==============================
# TELEGRAM
# ==============================
TELEGRAM_TOKEN = "7543685147:AAHGtjiy-wAQ7uIMTsuahx57MQ-8veDgcls"
TELEGRAM_CHAT_ID = "5324968421"

def send_telegram_message(text: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("❌ Error enviando a Telegram:", e)

# ==============================
# FUNCIONES PRINCIPALES
# ==============================
def fetch_klines(symbol, interval, limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
        return df
    except Exception as e:
        print("⚠️ Error Binance:", e)
        time.sleep(10)
        return fetch_klines(symbol, interval, limit)

def send_signal(code):
    try:
        payload = {"code": code}
        r = requests.post(WUNDER_WEBHOOK, json=payload,
                          headers={"Content-Type":"application/json"}, timeout=10)
        print(f"[{datetime.now(UTC)}] Signal -> {code} | status={r.status_code}")
    except Exception as e:
        print("⚠️ Error enviando señal:", e)

def log_trade(**kw):
    os.makedirs(os.path.dirname(LOG_CSV) or ".", exist_ok=True)
    row = {**kw, "ts_utc": datetime.now(UTC).isoformat()}
    pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False)

# ==============================
# INDICADORES
# ==============================
def compute_indicators(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=ATR_PERIOD)
    df["atr_ma"] = df["atr"].rolling(ATR_PERIOD).mean()
    return df

def in_range_zone(price, ema_f, ema_s):
    if pd.isna(ema_f) or pd.isna(ema_s): return True
    return abs(ema_f-ema_s)/price < ATR_MULT_RANGE_BLOCK/100

def atr_active(atr_now, atr_ma):
    if pd.isna(atr_now) or pd.isna(atr_ma): return False
    return atr_now >= atr_ma * ATR_ACTIVE_FACTOR

# ==============================
# ANÁLISIS DIARIO
# ==============================
def analyze_trades():
    if not os.path.exists(LOG_CSV): return
    df = pd.read_csv(LOG_CSV)
    if df.empty or "price" not in df.columns or "event" not in df.columns: return
    df = df[df["event"].str.contains("ENTER|EXIT", na=False)]
    if df.empty: return

    df["profit"] = df["price"].diff()
    df = df.dropna()
    wins, losses = df[df["profit"]>0]["profit"], df[df["profit"]<0]["profit"]

    win_rate = len(wins)/len(df)*100 if len(df) else 0
    profit_factor = wins.sum()/abs(losses.sum()) if not losses.empty else float("inf")
    expectancy = df["profit"].mean()
    equity = df["profit"].cumsum()
    drawdown = (equity - equity.cummax()).min()

    report = (
        f"📊 <b>REPORTE DIARIO - {datetime.now(UTC).strftime('%Y-%m-%d')}</b>\n\n"
        f"✅ Win Rate: <b>{win_rate:.2f}%</b>\n"
        f"💰 Profit Factor: <b>{profit_factor:.2f}</b>\n"
        f"📈 Expectativa: <b>{expectancy:.4f}</b>\n"
        f"📉 Max Drawdown: <b>{drawdown:.4f}</b>\n"
        f"🕒 Hora UTC: {datetime.now(UTC).strftime('%H:%M:%S')}"
    )
    send_telegram_message(report)
    print(report)

# ==============================
# BOT + MONITOR PRINCIPAL
# ==============================
def main():
    global capital
    print("🚀 Bot de trading inteligente iniciado.")
    send_telegram_message("🤖 Bot activo y escuchando el mercado...")

    last_side, entry_price = None, None
    last_analysis = datetime.now(UTC).date()
    last_size = os.path.getsize(LOG_CSV) if os.path.exists(LOG_CSV) else 0

    while True:
        try:
            # === 1. DATOS DEL MERCADO ===
            df = compute_indicators(fetch_klines(SYMBOL, INTERVAL, 300))
            price, ema_f, ema_s, rsi = df["close"].iloc[-1], df["ema_fast"].iloc[-1], df["ema_slow"].iloc[-1], df["rsi"].iloc[-1]
            atr_now, atr_ma = df["atr"].iloc[-1], df["atr_ma"].iloc[-1]
            print(f"⏱️ {datetime.now(UTC)} | Precio={price:.2f} | EMA9={ema_f:.2f} | EMA21={ema_s:.2f} | RSI={rsi:.1f} | Pos={last_side}")

            # === 2. LÓGICA DE ESTRATEGIA ===
            if in_range_zone(price, ema_f, ema_s) or not atr_active(atr_now, atr_ma):
                time.sleep(POLL_SECONDS); continue

            if ema_f > ema_s and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX and last_side != "LONG":
                if last_side == "SHORT": send_signal(EXIT_ALL)
                send_signal(ENTER_LONG)
                last_side, entry_price = "LONG", price
                log_trade(event="ENTER-LONG", price=price, rsi=rsi)
                send_telegram_message(f"🟢 LONG abierta a {price:.2f} USDT")

            elif ema_f < ema_s and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX and last_side != "SHORT":
                if last_side == "LONG": send_signal(EXIT_ALL)
                send_signal(ENTER_SHORT)
                last_side, entry_price = "SHORT", price
                log_trade(event="ENTER-SHORT", price=price, rsi=rsi)
                send_telegram_message(f"🔴 SHORT abierta a {price:.2f} USDT")

            elif last_side and 45 <= rsi <= 55:
                send_signal(EXIT_ALL)
                diff = ((price-entry_price)/entry_price)*100 if last_side=="LONG" else ((entry_price-price)/entry_price)*100
                status = "🟩 GANANCIA" if diff>0 else "🟥 PÉRDIDA"
                capital *= (1+diff/100)
                log_trade(event=f"EXIT-{last_side}", price=price, rsi=rsi, profit_pct=diff, capital=capital)
                send_telegram_message(f"⚪ Cierre {last_side} | {status}: {diff:.2f}% | 💵 Capital: ${capital:.2f}")
                last_side, entry_price = None, None

            # === 3. MONITORIZA NUEVOS REGISTROS EN LOG ===
            size_now = os.path.getsize(LOG_CSV) if os.path.exists(LOG_CSV) else 0
            if size_now != last_size:
                df = pd.read_csv(LOG_CSV)
                if not df.empty:
                    last_row = df.iloc[-1]
                    event = str(last_row.get("event",""))
                    price = last_row.get("price",0)
                    ts = last_row.get("ts_utc", datetime.now(UTC))
                    if "ENTER" in event:
                        send_telegram_message(f"📩 Nueva entrada detectada → {event}\n💰 {price} | 🕒 {ts}")
                    elif "EXIT" in event:
                        send_telegram_message(f"📤 Nueva salida detectada → {event}\n💰 {price} | 🕒 {ts}")
                last_size = size_now

            # === 4. REPORTE DIARIO ===
            today = datetime.now(UTC).date()
            if today != last_analysis:
                analyze_trades()
                last_analysis = today

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print("⚠️ Error general:", e)
            time.sleep(15)

# ==============================
# EJECUCIÓN
# ==============================
if __name__ == "__main__":
    print("✅ Binance respondió correctamente, iniciando cálculos...")
    main()
