import os, time, json, requests, pandas as pd, ta
from datetime import datetime, UTC

# ==============================
# CONFIG GENERAL
# ==============================
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
WUNDER_WEBHOOK = "https://wtalerts.com/bot/custom"
POLL_SECONDS = 60
LOG_CSV = "trades_log.csv"
STATE_FILE = "bot_state.json"
MIN_HOLD_BARS = 3
DUP_SIGNAL_COOLDOWN_SEC = 10

INITIAL_CAPITAL = 100.0
capital = INITIAL_CAPITAL

# ==============================
# ESTRATEGIA
# ==============================
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 40, 65
RSI_SHORT_MIN, RSI_SHORT_MAX = 35, 60
EMA_DIFF_MARGIN = 0.001  # 0.1% diferencia mínima para confirmar cruce
ATR_PERIOD = 14
ATR_MULT_RANGE_BLOCK = 0.15
ATR_ACTIVE_FACTOR = 1.0
ATR_SL_MULT = 1.8
ATR_TP_MULT = 2.0

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
ENTER_LONG  = "ENTER-LONG_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978"
ENTER_SHORT = "ENTER-SHORT_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978"
EXIT_ALL    = "EXIT-ALL_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978"

# ==============================
# TELEGRAM
# ==============================
TELEGRAM_TOKEN = "7543685147:AAGtQjY-wA97qmUTsahux75MQ-8vYeDgcls"
TELEGRAM_CHAT_ID = "1216693645"

def send_telegram_message(text: str):
    """Envía mensajes a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Error Telegram: {r.status_code}")
    except Exception as e:
        print("❌ Error enviando a Telegram:", e)

def test_telegram():
    """Prueba inicial de conexión a Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "✅ Prueba de conexión Telegram exitosa.\nEl bot ya puede enviarte notificaciones.",
            "parse_mode": "HTML"
        }
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("📩 Prueba de Telegram enviada correctamente ✅")
        else:
            print(f"⚠️ Telegram devolvió error: {r.status_code}")
    except Exception as e:
        print("❌ Error durante la prueba de Telegram:", e)

# ==============================
# ESTADO
# ==============================
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_side": None, "entry_price": None, "bars_held": 0, "last_signal": "", "last_signal_ts": 0}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"last_side": None, "entry_price": None, "bars_held": 0, "last_signal": "", "last_signal_ts": 0}

def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)

# ==============================
# FUNCIONES PRINCIPALES
# ==============================
def fetch_klines(symbol, interval, limit=500, retries=5, backoff=5):
    # 🔄 Endpoint más estable (Binance US evita bloqueos por IP)
    primary_url = "https://api.binance.us/api/v3/klines"
    backup_url  = "https://api.binance.com/api/v3/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(retries):
        try:
            # 🔁 Intenta primero con Binance US
            r = requests.get(primary_url, params=params, headers=headers, timeout=10)
            # Si Binance US no responde, prueba con Binance global
            if r.status_code in (418, 451) or not r.ok:
                print(f"⚠️ Endpoint US bloqueado (código {r.status_code}), probando Binance global...")
                r = requests.get(backup_url, params=params, headers=headers, timeout=10)

            r.raise_for_status()
            data = r.json()

            # Limpieza de datos (aseguramos estructura)
            clean_data = [row[:12] for row in data if isinstance(row, list) and len(row) >= 12]

            df = pd.DataFrame(clean_data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
            ])

            # Conversión de tipos
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Elimina filas con datos faltantes
            df = df.dropna(subset=["close"])

            return df

        except Exception as e:
            print(f"⚠️ Error Binance (intento {i+1}/{retries}): {e}")
            time.sleep(backoff * (i + 1))

    raise RuntimeError("⚠️ Binance no responde tras varios intentos.")

def send_signal(code):
    """Envía señal a WunderTrading y guarda estado."""
    state = load_state()
    now_ts = time.time()
    if state.get("last_signal") == code and (now_ts - state.get("last_signal_ts", 0)) < DUP_SIGNAL_COOLDOWN_SEC:
        return
    try:
        payload = {"code": code}
        r = requests.post(WUNDER_WEBHOOK, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        print(f"[{datetime.now(UTC)}] Signal -> {code} | status={r.status_code}")
        state["last_signal"] = code
        state["last_signal_ts"] = now_ts
        save_state(state)
    except Exception as e:
        print("⚠️ Error enviando señal:", e)

def log_trade(**kw):
    os.makedirs(os.path.dirname(LOG_CSV) or ".", exist_ok=True)
    row = {**kw, "ts_utc": datetime.now(UTC).isoformat()}
    try:
        pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False)
    except Exception as e:
        print("⚠️ Error escribiendo CSV:", e)

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
    return abs(ema_f-ema_s)/price < ATR_MULT_RANGE_BLOCK/100 if not (pd.isna(ema_f) or pd.isna(ema_s)) else True

def atr_active(atr_now, atr_ma):
    return atr_now >= atr_ma * ATR_ACTIVE_FACTOR if not (pd.isna(atr_now) or pd.isna(atr_ma)) else False

# ==============================
# ENTRADAS Y SALIDAS
# ==============================
def place_entry(side: str, price: float, rsi: float):
    state = load_state()
    last_side = state.get("last_side")
    entry_price = state.get("entry_price")

    if last_side and last_side != side:
        send_signal(EXIT_ALL)
        diff_pct = ((price - entry_price)/entry_price)*100 if last_side=="LONG" else ((entry_price - price)/entry_price)*100
        log_trade(event=f"EXIT-{last_side}", price=price, rsi=rsi, profit_pct=diff_pct)
        send_telegram_message(f"⚪ Cierre {last_side} por reversión | {'🟩' if diff_pct>0 else '🟥'} {diff_pct:.2f}%")
        time.sleep(5)

    if side == "LONG":
        send_signal(ENTER_LONG)
        send_telegram_message(f"🟢 LONG abierta a {price:.2f} USDT")
        log_trade(event="ENTER-LONG", price=price, rsi=rsi)
    else:
        send_signal(ENTER_SHORT)
        send_telegram_message(f"🔴 SHORT abierta a {price:.2f} USDT")
        log_trade(event="ENTER-SHORT", price=price, rsi=rsi)

    time.sleep(10)
    state.update({"last_side": side, "entry_price": float(price), "bars_held": 0})
    save_state(state)

# ==============================
# MONITOR PRINCIPAL
# ==============================
def main():
    print("🚀 Bot de trading inteligente iniciado.")
    send_telegram_message("🤖 Bot activo y escuchando el mercado...")

    state = load_state()
    while True:
        try:
            df = compute_indicators(fetch_klines(SYMBOL, INTERVAL, 300))
            price = df["close"].iloc[-1]
            ema_f, ema_s = df["ema_fast"].iloc[-1], df["ema_slow"].iloc[-1]
            rsi = df["rsi"].iloc[-1]
            atr_now, atr_ma = df["atr"].iloc[-1], df["atr_ma"].iloc[-1]

            print(f"⏱️ {datetime.now(UTC)} | Precio={price:.2f} | EMA9={ema_f:.2f} | EMA21={ema_s:.2f} | RSI={rsi:.1f} | ATR={atr_now:.2f} | Pos={state.get('last_side')}")
            print(f"🔎 DEBUG → ema_f>ema_s? {ema_f>ema_s}, RSI={rsi:.1f}")

            if in_range_zone(price, ema_f, ema_s) or not atr_active(atr_now, atr_ma):
                time.sleep(POLL_SECONDS); continue

            last_side = state.get("last_side")
            bars_held = state.get("bars_held", 0)
            can_flip = (bars_held >= MIN_HOLD_BARS)

            if ema_f > ema_s * (1 + EMA_DIFF_MARGIN) and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX and (last_side != "LONG" and (last_side is None or can_flip)):
                place_entry("LONG", price, rsi)
            elif ema_f < ema_s * (1 - EMA_DIFF_MARGIN) and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX and (last_side != "SHORT" and (last_side is None or can_flip)):
                place_entry("SHORT", price, rsi)

            time.sleep(POLL_SECONDS)
        except Exception as e:
            print("⚠️ Error general:", e)
            time.sleep(15)

# ==============================
# EJECUCIÓN
# ==============================
if __name__ == "__main__":
    print("✅ Binance respondió correctamente, iniciando cálculos...")
    test_telegram()
    main()

