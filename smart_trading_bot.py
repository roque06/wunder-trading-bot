import os, time, json, sys, requests, pandas as pd, ta, threading
from datetime import datetime, UTC

# ==============================
# CONFIG GENERAL
# ==============================
SYMBOLS = ["BTCUSDT"]  # Puedes añadir más símbolos
INTERVAL = "5m"
WUNDER_WEBHOOK = "https://wtalerts.com/bot/custom"
POLL_SECONDS = 60
LOG_CSV = "trades_log.csv"
STATE_FILE_TPL = "state_{symbol}.json"   # Estado por símbolo
MIN_HOLD_BARS = 3
DUP_SIGNAL_COOLDOWN_SEC = 10

INITIAL_CAPITAL = 100.0
capital = INITIAL_CAPITAL

# ==============================
# ESTRATEGIA
# ==============================
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX   = 40, 65
RSI_SHORT_MIN, RSI_SHORT_MAX = 35, 60

EMA_DIFF_MARGIN = 0.001
ATR_PERIOD = 14
ATR_MULT_RANGE_BLOCK = 0.15
ATR_ACTIVE_FACTOR = 1.0
ATR_SL_MULT = 1.8
ATR_TP_MULT = 2.0

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
SIGNAL_CODES = {
    "BTCUSDT": {
        "ENTER_LONG":  "ENTER-LONG_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "ENTER_SHORT": "ENTER-SHORT_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "EXIT_ALL":    "EXIT-ALL_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
    },
}

# ==============================
# TELEGRAM
# ==============================
TELEGRAM_TOKEN = "7543685147:AAGtQjY-wA97qmUTsahux75MQ-8vYeDgcls"
TELEGRAM_CHAT_ID = "1216693645"

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Error Telegram: {r.status_code}", flush=True)
    except Exception as e:
        print("❌ Error enviando a Telegram:", e, flush=True)

def test_telegram():
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "✅ Prueba de conexión Telegram exitosa.\nEl bot ya puede enviarte notificaciones.",
            "parse_mode": "HTML"
        }
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("📩 Prueba de Telegram enviada correctamente ✅", flush=True)
        else:
            print(f"⚠️ Telegram devolvió error: {r.status_code}", flush=True)
    except Exception as e:
        print("❌ Error durante la prueba de Telegram:", e, flush=True)

# ==============================
# ESTADO (por símbolo)
# ==============================
def state_path(symbol: str) -> str:
    return STATE_FILE_TPL.format(symbol=symbol)

def load_state(symbol: str):
    path = state_path(symbol)
    if not os.path.exists(path):
        return {"last_side": None, "entry_price": None, "bars_held": 0,
                "last_signal": "", "last_signal_ts": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"last_side": None, "entry_price": None, "bars_held": 0,
                "last_signal": "", "last_signal_ts": 0}

def save_state(symbol: str, state: dict):
    path = state_path(symbol)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)

# ==============================
# FUNCIONES PRINCIPALES
# ==============================
def fetch_klines(symbol, interval, limit=500, retries=5, backoff=5):
    primary_url = "https://api.binance.us/api/v3/klines"
    backup_url  = "https://api.binance.com/api/v3/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(retries):
        try:
            r = requests.get(primary_url, params=params, headers=headers, timeout=10)
            if r.status_code in (418, 451) or not r.ok:
                print(f"⚠️ Endpoint US bloqueado (código {r.status_code}), probando Binance global...")
                r = requests.get(backup_url, params=params, headers=headers, timeout=10)

            r.raise_for_status()
            data = r.json()

            clean_data = [row[:12] for row in data if isinstance(row, list) and len(row) >= 12]

            df = pd.DataFrame(clean_data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
            ])

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["close"])
            return df

        except Exception as e:
            print(f"⚠️ Error Binance (intento {i+1}/{retries}): {e}", flush=True)
            time.sleep(backoff * (i + 1))

    raise RuntimeError("⚠️ Binance no responde tras varios intentos.")

def send_signal(symbol: str, code: str):
    state = load_state(symbol)
    now_ts = time.time()
    if state.get("last_signal") == code and (now_ts - state.get("last_signal_ts", 0)) < DUP_SIGNAL_COOLDOWN_SEC:
        return
    try:
        payload = {"code": code}
        r = requests.post(WUNDER_WEBHOOK, json=payload,
                          headers={"Content-Type": "application/json"}, timeout=10)
        print(f"[{datetime.now(UTC)}] {symbol} Signal -> {code} | status={r.status_code}", flush=True)
        state["last_signal"] = code
        state["last_signal_ts"] = now_ts
        save_state(symbol, state)
    except Exception as e:
        print(f"⚠️ Error enviando señal {symbol}: {e}", flush=True)

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
# KEEP ALIVE (Mantener Render activo)
# ==============================
def keep_alive():
    url = os.getenv("RENDER_URL", "https://wunder-trading-bot.onrender.com")
    while True:
        try:
            requests.get(url, timeout=10)
            print("🔁 Ping de mantenimiento enviado a Render", flush=True)
        except Exception as e:
            print("⚠️ Error al enviar ping:", e, flush=True)
        time.sleep(240)

# ==============================
# MONITOR PRINCIPAL
# ==============================
def main():
    global last_activity_time
    print("🚀 Bot de trading inteligente iniciado.")
    send_telegram_message("🤖 Bot activo y escuchando el mercado...")

    while True:
        try:
            last_activity_time = time.time()
            for SYMBOL in SYMBOLS:
                state = load_state(SYMBOL)
                df = compute_indicators(fetch_klines(SYMBOL, INTERVAL, 300))

                price = df["close"].iloc[-1]
                ema_f, ema_s = df["ema_fast"].iloc[-1], df["ema_slow"].iloc[-1]
                rsi = df["rsi"].iloc[-1]
                atr_now, atr_ma = df["atr"].iloc[-1], df["atr_ma"].iloc[-1]

                print(f"⏱️ {datetime.now(UTC)} | {SYMBOL} | P={price:.2f} | EMA9={ema_f:.2f} | EMA21={ema_s:.2f} | RSI={rsi:.1f} | ATR={atr_now:.2f} | Pos={state.get('last_side')}", flush=True)

                if in_range_zone(price, ema_f, ema_s) or not atr_active(atr_now, atr_ma):
                    print(f"⏸️ {SYMBOL} lateral/baja volatilidad. Sin operación.", flush=True)
                    print("💤 Esperando nueva oportunidad...", flush=True)
                else:
                    print("🟢 Condiciones listas para operar...", flush=True)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print("⚠️ Error general en main():", e, flush=True)
            send_telegram_message(f"⚠️ Error en main(): {e}")
            time.sleep(15)


# ==============================
# WATCHDOG
# ==============================
def watchdog():
    global last_activity_time
    last_activity_time = time.time()
    while True:
        if time.time() - last_activity_time > 300:
            print("⚠️ Watchdog: No hay actividad en 5 minutos, reiniciando bot...", flush=True)
            send_telegram_message("⚠️ Watchdog detectó inactividad. Reiniciando bot.")
            os.execv(sys.executable, ['python'] + sys.argv)
        time.sleep(60)


# ==============================
# EJECUCIÓN
# ==============================
if __name__ == "__main__":
    print("✅ Binance respondió correctamente, iniciando cálculos...", flush=True)
    test_telegram()
    threading.Thread(target=keep_alive, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    main()

