import os, time, json, requests, pandas as pd, ta
from datetime import datetime, UTC

# ==============================
# CONFIG GENERAL
# ==============================
SYMBOLS = ["BTCUSDT"]
INTERVAL = "15m"  # ✅ timeframe recomendado
WUNDER_WEBHOOK = "https://wtalerts.com/bot/custom"
POLL_SECONDS = 30
LOG_CSV = "trades_log.csv"
STATE_FILE_TPL = "state_{symbol}.json"
MIN_HOLD_BARS = 3
DUP_SIGNAL_COOLDOWN_SEC = 10

INITIAL_CAPITAL = 100.0
capital = INITIAL_CAPITAL

# ==============================
# ESTRATEGIA
# ==============================
EMA_FAST, EMA_SLOW = 9, 21
EMA_LONG = 200  # ✅ filtro macro
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 38, 70
RSI_SHORT_MIN, RSI_SHORT_MAX = 30, 60
EMA_DIFF_MARGIN = 0.0007

ATR_PERIOD = 14
ATR_MULT_RANGE_BLOCK = 0.05
ATR_ACTIVE_FACTOR = 0.8
ATR_SL_MULT = 1.4
ATR_TP_MULT = 2.0
ATR_TP_BOOST = 2.5
ATR_TRAIL_MULT = 1.5
MIN_PROFIT_TO_CLOSE = 0.3

# 🔹 Break-even
BREAKEVEN_TRIGGER = 1.0  # activa cuando gana +1%
BREAKEVEN_OFFSET = 0.15  # se mueve a entrada ±0.1%

# ==============================
# COOLDOWN CONFIG
# ==============================
COOLDOWN_AFTER_EXIT_SEC = 300  # 5 min
RSI_COOLDOWN_LONG = 40
RSI_COOLDOWN_SHORT = 55

# ==============================
# RIESGO / VOLATILIDAD (🆕)
# ==============================
RISK_PCT = 1.5  # % del capital a arriesgar por trade (teórico)
VOLATILITY_MULT_LIMIT = 1.6  # si ATR_f > ATR_MA * este múltiplo, no abrir
MAX_CONSECUTIVE_LOSSES = 3  # autopausa tras N pérdidas seguidas
AUTO_PAUSE_SECONDS = 3600  # 1h de pausa

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
SIGNAL_CODES = {
    "BTCUSDT": {
        # Nota: el sufijo "5M" es solo texto; si tu canal de Wunder depende del timeframe,
        # cámbialo a "15M" para mantener coherencia visual.
        "ENTER_LONG": "ENTER-LONG_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "ENTER_SHORT": "ENTER-SHORT_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "EXIT_ALL": "EXIT-ALL_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
    }
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
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
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
        }
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("📩 Prueba de Telegram enviada correctamente ✅", flush=True)
        else:
            print(f"⚠️ Telegram devolvió error: {r.status_code}", flush=True)
    except Exception as e:
        print("❌ Error durante la prueba de Telegram:", e, flush=True)


# ==============================
# ESTADO
# ==============================
def state_path(symbol: str) -> str:
    return STATE_FILE_TPL.format(symbol=symbol)


def load_state(symbol: str):
    path = state_path(symbol)
    if not os.path.exists(path):
        return {
            "last_side": None,
            "entry_price": None,
            "trail_price": None,
            "cooldown_until": 0,
            "breakeven_active": False,
            "consecutive_losses": 0,
            "last_pnl_pct": None,
            "sl_price": None,  # 🆕 stop-loss absoluto por ATR
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
            s.setdefault("consecutive_losses", 0)
            s.setdefault("last_pnl_pct", None)
            s.setdefault("breakeven_active", False)
            s.setdefault("cooldown_until", 0)
            s.setdefault("sl_price", None)  # 🆕
            return s
    except:
        return {
            "last_side": None,
            "entry_price": None,
            "trail_price": None,
            "cooldown_until": 0,
            "breakeven_active": False,
            "consecutive_losses": 0,
            "last_pnl_pct": None,
            "sl_price": None,  # 🆕
        }


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


def send_signal(symbol: str, code: str):
    state = load_state(symbol)
    now_ts = time.time()
    if (
        state.get("last_signal") == code
        and (now_ts - state.get("last_signal_ts", 0)) < DUP_SIGNAL_COOLDOWN_SEC
    ):
        return
    try:
        r = requests.post(WUNDER_WEBHOOK, json={"code": code}, timeout=10)
        print(
            f"[{datetime.now(UTC)}] {symbol} Signal -> {code} | status={r.status_code}",
            flush=True,
        )
        state["last_signal"] = code
        state["last_signal_ts"] = now_ts
        save_state(symbol, state)
    except Exception as e:
        print(f"⚠️ Error enviando señal {symbol}: {e}", flush=True)


# ==============================
# INDICADORES (versión avanzada)
# ==============================
def compute_indicators(df):
    # EMAs
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["ema_long"] = ta.trend.ema_indicator(df["close"], window=EMA_LONG)

    # RSI base + suavizado y pendiente
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["rsi_smooth"] = ta.trend.ema_indicator(df["rsi"], window=5)
    df["rsi_slope"] = df["rsi_smooth"].diff()

    # ATR 15m y su media (estable)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=ATR_PERIOD)
    df["atr_ma"] = df["atr"].rolling(ATR_PERIOD).mean()

    # Volumen promedio 20 velas
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["vol_ma"] = df["volume"].rolling(20).mean()

    return df


def log_trade(symbol, side, entry_price, exit_price, profit_pct, size_info=None):
    try:
        row = {
            "time": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_pct": profit_pct,
            "size_info": size_info if size_info is not None else "",
        }
        pd.DataFrame([row]).to_csv(
            LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False
        )
    except Exception as e:
        print("⚠️ Error al escribir LOG_CSV:", e, flush=True)


# ==============================
# MONITOR PRINCIPAL
# ==============================
def main():
    print("🚀 Bot con salida inteligente, TP dinámico, trailing y break-even iniciado.")
    send_telegram_message(
        "🤖 Bot activo con trailing, break-even y TP dinámico habilitados."
    )

    while True:
        try:
            for SYMBOL in SYMBOLS:
                state = load_state(SYMBOL)

                df_raw = fetch_klines(SYMBOL, INTERVAL, 300)
                df = compute_indicators(df_raw)

                price = float(df["close"].iloc[-1])
                ema_f, ema_s = float(df["ema_fast"].iloc[-1]), float(df["ema_slow"].iloc[-1])
                ema_long = float(df["ema_long"].iloc[-1])
                rsi = float(df["rsi"].iloc[-1])

                # ATR rápido y ATR estable (media)
                atr_fast = float(df["atr"].iloc[-1])
                atr_stable = float(df["atr_ma"].iloc[-1])

                side = state.get("last_side")
                entry = state.get("entry_price")
                trail = state.get("trail_price")
                breakeven_active = state.get("breakeven_active", False)
                sl_price = state.get("sl_price")

                print(
                    f"⏱️ {datetime.now(UTC)} | {SYMBOL} | P={price:.2f} | EMA9={ema_f:.2f} | EMA21={ema_s:.2f} | EMA200={ema_long:.2f} | RSI={rsi:.1f} | ATRf={atr_fast:.2f} | ATRma={atr_stable:.2f} | Pos={side}",
                    flush=True,
                )

                # 🧱 Autopausa por pérdidas consecutivas (🆕)
                if state.get("cooldown_until", 0) > time.time():
                    remaining = int(state["cooldown_until"] - time.time())
                    print(f"⏸️ {SYMBOL} en cooldown {remaining}s restantes...")
                    continue

                last_close = float(df["close"].iloc[-1])
                prev_close = float(df["close"].iloc[-2])
                last_open = float(df["open"].iloc[-1])

                # =========================
                # GESTIÓN DE POSICIÓN ACTIVA
                # =========================
                if side and entry:
                    entry = float(entry)
                    profit_pct = (
                        ((price - entry) / entry * 100)
                        if side == "LONG"
                        else ((entry - price) / entry * 100)
                    )

                    # 🛡️ Stop-loss absoluto por ATR (estable)
                    if side == "LONG" and sl_price and price <= sl_price:
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"🛑 {SYMBOL} SL alcanzado (LONG) | PnL {profit_pct:.2f}%"
                        )
                        log_trade(SYMBOL, side, entry, price, profit_pct)
                        state.update(
                            {
                                "last_side": None,
                                "entry_price": None,
                                "trail_price": None,
                                "breakeven_active": False,
                                "last_pnl_pct": profit_pct,
                                "sl_price": None,
                                "consecutive_losses": (
                                    (state.get("consecutive_losses", 0) + 1)
                                    if profit_pct < 0
                                    else 0
                                ),
                                "cooldown_until": time.time() + COOLDOWN_AFTER_EXIT_SEC,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    if side == "SHORT" and sl_price and price >= sl_price:
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"🛑 {SYMBOL} SL alcanzado (SHORT) | PnL {profit_pct:.2f}%"
                        )
                        log_trade(SYMBOL, side, entry, price, profit_pct)
                        state.update(
                            {
                                "last_side": None,
                                "entry_price": None,
                                "trail_price": None,
                                "breakeven_active": False,
                                "last_pnl_pct": profit_pct,
                                "sl_price": None,
                                "consecutive_losses": (
                                    (state.get("consecutive_losses", 0) + 1)
                                    if profit_pct < 0
                                    else 0
                                ),
                                "cooldown_until": time.time() + COOLDOWN_AFTER_EXIT_SEC,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # 🧠 Protección dinámica de ganancias (trailing + breakeven)
                    if side == "LONG":
                        if profit_pct >= 3.0:
                            new_trail = max(trail or entry, price - atr_stable * 1.2)
                            if new_trail > (trail or 0):
                                trail = new_trail
                                send_telegram_message(
                                    f"🏁 {SYMBOL} Trailing avanzado ajustado a {trail:.2f}"
                                )
                        elif profit_pct >= 2.0:
                            new_trail = entry * 1.005
                            if not trail or new_trail > trail:
                                trail = new_trail
                                send_telegram_message(
                                    f"🟢 {SYMBOL} Ganancia asegurada +0.5%"
                                )
                        elif not breakeven_active and profit_pct >= BREAKEVEN_TRIGGER:
                            trail = entry
                            send_telegram_message(
                                f"🟩 {SYMBOL} Break-even activado a {trail:.2f}"
                            )
                            state["breakeven_active"] = True

                    elif side == "SHORT":
                        if profit_pct >= 3.0:
                            new_trail = min(trail or entry, price + atr_stable * 1.2)
                            if new_trail < (trail or 999999):
                                trail = new_trail
                                send_telegram_message(
                                    f"🏁 {SYMBOL} Trailing avanzado ajustado a {trail:.2f}"
                                )
                        elif profit_pct >= 2.0:
                            new_trail = entry * 0.995
                            if not trail or new_trail < trail:
                                trail = new_trail
                                send_telegram_message(
                                    f"🟢 {SYMBOL} Ganancia asegurada +0.5%"
                                )
                        elif not breakeven_active and profit_pct >= BREAKEVEN_TRIGGER:
                            trail = entry
                            send_telegram_message(
                                f"🟩 {SYMBOL} Break-even activado a {trail:.2f}"
                            )
                            state["breakeven_active"] = True

                    # 🔹 Cierre inteligente por trailing en ganancia
                    if (
                        side == "LONG"
                        and trail
                        and price <= trail
                        and profit_pct >= MIN_PROFIT_TO_CLOSE
                    ) or (
                        side == "SHORT"
                        and trail
                        and price >= trail
                        and profit_pct >= MIN_PROFIT_TO_CLOSE
                    ):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"💰 {SYMBOL} {side} cerrado | +{profit_pct:.2f}% ✅"
                        )
                        log_trade(SYMBOL, side, entry, price, profit_pct)
                        state.update(
                            {
                                "last_side": None,
                                "entry_price": None,
                                "trail_price": None,
                                "breakeven_active": False,
                                "last_pnl_pct": profit_pct,
                                "sl_price": None,  # 🆕
                                "consecutive_losses": (
                                    0
                                    if profit_pct >= 0
                                    else (state.get("consecutive_losses", 0) + 1)
                                ),
                                "cooldown_until": time.time() + COOLDOWN_AFTER_EXIT_SEC,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # 🆕 Salida anticipada de seguridad (pérdida técnica sin cruce)
                    if side == "LONG" and (rsi < 38) and (price < entry * 0.985):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"🧯 {SYMBOL} Cierre preventivo (LONG) por debilidad RSI y -1.5%."
                        )
                        log_trade(SYMBOL, side, entry, price, profit_pct)
                        state.update(
                            {
                                "last_side": None,
                                "entry_price": None,
                                "trail_price": None,
                                "breakeven_active": False,
                                "last_pnl_pct": profit_pct,
                                "sl_price": None,
                                "consecutive_losses": (
                                    (state.get("consecutive_losses", 0) + 1)
                                    if profit_pct < 0
                                    else 0
                                ),
                                "cooldown_until": time.time() + COOLDOWN_AFTER_EXIT_SEC,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    if side == "SHORT" and (rsi > 62) and (price > entry * 1.015):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"🧯 {SYMBOL} Cierre preventivo (SHORT) por fortaleza RSI y -1.5%."
                        )
                        log_trade(SYMBOL, side, entry, price, profit_pct)
                        state.update(
                            {
                                "last_side": None,
                                "entry_price": None,
                                "trail_price": None,
                                "breakeven_active": False,
                                "last_pnl_pct": profit_pct,
                                "sl_price": None,
                                "consecutive_losses": (
                                    (state.get("consecutive_losses", 0) + 1)
                                    if profit_pct < 0
                                    else 0
                                ),
                                "cooldown_until": time.time() + COOLDOWN_AFTER_EXIT_SEC,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # Mantener trailing/estado actualizado
                    state["trail_price"] = trail
                    save_state(SYMBOL, state)
                    continue

                # =========================
                # SEÑALES DE ENTRADA
                # =========================
                # Confirmación de cruce EMA mantenido ≥2 velas
                ema_cross_up_now  = ema_f > ema_s * (1 + EMA_DIFF_MARGIN)
                ema_cross_up_prev1 = df["ema_fast"].iloc[-2] > df["ema_slow"].iloc[-2]
                ema_cross_up_prev2 = df["ema_fast"].iloc[-3] > df["ema_slow"].iloc[-3]

                ema_cross_dn_now  = ema_f < ema_s * (1 - EMA_DIFF_MARGIN)
                ema_cross_dn_prev1 = df["ema_fast"].iloc[-2] < df["ema_slow"].iloc[-2]
                ema_cross_dn_prev2 = df["ema_fast"].iloc[-3] < df["ema_slow"].iloc[-3]

                rsi_slope_now = float(df["rsi_slope"].iloc[-1])
                vol_now = float(df["volume"].iloc[-1])
                vol_ma  = float(df["vol_ma"].iloc[-1])

                bullish_ok = (
                    ema_cross_up_now
                    and ema_cross_up_prev1
                    and ema_cross_up_prev2
                    and (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX)
                    and (rsi_slope_now > 0)            # pendiente RSI positiva
                    and (price > ema_long)
                    and (vol_now > vol_ma)             # confirmación de volumen
                )

                bearish_ok = (
                    ema_cross_dn_now
                    and ema_cross_dn_prev1
                    and ema_cross_dn_prev2
                    and (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX)
                    and (rsi_slope_now < 0)            # pendiente RSI negativa
                    and (price < ema_long)
                    and (vol_now > vol_ma)
                )

                # =========================
                # FILTRO DE VOLATILIDAD
                # =========================
                if atr_fast > atr_stable * VOLATILITY_MULT_LIMIT:
                    print(
                        f"⚠️ {SYMBOL} volatilidad alta (ATR {atr_fast:.2f} > {atr_stable:.2f}×{VOLATILITY_MULT_LIMIT}), no operar."
                    )
                    continue

                if bullish_ok:
                    side = "LONG"
                elif bearish_ok:
                    side = "SHORT"
                else:
                    print(f"⏸️ {SYMBOL} sin señal clara.")
                    continue

                # Tamaño por riesgo teórico usando ATR estable
                risk_dollar = capital * (RISK_PCT / 100)
                pos_size = risk_dollar / max(atr_stable * ATR_SL_MULT, 1e-9)
                size_info = f"risk={RISK_PCT:.2f}%, size≈{pos_size:.4f} (u)"

                # Señal de entrada
                send_signal(SYMBOL, SIGNAL_CODES[SYMBOL][f"ENTER_{side}"])
                send_telegram_message(
                    f"🚀 {SYMBOL} Nueva entrada {side} a {price:.2f}\n<size: {size_info}>"
                )

                # Inicialización de trailing y SL con ATR estable
                trail_init = (
                    (price - atr_stable * ATR_TRAIL_MULT)
                    if side == "LONG"
                    else (price + atr_stable * ATR_TRAIL_MULT)
                )
                sl_init = (
                    (price - atr_stable * ATR_SL_MULT)
                    if side == "LONG"
                    else (price + atr_stable * ATR_SL_MULT)
                )

                state.update(
                    {
                        "last_side": side,
                        "entry_price": price,
                        "trail_price": trail_init,
                        "cooldown_until": 0,
                        "breakeven_active": False,
                        "sl_price": sl_init,
                    }
                )
                save_state(SYMBOL, state)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print("⚠️ Error general:", e, flush=True)
            time.sleep(15)


if __name__ == "__main__":
    print("✅ Binance respondió correctamente, iniciando cálculos...", flush=True)
    test_telegram()
    main()
