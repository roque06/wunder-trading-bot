import os, time, json, requests, pandas as pd, ta
from datetime import datetime, UTC

# ==============================
# CONFIG GENERAL
# ==============================
SYMBOLS = ["BTCUSDT"]
INTERVAL = "15m"  # timeframe recomendado
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
EMA_LONG = 200  # filtro macro
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

# Break-even
BREAKEVEN_TRIGGER = 1.0  # activa cuando gana +1%
BREAKEVEN_OFFSET = 0.15  # se mueve a entrada ±0.1%

# ==============================
# COOLDOWN CONFIG
# ==============================
COOLDOWN_AFTER_EXIT_SEC = 300  # 5 min

# ==============================
# RIESGO / VOLATILIDAD
# ==============================
RISK_PCT = 1.5  # % del capital a arriesgar por trade (teórico)
VOLATILITY_MULT_LIMIT = 1.6  # si ATR_f > ATR_MA * este múltiplo, no abrir
MAX_CONSECUTIVE_LOSSES = 3  # autopausa tras N pérdidas seguidas
AUTO_PAUSE_SECONDS = 3600  # 1h de pausa

# Mejoras nuevas
ADX_MIN = 20  # fuerza mínima de tendencia
ATR_PCTL_WINDOW = 200  # histórico para percentil de ATR
ATR_PCTL_THRESHOLD = 0.90  # si ATR actual > percentil 90, evitar
CANDLE_STRENGTH_MIN = 0.6  # fuerza mínima de vela para entrar
TRAIL_MIN_MOVE_ATR = 0.3  # no ajustar trailing si el avance < 0.3x ATR estable
SKIP_UTC_HOURS = (0, 1, 2, 3)  # horarios de baja liquidez (UTC)
SKIP_WEEKENDS = False  # pon True si no quieres operar fines de semana
PARTIAL_TAKE_PROFIT_PCT = 1.5  # % para salida parcial
ENABLE_PARTIALS = True  # activa la lógica de parciales (en señales solo informa)

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
SIGNAL_CODES = {
    "BTCUSDT": {
        "ENTER_LONG": "ENTER-LONG_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "ENTER_SHORT": "ENTER-SHORT_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "EXIT_ALL": "EXIT-ALL_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        # Opcional: si tu canal admite parciales, agrega una clave "TAKE_PROFIT_PARTIAL"
        # "TAKE_PROFIT_PARTIAL": "TP-PARTIAL_Binance_BTCUSDT_BTC-BOT_15M_xxx"
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
            print(f"Warning Telegram: {r.status_code}", flush=True)
    except Exception as e:
        print("Error enviando a Telegram:", e, flush=True)


def test_telegram():
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "Prueba de conexión Telegram exitosa.\\nEl bot ya puede enviarte notificaciones.",
        }
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("Prueba de Telegram enviada correctamente", flush=True)
        else:
            print(f"Telegram devolvió error: {r.status_code}", flush=True)
    except Exception as e:
        print("Error durante la prueba de Telegram:", e, flush=True)


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
            "sl_price": None,
            "partial_taken": False,
            "entry_snapshot": None,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
            s.setdefault("consecutive_losses", 0)
            s.setdefault("last_pnl_pct", None)
            s.setdefault("breakeven_active", False)
            s.setdefault("cooldown_until", 0)
            s.setdefault("sl_price", None)
            s.setdefault("partial_taken", False)
            s.setdefault("entry_snapshot", None)
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
            "sl_price": None,
            "partial_taken": False,
            "entry_snapshot": None,
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
    # Descarga velas con fallback US->Global, user-agent y limpieza de datos.
    primary_url = "https://api.binance.us/api/v3/klines"
    backup_url = "https://api.binance.com/api/v3/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(retries):
        try:
            r = requests.get(primary_url, params=params, headers=headers, timeout=10)
            if r.status_code in (418, 451) or not r.ok:
                print(
                    f"Endpoint US bloqueado ({r.status_code}), usando Binance global…",
                    flush=True,
                )
                r = requests.get(backup_url, params=params, headers=headers, timeout=10)

            r.raise_for_status()
            data = r.json()

            clean_data = [
                row[:12] for row in data if isinstance(row, list) and len(row) >= 12
            ]
            df = pd.DataFrame(
                clean_data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "qav",
                    "num_trades",
                    "taker_base",
                    "taker_quote",
                    "ignore",
                ],
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "taker_base",
                "taker_quote",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df.dropna(subset=["close"])
            return df

        except Exception as e:
            print(f"Error Binance {symbol} (intento {i+1}/{retries}): {e}", flush=True)
            time.sleep(backoff * (i + 1))

    raise RuntimeError(f"Binance no responde para {symbol} tras varios intentos.")


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
        print(f"Error enviando señal {symbol}: {e}", flush=True)


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
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=ATR_PERIOD
    )
    df["atr_ma"] = df["atr"].rolling(ATR_PERIOD).mean()

    # Percentil de ATR (riesgo extremo)
    df["atr_p90"] = df["atr"].rolling(ATR_PCTL_WINDOW).quantile(ATR_PCTL_THRESHOLD)

    # Volumen promedio 20 velas
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["vol_ma"] = df["volume"].rolling(20).mean()

    # ADX (fuerza de tendencia)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    return df


def log_trade(
    symbol, side, entry_price, exit_price, profit_pct, size_info=None, reason=""
):
    try:
        row = {
            "time": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_pct": profit_pct,
            "size_info": size_info if size_info is not None else "",
            "reason": reason,
        }
        pd.DataFrame([row]).to_csv(
            LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False
        )
    except Exception as e:
        print("Error al escribir LOG_CSV:", e, flush=True)


# ==============================
# MONITOR PRINCIPAL
# ==============================
def main():
    print(
        "Bot v3.2: señales robustas, ADX/ATR/Vol/RSI con pendiente, trailing estable, parciales y alertas."
    )
    send_telegram_message(
        "Bot v3.2 iniciado: 15m, ADX+horarios, percentil ATR, trailing/breakeven, riesgo adaptativo."
    )

    consecutive_fetch_errors = 0

    while True:
        try:
            for SYMBOL in SYMBOLS:
                state = load_state(SYMBOL)

                # Filtro horario
                now_utc = datetime.now(UTC)
                if SKIP_WEEKENDS and now_utc.weekday() >= 5:
                    print(f"{SYMBOL} fin de semana, no operar.")
                    continue
                if now_utc.hour in SKIP_UTC_HOURS:
                    print(f"{SYMBOL} horario muerto UTC {now_utc.hour:02d}, no operar.")
                    continue

                df_raw = fetch_klines(SYMBOL, INTERVAL, 300)
                consecutive_fetch_errors = 0  # reset si fue OK

                df = compute_indicators(df_raw)

                price = float(df["close"].iloc[-1])
                ema_f, ema_s = float(df["ema_fast"].iloc[-1]), float(
                    df["ema_slow"].iloc[-1]
                )
                ema_long = float(df["ema_long"].iloc[-1])
                rsi = float(df["rsi"].iloc[-1])
                rsi_slope_now = float(df["rsi_slope"].iloc[-1])
                adx_now = float(df["adx"].iloc[-1])

                # ATR rápido y ATR estable (media)
                atr_fast = float(df["atr"].iloc[-1])
                atr_stable = float(df["atr_ma"].iloc[-1])
                atr_p90 = (
                    float(df["atr_p90"].iloc[-1])
                    if not pd.isna(df["atr_p90"].iloc[-1])
                    else None
                )

                side = state.get("last_side")
                entry = state.get("entry_price")
                trail = state.get("trail_price")
                breakeven_active = state.get("breakeven_active", False)
                sl_price = state.get("sl_price")

                print(
                    f"{now_utc} | {SYMBOL} | P={price:.2f} | EMA9={ema_f:.2f} | EMA21={ema_s:.2f} "
                    f"| EMA200={ema_long:.2f} | RSI={rsi:.1f} | slope={rsi_slope_now:.3f} | ADX={adx_now:.1f} "
                    f"| ATRf={atr_fast:.2f} | ATRma={atr_stable:.2f} | Pos={side}",
                    flush=True,
                )

                # Autopausa por pérdidas consecutivas
                if state.get("cooldown_until", 0) > time.time():
                    remaining = int(state["cooldown_until"] - time.time())
                    print(f"{SYMBOL} en cooldown {remaining}s restantes...")
                    continue

                last_close = float(df["close"].iloc[-1])
                last_open = float(df["open"].iloc[-1])

                # =========================
                # GESTIÓN DE POSICIÓN ACTIVA
                # =========================
                if side and entry:
                    entry = float(entry)
                    profit_pct = (
                        (((price - entry) / entry) * 100)
                        if side == "LONG"
                        else (((entry - price) / entry) * 100)
                    )

                    # Stop-loss absoluto
                    if (side == "LONG" and sl_price and price <= sl_price) or (
                        side == "SHORT" and sl_price and price >= sl_price
                    ):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"{SYMBOL} SL alcanzado ({side}) | PnL {profit_pct:.2f}%"
                        )
                        log_trade(
                            SYMBOL, side, entry, price, profit_pct, reason="stop_loss"
                        )
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
                                "partial_taken": False,
                                "entry_snapshot": None,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # Trailing y Break-even (con filtro de movimiento mínimo)
                    if abs(price - entry) >= atr_stable * TRAIL_MIN_MOVE_ATR:
                        if side == "LONG":
                            if profit_pct >= 3.0:
                                new_trail = max(
                                    trail or entry, price - atr_stable * 1.2
                                )
                                if new_trail > (trail or 0):
                                    trail = new_trail
                                    send_telegram_message(
                                        f"{SYMBOL} Trailing avanzado ajustado a {trail:.2f}"
                                    )
                            elif profit_pct >= 2.0:
                                new_trail = entry * 1.005
                                if not trail or new_trail > trail:
                                    trail = new_trail
                                    send_telegram_message(
                                        f"{SYMBOL} Ganancia asegurada +0.5%"
                                    )
                            elif (
                                not breakeven_active and profit_pct >= BREAKEVEN_TRIGGER
                            ):
                                trail = entry
                                send_telegram_message(
                                    f"{SYMBOL} Break-even activado a {trail:.2f}"
                                )
                                state["breakeven_active"] = True
                        else:  # SHORT
                            if profit_pct >= 3.0:
                                new_trail = min(
                                    trail or entry, price + atr_stable * 1.2
                                )
                                if new_trail < (trail or 999999):
                                    trail = new_trail
                                    send_telegram_message(
                                        f"{SYMBOL} Trailing avanzado ajustado a {trail:.2f}"
                                    )
                            elif profit_pct >= 2.0:
                                new_trail = entry * 0.995
                                if not trail or new_trail < trail:
                                    trail = new_trail
                                    send_telegram_message(
                                        f"{SYMBOL} Ganancia asegurada +0.5%"
                                    )
                            elif (
                                not breakeven_active and profit_pct >= BREAKEVEN_TRIGGER
                            ):
                                trail = entry
                                send_telegram_message(
                                    f"{SYMBOL} Break-even activado a {trail:.2f}"
                                )
                                state["breakeven_active"] = True

                    # Parcial (informativa) al +1.5%
                    if (
                        ENABLE_PARTIALS
                        and not state.get("partial_taken", False)
                        and profit_pct >= PARTIAL_TAKE_PROFIT_PCT
                    ):
                        code = SIGNAL_CODES[SYMBOL].get("TAKE_PROFIT_PARTIAL")
                        if code:
                            send_signal(SYMBOL, code)
                        send_telegram_message(
                            f"{SYMBOL} Parcial informativa al +{PARTIAL_TAKE_PROFIT_PCT:.1f}% ({side})."
                        )
                        state["partial_taken"] = True

                    # Cierre por trailing
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
                            f"{SYMBOL} {side} cerrado | +{profit_pct:.2f}%"
                        )
                        reason = "trailing" if profit_pct >= 0 else "trailing_loss"
                        log_trade(SYMBOL, side, entry, price, profit_pct, reason=reason)
                        state.update(
                            {
                                "last_side": None,
                                "entry_price": None,
                                "trail_price": None,
                                "breakeven_active": False,
                                "last_pnl_pct": profit_pct,
                                "sl_price": None,
                                "consecutive_losses": (
                                    0
                                    if profit_pct >= 0
                                    else (state.get("consecutive_losses", 0) + 1)
                                ),
                                "cooldown_until": time.time() + COOLDOWN_AFTER_EXIT_SEC,
                                "partial_taken": False,
                                "entry_snapshot": None,
                            }
                        )
                        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                            state["cooldown_until"] = time.time() + AUTO_PAUSE_SECONDS
                            send_telegram_message(
                                f"Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # Cierres preventivos
                    if side == "LONG" and (rsi < 38) and (price < entry * 0.985):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"{SYMBOL} Cierre preventivo (LONG) por debilidad RSI y -1.5%."
                        )
                        log_trade(
                            SYMBOL,
                            side,
                            entry,
                            price,
                            profit_pct,
                            reason="preventive_long",
                        )
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
                                "partial_taken": False,
                                "entry_snapshot": None,
                            }
                        )
                        save_state(SYMBOL, state)
                        continue

                    if side == "SHORT" and (rsi > 62) and (price > entry * 1.015):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"{SYMBOL} Cierre preventivo (SHORT) por fortaleza RSI y -1.5%."
                        )
                        log_trade(
                            SYMBOL,
                            side,
                            entry,
                            price,
                            profit_pct,
                            reason="preventive_short",
                        )
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
                                "partial_taken": False,
                                "entry_snapshot": None,
                            }
                        )
                        save_state(SYMBOL, state)
                        continue

                    # Mantener trailing/estado
                    state["trail_price"] = trail
                    save_state(SYMBOL, state)
                    continue

                # =========================
                # SEÑALES DE ENTRADA
                # =========================
                # Confirmación de cruce EMA mantenido ≥2 velas
                ema_cross_up_now = ema_f > ema_s * (1 + EMA_DIFF_MARGIN)
                ema_cross_up_prev1 = df["ema_fast"].iloc[-2] > df["ema_slow"].iloc[-2]
                ema_cross_up_prev2 = df["ema_fast"].iloc[-3] > df["ema_slow"].iloc[-3]

                ema_cross_dn_now = ema_f < ema_s * (1 - EMA_DIFF_MARGIN)
                ema_cross_dn_prev1 = df["ema_fast"].iloc[-2] < df["ema_slow"].iloc[-2]
                ema_cross_dn_prev2 = df["ema_fast"].iloc[-3] < df["ema_slow"].iloc[-3]

                vol_now = float(df["volume"].iloc[-1])
                vol_ma = float(df["vol_ma"].iloc[-1])

                # Filtros extra: ADX y percentil de ATR
                if adx_now < ADX_MIN:
                    print(f"{SYMBOL} ADX {adx_now:.1f} < {ADX_MIN}, mercado lateral.")
                    continue
                if atr_p90 is not None and atr_fast > atr_p90:
                    print(
                        f"{SYMBOL} ATR {atr_fast:.2f} > p{int(ATR_PCTL_THRESHOLD*100)} {atr_p90:.2f}, volatilidad extrema."
                    )
                    continue
                if atr_fast > atr_stable * VOLATILITY_MULT_LIMIT:
                    print(
                        f"{SYMBOL} volatilidad alta (ATR {atr_fast:.2f} > {atr_stable:.2f}x{VOLATILITY_MULT_LIMIT}), no operar."
                    )
                    continue

                # Señales
                bullish_ok = (
                    ema_cross_up_now
                    and ema_cross_up_prev1
                    and ema_cross_up_prev2
                    and (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX)
                    and (rsi_slope_now > 0)
                    and (price > ema_long)
                    and (vol_now > vol_ma)
                )

                bearish_ok = (
                    ema_cross_dn_now
                    and ema_cross_dn_prev1
                    and ema_cross_dn_prev2
                    and (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX)
                    and (rsi_slope_now < 0)
                    and (price < ema_long)
                    and (vol_now > vol_ma)
                )

                if not (bullish_ok or bearish_ok):
                    print(f"{SYMBOL} sin señal clara.")
                    continue

                side = "LONG" if bullish_ok else "SHORT"

                # Fuerza de vela (price action)
                rng = max(df["high"].iloc[-1] - df["low"].iloc[-1], 1e-9)
                candle_strength = (last_close - last_open) / rng
                if side == "LONG" and candle_strength < CANDLE_STRENGTH_MIN:
                    print(
                        f"{SYMBOL} vela sin fuerza para LONG (strength {candle_strength:.2f} < {CANDLE_STRENGTH_MIN})."
                    )
                    continue
                if side == "SHORT" and candle_strength > -CANDLE_STRENGTH_MIN:
                    print(
                        f"{SYMBOL} vela sin fuerza para SHORT (strength {candle_strength:.2f} > -{CANDLE_STRENGTH_MIN})."
                    )
                    continue

                # Riesgo adaptativo según racha
                losses = state.get("consecutive_losses", 0)
                adj_risk_pct = max(0.5, RISK_PCT * (1 - 0.2 * losses))

                # Tamaño por riesgo teórico usando ATR estable
                risk_dollar = capital * (adj_risk_pct / 100)
                pos_size = risk_dollar / max(atr_stable * ATR_SL_MULT, 1e-9)
                size_info = f"risk={adj_risk_pct:.2f}%, size≈{pos_size:.4f} (u)"

                # Señal de entrada
                send_signal(SYMBOL, SIGNAL_CODES[SYMBOL][f"ENTER_{side}"])
                send_telegram_message(
                    f"🚀 {SYMBOL} Nueva entrada {side} a {price:.2f}\n"
                    f"RSI={rsi:.1f} slope={rsi_slope_now:.3f} ADX={adx_now:.1f} Vol>{'sí' if vol_now>vol_ma else 'no'}\n"
                    f"{size_info} | ATR={atr_stable:.2f}"
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

                # Guardar snapshot de entrada para logs
                state.update(
                    {
                        "last_side": side,
                        "entry_price": price,
                        "trail_price": trail_init,
                        "cooldown_until": 0,
                        "breakeven_active": False,
                        "sl_price": sl_init,
                        "partial_taken": False,
                        "entry_snapshot": {
                            "time": now_utc.isoformat(),
                            "rsi": rsi,
                            "rsi_slope": rsi_slope_now,
                            "ema_fast": ema_f,
                            "ema_slow": ema_s,
                            "ema_long": ema_long,
                            "atr": atr_stable,
                            "adx": adx_now,
                            "volume": vol_now,
                            "vol_ma": vol_ma,
                        },
                    }
                )
                save_state(SYMBOL, state)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            consecutive_fetch_errors += 1
            if consecutive_fetch_errors in (3, 10, 20):
                send_telegram_message(
                    f"Error repetido de datos ({consecutive_fetch_errors}): {e}"
                )
            print("Error general:", e, flush=True)
            time.sleep(15)


if __name__ == "__main__":
    print("Binance respondió correctamente, iniciando cálculos...", flush=True)
    test_telegram()
    main()
