import os, time, json, requests, pandas as pd, ta
from datetime import datetime, UTC, timedelta
import http.server, socketserver, requests

# =====================================================
# SMART TRADING BOT v6 ULTIMATE (BTCUSDT) - Adaptativo
# =====================================================
# - Régimen de mercado (trend / range / explosive)
# - Multi-timeframe (15m base, 5m confirm, 1h macro)
# - ML-lite (clasificador de confluencia)
# - Drawdown diario/semanal + profit-lock
# - Auto-optimización semanal (crea/actualiza params.json)
# - Reload de parámetros en caliente
# - Riesgo adaptativo (régimen x racha x equity x profit-lock)
# - Trailing / Break-even / Parciales / Cooldown / Filtros ADX & ATR
# - Señales WunderTrading + Telegram

# ==============================
# CONFIG GENERAL
# ==============================
SYMBOLS = ["BTCUSDT"]  # Solo BTCUSDT
INTERVAL = "15m"  # timeframe base
CONFIRM_INTERVAL = "5m"  # confirmación táctica
CONFIRM_INTERVAL_MACRO = "1h"  # confirmación macro
WUNDER_WEBHOOK = "https://wtalerts.com/bot/custom"
POLL_SECONDS = 30
LOG_CSV = "trades_log.csv"
STATE_FILE_TPL = "state_{symbol}.json"
PARAMS_FILE = "params.json"
DUP_SIGNAL_COOLDOWN_SEC = 10

INITIAL_CAPITAL = 100.0
capital = INITIAL_CAPITAL

# ==============================
# ESTRATEGIA (parámetros base)
# ==============================
EMA_FAST, EMA_SLOW, EMA_LONG = 9, 21, 200
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 38, 70
RSI_SHORT_MIN, RSI_SHORT_MAX = 30, 60
EMA_DIFF_MARGIN = 0.0007

ATR_PERIOD = 14
ATR_SL_MULT = 1.6
ATR_TRAIL_MULT = 1.5
MIN_PROFIT_TO_CLOSE = 0.3  # %

# Break-even
BREAKEVEN_TRIGGER = 1.0  # %
BREAKEVEN_OFFSET = 0.15  # % (reservado para mejoras)

# ==============================
# COOLDOWN / SEGURIDAD
# ==============================
COOLDOWN_AFTER_EXIT_SEC = 300  # 5 min tras cualquier salida
MAX_CONSECUTIVE_LOSSES = 3  # autopausa tras N pérdidas seguidas
AUTO_PAUSE_SECONDS = 3600  # 1h

# ==============================
# RIESGO / VOLATILIDAD
# ==============================
RISK_PCT_BASE = 1.5  # % riesgo teórico base por operación
RISK_PCT_MIN, RISK_PCT_MAX = 0.5, 2.0
VOLATILITY_MULT_LIMIT = 1.6  # si ATR_f > ATR_MA * este múltiplo, no abrir
EQUITY_CURVE_LOOKBACK = 10  # trades para controlar curva de equity
EQUITY_DRAWDOWN_TH_PCT = -2.0  # si últimos N trades suman <-2%, bajar riesgo

# ==============================
# DRAWDOWN INSTITUCIONAL
# ==============================
DAILY_MAX_LOSS_PCT = 2.0  # límite de pérdidas diarias (% sumado del log)
WEEKLY_MAX_LOSS_PCT = 5.0  # límite de pérdidas semanales
PROFIT_LOCK_PCT = 2.0  # si en el día ya ganaste >=2% bloquea subir riesgo

# ==============================
# FILTROS AVANZADOS
# ==============================
ADX_MIN = 20  # fuerza mínima de tendencia
ATR_PCTL_WINDOW = 200
ATR_PCTL_THRESHOLD = 0.90
CANDLE_STRENGTH_MIN = 0.6
TRAIL_MIN_MOVE_ATR = 0.3
SKIP_UTC_HOURS = (0, 1, 2, 3)  # horas muertas
SKIP_WEEKENDS = False
PARTIAL_TAKE_PROFIT_PCT = 1.5
ENABLE_PARTIALS = True

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
SIGNAL_CODES = {
    "BTCUSDT": {
        "ENTER_LONG": "ENTER-LONG_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "ENTER_SHORT": "ENTER-SHORT_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
        "EXIT_ALL": "EXIT-ALL_Binance_BTCUSDT_BTC-BOT_5M_d99ea7958787b1d1fa0e0978",
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
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass


def test_telegram():
    send_telegram_message(
        "✅ Bot v6 ULTIMATE conectado con Telegram.\n"
        "Cargando parámetros y verificando data…"
    )


# ==============================
# ESTADO (persistencia por símbolo)
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
            "regime": None,
            "last_signal": None,
            "last_signal_ts": 0,
            # para auto-optimización semanal:
            "last_autoopt_date": None,  # YYYY-MM-DD ejecutado por última vez (UTC)
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
            # sane defaults
            s.setdefault("consecutive_losses", 0)
            s.setdefault("last_pnl_pct", None)
            s.setdefault("breakeven_active", False)
            s.setdefault("cooldown_until", 0)
            s.setdefault("sl_price", None)
            s.setdefault("partial_taken", False)
            s.setdefault("entry_snapshot", None)
            s.setdefault("regime", None)
            s.setdefault("last_signal", None)
            s.setdefault("last_signal_ts", 0)
            s.setdefault("last_autoopt_date", None)
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
            "regime": None,
            "last_signal": None,
            "last_signal_ts": 0,
            "last_autoopt_date": None,
        }


def save_state(symbol: str, state: dict):
    path = state_path(symbol)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# ==============================
# PARAM RELOAD (auto-optimización externa)
# ==============================
def maybe_reload_params():
    try:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                p = json.load(f)
            changed = []
            for k, v in p.items():
                if k in globals() and globals()[k] != v:
                    globals()[k] = v
                    changed.append((k, v))
            if changed:
                send_telegram_message(
                    "♻️ Parámetros recargados: "
                    + ", ".join(f"{k}={v}" for k, v in changed)
                )
    except Exception as e:
        print("⚠️ Error recargando params:", e, flush=True)


# ==============================
# DESCARGA DE DATOS
# ==============================
def fetch_klines(symbol, interval, limit=500, retries=5, backoff=5):
    """Descarga velas con fallback US->Global, user-agent y limpieza de datos."""
    primary_url = "https://api.binance.us/api/v3/klines"
    backup_url = "https://api.binance.com/api/v3/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(retries):
        try:
            r = requests.get(primary_url, params=params, headers=headers, timeout=10)
            if r.status_code in (418, 451) or not r.ok:
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
            print(
                f"⚠️ Error Binance {symbol} (intento {i+1}/{retries}): {e}", flush=True
            )
            time.sleep(backoff * (i + 1))

    raise RuntimeError(f"⚠️ Binance no responde para {symbol} tras varios intentos.")


# ==============================
# SEÑALES (WunderTrading)
# ==============================
def send_signal(symbol: str, code: str):
    state = load_state(symbol)
    now_ts = time.time()
    # anti-duplicados
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
# INDICADORES
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

    # ATR y su media (estable)
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


# ==============================
# RÉGIMEN DE MERCADO
# ==============================
def detect_regime(df):
    """Devuelve: 'trend', 'range', 'explosive' según ADX, ATR y pendiente EMA200."""
    adx_now = float(df["adx"].iloc[-1])
    atr_now = float(df["atr"].iloc[-1])
    atr_ma = float(df["atr_ma"].iloc[-1])
    atr_p90 = df["atr_p90"].iloc[-1]
    ema200_now = float(df["ema_long"].iloc[-1])
    ema200_prev = float(df["ema_long"].iloc[-5]) if len(df) > 5 else ema200_now
    ema200_slope = (ema200_now - ema200_prev) / max(1e-9, ema200_prev)

    explosive = False
    if not pd.isna(atr_p90) and atr_now > float(atr_p90):
        explosive = True
    if atr_now > atr_ma * (VOLATILITY_MULT_LIMIT * 1.1):
        explosive = True

    if explosive:
        return "explosive"
    if adx_now >= 30 and abs(ema200_slope) > 0.0005:
        return "trend"
    if adx_now < ADX_MIN:
        return "range"
    return "range"


# ==============================
# EQUITY CURVE CONTROL
# ==============================
def equity_curve_adjustment():
    """Lee los últimos trades del LOG_CSV y devuelve multiplicador de riesgo (0.6~1.1)."""
    try:
        if not os.path.exists(LOG_CSV):
            return 1.0
        df = pd.read_csv(LOG_CSV)
        if "profit_pct" not in df.columns:
            return 1.0
        last = df.tail(EQUITY_CURVE_LOOKBACK)
        perf = float(last["profit_pct"].fillna(0).sum())
        if perf <= EQUITY_DRAWDOWN_TH_PCT:
            return 0.7  # baja riesgo si rinde mal
        if perf > 3.0:
            return 1.1  # sube levemente si rinde muy bien
        return 1.0
    except Exception:
        return 1.0


# ==============================
# DRAWDOWN DIARIO/SEMANAL
# ==============================
def read_trades():
    try:
        return pd.read_csv(LOG_CSV)
    except:
        return pd.DataFrame()


def check_drawdown_limits():
    df = read_trades()
    if df.empty:
        return (False, False, False)
    try:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["profit_pct"] = pd.to_numeric(df["profit_pct"], errors="coerce")
        now = datetime.now(UTC)
        today = df[df["time"].dt.date == now.date()]
        week = df[df["time"] >= (now - pd.Timedelta(days=7))]
        day_sum = today["profit_pct"].sum()
        week_sum = week["profit_pct"].sum()
        profit_lock = day_sum >= PROFIT_LOCK_PCT
        day_limit = day_sum <= -DAILY_MAX_LOSS_PCT
        week_limit = week_sum <= -WEEKLY_MAX_LOSS_PCT
        return day_limit, week_limit, profit_lock
    except:
        return (False, False, False)


# ==============================
# LOG DE TRADES
# ==============================
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
        print("⚠️ Error al escribir LOG_CSV:", e, flush=True)


# ==============================
# ML-LITE (clasificador simple)
# ==============================
def ml_score(features):
    # pesos heurísticos
    w = {
        "rsi": 0.04,
        "rsi_slope": 2.0,
        "adx": 0.03,
        "ema_trend": 0.8,
        "vol_rel": 0.5,
        "regime_trend": 0.6,
    }
    score = 0.0
    score += w["rsi"] * (features["rsi"] - 50)
    score += w["rsi_slope"] * features["rsi_slope"]
    score += w["adx"] * (features["adx"] - 20)
    score += w["ema_trend"] * (1 if features["ema_trend"] else -1)
    score += w["vol_rel"] * features["vol_rel"]
    score += w["regime_trend"] * (
        1
        if features["regime"] == "trend"
        else (-0.5 if features["regime"] == "explosive" else 0)
    )
    return score


ML_THRESHOLD = 0.0  # si quieres ser más estricto, súbelo a 1.0


# ==============================
# AUTO-OPTIMIZACIÓN SEMANAL
# ==============================
def auto_optimize_params():
    """
    Analiza últimos 7 días de trades y ajusta:
      - RSI_LONG_MIN / RSI_LONG_MAX
      - ADX_MIN
      - RISK_PCT_BASE
    Reglas simples y conservadoras.
    """
    df = read_trades()
    now = datetime.now(UTC)
    week_df = df[
        pd.to_datetime(df.get("time", pd.Series([])), errors="coerce")
        >= (now - timedelta(days=7))
    ]
    if week_df.empty or "profit_pct" not in week_df.columns:
        return None  # nada que optimizar

    week_df["profit_pct"] = pd.to_numeric(
        week_df["profit_pct"], errors="coerce"
    ).fillna(0.0)
    n = len(week_df)
    wins = (week_df["profit_pct"] > 0).sum()
    winrate = wins / max(n, 1)
    avg_pnl = week_df["profit_pct"].mean()

    # Copiamos los valores actuales
    new_params = {
        "RSI_LONG_MIN": RSI_LONG_MIN,
        "RSI_LONG_MAX": RSI_LONG_MAX,
        "ADX_MIN": ADX_MIN,
        "RISK_PCT_BASE": RISK_PCT_BASE,
    }

    # Ajustes de RSI: si exceso de whipsaw (bajo winrate), endurecer filtros
    if winrate < 0.40 or avg_pnl < -0.5:
        # endurecer LONG: sube piso y baja techo
        new_params["RSI_LONG_MIN"] = min(max(RSI_LONG_MIN + 2, 30), 50)
        new_params["RSI_LONG_MAX"] = max(min(RSI_LONG_MAX - 2, 80), 55)
        # pedir más tendencia
        new_params["ADX_MIN"] = min(max(ADX_MIN + 2, 10), 35)
        # baja riesgo base
        new_params["RISK_PCT_BASE"] = float(
            max(RISK_PCT_MIN, round(RISK_PCT_BASE - 0.2, 2))
        )

    # Si está funcionando bien, relajar levemente (sin exceder límites)
    elif winrate > 0.55 and avg_pnl > 0.5:
        new_params["RSI_LONG_MIN"] = min(max(RSI_LONG_MIN - 1, 30), 50)
        new_params["RSI_LONG_MAX"] = max(min(RSI_LONG_MAX + 1, 80), 55)
        new_params["ADX_MIN"] = min(max(ADX_MIN - 1, 10), 35)
        new_params["RISK_PCT_BASE"] = float(
            min(RISK_PCT_MAX, round(RISK_PCT_BASE + 0.1, 2))
        )

    # Guardar params.json
    try:
        # mezclar con params existentes si los hay
        base = {}
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                base = json.load(f)
        base.update(new_params)
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False)
        return {
            "n": n,
            "winrate": round(winrate, 3),
            "avg": round(avg_pnl, 3),
            "new_params": new_params,
        }
    except Exception as e:
        print("⚠️ Error guardando params.json:", e, flush=True)
        return None


def is_sunday_utc(dt: datetime) -> bool:
    # Monday=0 ... Sunday=6
    return dt.weekday() == 6


def maybe_run_weekly_autoopt(symbol: str, state: dict):
    """
    Ejecuta la auto-optimización una sola vez cada domingo UTC.
    Guarda en el state la fecha (YYYY-MM-DD) de última ejecución.
    """
    now = datetime.now(UTC)
    if not is_sunday_utc(now):
        return state  # solo domingos

    today_str = now.date().isoformat()
    last_run = state.get("last_autoopt_date")
    if last_run == today_str:
        return state  # ya se ejecutó hoy

    result = auto_optimize_params()
    if result:
        msg = f"🧠 Auto-optimización semanal: trades={result['n']}, WinRate={result['winrate']*100:.1f}%, AvgPnL={result['avg']:.2f}%\n"
        msg += "Nuevos parámetros: " + ", ".join(
            [f"{k}={v}" for k, v in result["new_params"].items()]
        )
        send_telegram_message(msg)
        print(msg, flush=True)
    else:
        send_telegram_message(
            "🧠 Auto-optimización semanal: sin datos suficientes o sin cambios."
        )

    state["last_autoopt_date"] = today_str
    save_state(symbol, state)
    # recargar inmediatamente los nuevos parámetros si existen
    maybe_reload_params()
    return state


# ==============================
# MONITOR PRINCIPAL
# ==============================
def main():
    print(
        "🚀 Bot v6 ULTIMATE: régimen + multiTF + ML-lite + DD + riesgo adaptativo + auto-opt semanal.",
        flush=True,
    )
    test_telegram()
    send_telegram_message(
        "🤖 v6 ULTIMATE activo (15m/5m/1h). Auto-optimización semanal ON. Cargando params.json si existe…"
    )
    maybe_reload_params()

    consecutive_fetch_errors = 0

    while True:
        try:
            for SYMBOL in SYMBOLS:
                state = load_state(SYMBOL)

                # Auto-optimización semanal (solo domingo UTC, 1 vez por día)
                state = maybe_run_weekly_autoopt(SYMBOL, state)

                # Reload de params en caliente (por si editaste params.json a mano)
                maybe_reload_params()

                # Filtro horario
                now_utc = datetime.now(UTC)
                if SKIP_WEEKENDS and now_utc.weekday() >= 5:
                    print(f"📅 {SYMBOL} fin de semana, no operar.")
                    continue
                if now_utc.hour in SKIP_UTC_HOURS:
                    print(
                        f"🌙 {SYMBOL} horario muerto UTC {now_utc.hour:02d}, no operar."
                    )
                    continue

                # Datos multi-timeframe
                df15 = compute_indicators(fetch_klines(SYMBOL, INTERVAL, 300))
                df5 = compute_indicators(fetch_klines(SYMBOL, CONFIRM_INTERVAL, 300))
                df1h = compute_indicators(
                    fetch_klines(SYMBOL, CONFIRM_INTERVAL_MACRO, 300)
                )
                consecutive_fetch_errors = 0

                # Indicadores 15m
                price = float(df15["close"].iloc[-1])
                ema_f, ema_s = float(df15["ema_fast"].iloc[-1]), float(
                    df15["ema_slow"].iloc[-1]
                )
                ema_long = float(df15["ema_long"].iloc[-1])
                rsi = float(df15["rsi"].iloc[-1])
                rsi_slope_now = float(df15["rsi_slope"].iloc[-1])
                adx_now = float(df15["adx"].iloc[-1])
                atr_fast = float(df15["atr"].iloc[-1])
                atr_stable = float(df15["atr_ma"].iloc[-1])
                atr_p90_val = df15["atr_p90"].iloc[-1]
                atr_p90 = float(atr_p90_val) if not pd.isna(atr_p90_val) else None
                vol_now = float(df15["volume"].iloc[-1])
                vol_ma = float(df15["vol_ma"].iloc[-1])

                # 5m confluencia
                ema_f_5 = float(df5["ema_fast"].iloc[-1])
                ema_s_5 = float(df5["ema_slow"].iloc[-1])
                rsi_5 = float(df5["rsi"].iloc[-1])
                rsi_slope_5 = float(df5["rsi_slope"].iloc[-1])

                # 1h macro (solo EMA Fast/Slow)
                ema_f_1h = float(df1h["ema_fast"].iloc[-1])
                ema_s_1h = float(df1h["ema_slow"].iloc[-1])
                macro_ok = ema_f_1h > ema_s_1h and adx_now >= 20

                # Régimen
                regime = detect_regime(df15)
                state["regime"] = regime
                save_state(SYMBOL, state)

                side = state.get("last_side")
                entry = state.get("entry_price")
                trail = state.get("trail_price")
                breakeven_active = state.get("breakeven_active", False)
                sl_price = state.get("sl_price")

                print(
                    f"⏱️ {now_utc} | {SYMBOL} | P={price:.2f} | EMA9={ema_f:.2f} | EMA21={ema_s:.2f} | EMA200={ema_long:.2f} "
                    f"| RSI={rsi:.1f}/{rsi_5:.1f} (5m) | slope={rsi_slope_now:.3f}/{rsi_slope_5:.3f} (5m) | ADX={adx_now:.1f} "
                    f"| ATRf={atr_fast:.2f} | ATRma={atr_stable:.2f} | regime={regime} | Pos={side}",
                    flush=True,
                )

                # Autopausa por pérdidas consecutivas
                if state.get("cooldown_until", 0) > time.time():
                    remaining = int(state["cooldown_until"] - time.time())
                    print(f"⏸️ {SYMBOL} en cooldown {remaining}s restantes...")
                    continue

                last_close = float(df15["close"].iloc[-1])
                last_open = float(df15["open"].iloc[-1])

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
                            f"🛑 {SYMBOL} SL alcanzado ({side}) | PnL {profit_pct:.2f}%"
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
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # Trailing y Break-even (solo si el precio avanzó suficiente)
                    if abs(price - entry) >= atr_stable * TRAIL_MIN_MOVE_ATR:
                        if side == "LONG":
                            if profit_pct >= 3.0:
                                new_trail = max(
                                    trail or entry, price - atr_stable * 1.2
                                )
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
                            elif (
                                not breakeven_active and profit_pct >= BREAKEVEN_TRIGGER
                            ):
                                trail = entry
                                send_telegram_message(
                                    f"🟩 {SYMBOL} Break-even activado a {trail:.2f}"
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
                                        f"🏁 {SYMBOL} Trailing avanzado ajustado a {trail:.2f}"
                                    )
                            elif profit_pct >= 2.0:
                                new_trail = entry * 0.995
                                if not trail or new_trail < trail:
                                    trail = new_trail
                                    send_telegram_message(
                                        f"🟢 {SYMBOL} Ganancia asegurada +0.5%"
                                    )
                            elif (
                                not breakeven_active and profit_pct >= BREAKEVEN_TRIGGER
                            ):
                                trail = entry
                                send_telegram_message(
                                    f"🟩 {SYMBOL} Break-even activado a {trail:.2f}"
                                )
                                state["breakeven_active"] = True

                    # Parcial informativa (solo aviso)
                    if (
                        ENABLE_PARTIALS
                        and not state.get("partial_taken", False)
                        and profit_pct >= PARTIAL_TAKE_PROFIT_PCT
                    ):
                        code = SIGNAL_CODES[SYMBOL].get("TAKE_PROFIT_PARTIAL")
                        if code:
                            send_signal(SYMBOL, code)
                        send_telegram_message(
                            f"✂️ {SYMBOL} Parcial informativa al +{PARTIAL_TAKE_PROFIT_PCT:.1f}% ({side})."
                        )
                        state["partial_taken"] = True

                    # Cierre por trailing (ganancia mínima asegurada)
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
                                f"⏸️ Autopausa {SYMBOL} por {MAX_CONSECUTIVE_LOSSES} pérdidas seguidas ({AUTO_PAUSE_SECONDS//60} min)."
                            )
                        save_state(SYMBOL, state)
                        continue

                    # Cierres preventivos por debilidad/fuerza técnica
                    if side == "LONG" and (rsi < 38) and (price < entry * 0.985):
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        send_telegram_message(
                            f"🧯 {SYMBOL} Cierre preventivo (LONG) por debilidad RSI y -1.5%."
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
                            f"🧯 {SYMBOL} Cierre preventivo (SHORT) por fortaleza RSI y -1.5%."
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
                    continue  # fin gestión activa

                # =========================
                # SEÑALES DE ENTRADA
                # =========================

                # LÍMITES DE DRAWDOWN (día / semana) y profit-lock
                day_limit, week_limit, profit_lock = check_drawdown_limits()
                if day_limit or week_limit:
                    send_telegram_message(
                        f"⚠️ {SYMBOL} Drawdown límite alcanzado. Día={day_limit}, Semana={week_limit}. No abrir nuevas."
                    )
                    continue

                # Filtros de volatilidad/ADX
                if atr_p90 is not None and atr_fast > atr_p90:
                    print(
                        f"⚠️ {SYMBOL} ATR {atr_fast:.2f} > p{int(ATR_PCTL_THRESHOLD*100)} {atr_p90:.2f}, volatilidad extrema."
                    )
                    continue
                if atr_fast > atr_stable * VOLATILITY_MULT_LIMIT:
                    print(
                        f"⚠️ {SYMBOL} volatilidad alta (ATR {atr_fast:.2f} > {atr_stable:.2f}×{VOLATILITY_MULT_LIMIT}), no operar."
                    )
                    continue
                if adx_now < ADX_MIN:
                    print(f"⚠️ {SYMBOL} ADX {adx_now:.1f} < {ADX_MIN}, mercado lateral.")
                    continue

                # Cruces EMA (15m)
                ema_cross_up_now = ema_f > ema_s * (1 + EMA_DIFF_MARGIN)
                ema_cross_up_prev1 = (
                    df15["ema_fast"].iloc[-2] > df15["ema_slow"].iloc[-2]
                )
                ema_cross_up_prev2 = (
                    df15["ema_fast"].iloc[-3] > df15["ema_slow"].iloc[-3]
                )

                ema_cross_dn_now = ema_f < ema_s * (1 - EMA_DIFF_MARGIN)
                ema_cross_dn_prev1 = (
                    df15["ema_fast"].iloc[-2] < df15["ema_slow"].iloc[-2]
                )
                ema_cross_dn_prev2 = (
                    df15["ema_fast"].iloc[-3] < df15["ema_slow"].iloc[-3]
                )

                # Confluencia 5m
                long_align_5m = (
                    (ema_f_5 > ema_s_5) and (rsi_slope_5 > 0) and (rsi_5 >= 40)
                )
                short_align_5m = (
                    (ema_f_5 < ema_s_5) and (rsi_slope_5 < 0) and (rsi_5 <= 60)
                )

                # Señales base
                bullish_ok = (
                    ema_cross_up_now
                    and ema_cross_up_prev1
                    and ema_cross_up_prev2
                    and (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX)
                    and (rsi_slope_now > 0)
                    and (price > ema_long)
                    and (vol_now > vol_ma)
                    and long_align_5m
                )
                bearish_ok = (
                    ema_cross_dn_now
                    and ema_cross_dn_prev1
                    and ema_cross_dn_prev2
                    and (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX)
                    and (rsi_slope_now < 0)
                    and (price < ema_long)
                    and (vol_now > vol_ma)
                    and short_align_5m
                )

                # Fuerza de vela (price action básico)
                rng = max(df15["high"].iloc[-1] - df15["low"].iloc[-1], 1e-9)
                candle_strength = (last_close - last_open) / rng
                if bullish_ok and candle_strength < CANDLE_STRENGTH_MIN:
                    print(
                        f"⚠️ {SYMBOL} vela sin fuerza para LONG (strength {candle_strength:.2f} < {CANDLE_STRENGTH_MIN})."
                    )
                    bullish_ok = False
                if bearish_ok and candle_strength > -CANDLE_STRENGTH_MIN:
                    print(
                        f"⚠️ {SYMBOL} vela sin fuerza para SHORT (strength {candle_strength:.2f} > -{CANDLE_STRENGTH_MIN})."
                    )
                    bearish_ok = False

                # Ajustes por régimen
                if regime == "range":
                    if bullish_ok:
                        bullish_ok = candle_strength >= (CANDLE_STRENGTH_MIN + 0.1)
                    if bearish_ok:
                        bearish_ok = candle_strength <= (-(CANDLE_STRENGTH_MIN + 0.1))
                if regime == "explosive":
                    if bullish_ok:
                        bullish_ok = rsi <= 65
                    if bearish_ok:
                        bearish_ok = rsi >= 35

                # Macro confirmación 1h (tendencia macro a favor)
                if not macro_ok:
                    print(f"⚠️ {SYMBOL} macro 1h en contra. Saltando entrada.")
                    bullish_ok = bearish_ok = False

                # ML-lite final filter
                features = {
                    "rsi": rsi,
                    "rsi_slope": rsi_slope_now,
                    "adx": adx_now,
                    "ema_trend": ema_f > ema_s,
                    "vol_rel": (vol_now / max(vol_ma, 1e-9)) - 1.0,
                    "regime": regime,
                }
                score = ml_score(features)

                if not (bullish_ok or bearish_ok):
                    print(f"⏸️ {SYMBOL} sin señal clara (pre-ML).")
                    continue
                if score < ML_THRESHOLD:
                    print(f"⏸️ {SYMBOL} ML-score {score:.2f} < {ML_THRESHOLD:.2f}.")
                    continue

                side = "LONG" if bullish_ok else "SHORT"

                # Riesgo adaptativo
                losses = state.get("consecutive_losses", 0)
                regime_mult = (
                    1.0 if regime == "trend" else (0.8 if regime == "range" else 0.6)
                )
                streak_mult = 0.85**losses
                equity_mult = equity_curve_adjustment()
                if profit_lock:  # si ya ganaste el día, baja vela
                    equity_mult *= 0.6
                adj_risk_pct = max(
                    RISK_PCT_MIN,
                    min(
                        RISK_PCT_MAX,
                        RISK_PCT_BASE * regime_mult * streak_mult * equity_mult,
                    ),
                )

                risk_dollar = capital * (adj_risk_pct / 100.0)
                pos_size = risk_dollar / max(atr_stable * ATR_SL_MULT, 1e-9)
                size_info = f"risk={adj_risk_pct:.2f}%, size≈{pos_size:.4f} (u)"

                # Señal de entrada
                send_signal(SYMBOL, SIGNAL_CODES[SYMBOL][f"ENTER_{side}"])
                send_telegram_message(
                    f"🚀 {SYMBOL} Nueva entrada {side} a {price:.2f}\n"
                    f"Regime={regime} | RSI15/5={rsi:.1f}/{rsi_5:.1f} | ADX={adx_now:.1f} | ML={score:.2f}\n"
                    f"{size_info} | ATR(15m)={atr_stable:.2f} | MacroOK={macro_ok}"
                )

                # Inicialización de trailing y SL
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
                        "partial_taken": False,
                        "entry_snapshot": {
                            "time": now_utc.isoformat(),
                            "rsi": rsi,
                            "rsi_5": rsi_5,
                            "rsi_slope": rsi_slope_now,
                            "rsi_slope_5": rsi_slope_5,
                            "ema_fast": ema_f,
                            "ema_slow": ema_s,
                            "ema_long": ema_long,
                            "atr": atr_stable,
                            "adx": adx_now,
                            "volume": vol_now,
                            "vol_ma": vol_ma,
                            "regime": regime,
                        },
                    }
                )
                save_state(SYMBOL, state)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            consecutive_fetch_errors += 1
            if consecutive_fetch_errors in (3, 10, 20):
                send_telegram_message(
                    f"⚠️ Error repetido de datos ({consecutive_fetch_errors}): {e}"
                )
            print("⚠️ Error general:", e, flush=True)
            time.sleep(15)


class IPHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ip":
            ip = requests.get("https://ifconfig.me").text
            self.send_response(200)
            self.end_headers()
            self.wfile.write(ip.encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write("Bot running ✅".encode("utf-8"))


import threading

if __name__ == "__main__":
    print("Binance OK, iniciando v6 ULTIMATE...", flush=True)
    test_telegram()

    # Servidor HTTP para mostrar IP pública
    def start_http():
        with socketserver.TCPServer(("", 8080), IPHandler) as httpd:
            print("🌐 Servidor HTTP escuchando en puerto 8080 (/ip disponible)")
            httpd.serve_forever()

    threading.Thread(target=start_http, daemon=True).start()

    # Inicia el bot principal
    main()
