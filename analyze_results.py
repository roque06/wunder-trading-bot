import os
import time
import pandas as pd
import requests
from datetime import datetime, UTC

# ==============================
# CONFIG TELEGRAM
# ==============================
TELEGRAM_TOKEN = "7543685147:AAHGtjiy-wAQ7uIMTsuahx57MQ-8veDgcls"
TELEGRAM_CHAT_ID = "5324968421"
LOG_CSV = "trades_log.csv"

# ==============================
# FUNCIONES TELEGRAM
# ==============================
def send_telegram_message(text: str):
    """Envía mensajes a Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("❌ Error enviando a Telegram:", e)

# ==============================
# ANÁLISIS DIARIO
# ==============================
def analyze_trades():
    """Envía un resumen diario al Telegram."""
    try:
        if not os.path.exists(LOG_CSV):
            return
        df = pd.read_csv(LOG_CSV)
        if df.empty or "price" not in df.columns:
            return

        df = df[df["event"].str.contains("ENTER|EXIT", na=False)]
        if df.empty:
            return

        df["profit"] = df["price"].diff()
        df = df.dropna()

        wins = df[df["profit"] > 0]["profit"]
        losses = df[df["profit"] < 0]["profit"]

        win_rate = len(wins) / len(df) * 100 if len(df) else 0
        profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else float("inf")
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
    except Exception as e:
        print("⚠️ Error analizando trades:", e)

# ==============================
# MONITOR DE NUEVAS OPERACIONES
# ==============================
def watch_trades():
    """Monitorea trades_log.csv en tiempo real y notifica nuevos eventos."""
    print("👁️‍🗨️ Monitoreando operaciones en tiempo real...")
    send_telegram_message("📡 Monitor de operaciones iniciado...")

    last_size = 0
    last_analysis_date = datetime.now(UTC).date()

    while True:
        try:
            if not os.path.exists(LOG_CSV):
                time.sleep(5)
                continue

            # Verifica cambios de tamaño del archivo
            size_now = os.path.getsize(LOG_CSV)
            if size_now != last_size and size_now > 0:
                df = pd.read_csv(LOG_CSV)
                if not df.empty:
                    last_row = df.iloc[-1]
                    event = str(last_row.get("event", ""))
                    price = last_row.get("price", 0)
                    ts = last_row.get("ts_utc", datetime.now(UTC))
                    if "ENTER" in event:
                        msg = f"🟢 <b>Entrada detectada:</b> {event}\n💰 Precio: {price}\n🕒 {ts}"
                        send_telegram_message(msg)
                        print(msg)
                    elif "EXIT" in event:
                        msg = f"⚪ <b>Salida detectada:</b> {event}\n💰 Precio: {price}\n🕒 {ts}"
                        send_telegram_message(msg)
                        print(msg)
                last_size = size_now

            # Envío de reporte diario (una vez por día)
            today = datetime.now(UTC).date()
            if today != last_analysis_date:
                analyze_trades()
                last_analysis_date = today

            time.sleep(10)  # revisa cada 10 segundos
        except Exception as e:
            print("⚠️ Error en monitor:", e)
            time.sleep(15)

# ==============================
# EJECUCIÓN
# ==============================
if __name__ == "__main__":
    watch_trades()
