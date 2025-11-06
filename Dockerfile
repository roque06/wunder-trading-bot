# =========================================
#  DOCKERFILE PARA WUNDER-TRADING-BOT v6
#  Python 3.11 + dependencias directas
# =========================================
FROM python:3.11-slim

# Evita prompts interactivos y optimiza compilación
ENV PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Crea directorio de trabajo
WORKDIR /app

# Copia archivos del proyecto
COPY . .

# Instala dependencias necesarias directamente (sin Poetry)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    pip install --no-cache-dir pandas numpy requests ta && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# =========================================
# Ejecuta el bot + abre un puerto ficticio para Render Free
# =========================================
CMD bash -c "poetry run python smart_trading_bot.py & python -m http.server 8080"
