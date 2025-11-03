# =========================================
#  DOCKERFILE PARA WUNDER-TRADING-BOT
#  Python 3.11.9 + Poetry 2.1.3
# =========================================
FROM python:3.11.9-slim

# Evita prompts interactivos y optimiza compilación
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/root/.local/bin:$PATH"

# Crea directorio de trabajo
WORKDIR /app

# Copia archivos del proyecto
COPY . .

# Instala dependencias necesarias y Poetry
RUN apt-get update && apt-get install -y curl build-essential && \
    pip install --no-cache-dir poetry==$POETRY_VERSION && \
    poetry install --no-root && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# =========================================
# Ejecuta el bot + abre un puerto ficticio para Render Free
# =========================================
CMD bash -c "poetry run python smart_trading_bot.py & python -m http.server 8080"
