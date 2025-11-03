# ==========================
#  Etapa base
# ==========================
FROM python:3.11.9-slim

# Establece directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . .

# Instala Poetry (si usas pyproject.toml)
RUN pip install --no-cache-dir poetry

# Instala dependencias del proyecto
RUN if [ -f pyproject.toml ]; then \
        poetry install --no-root; \
    elif [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Comando por defecto al iniciar el contenedor
CMD ["poetry", "run", "python", "smart_trading_bot.py"]
