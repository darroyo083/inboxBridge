# InboxBridge — reproducible build image
# Fija versión de Python y build reproducible (digest del índice multi-arch).
FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (caché de capas)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Usuario no root + directorio de datos pre-creado (volume en compose)
RUN useradd --create-home --uid 10001 inboxbridge && mkdir -p /app/data /app/credentials \
    && chown -R inboxbridge:inboxbridge /app/data /app/credentials
USER inboxbridge

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "inboxbridge.healthcheck"]

CMD ["python", "-m", "inboxbridge.app"]
