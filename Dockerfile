# InboxBridge — reproducible build image
# Fija versión de Python y build reproducible.
FROM python:3.12-slim@sha256:48b4dc1f3cf5d4f4363ea66f7c7d56d47f3bd44c3adf57832a3c7f1e26e5d097

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (caché de capas)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Usuario no root
RUN useradd --create-home --uid 10001 inboxbridge
USER inboxbridge

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "inboxbridge.healthcheck"]

CMD ["python", "-m", "inboxbridge.app"]
