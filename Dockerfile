FROM python:3.13.16-slim-trixie AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.13.16-slim-trixie

RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --system appgroup && \
    adduser --system --group --home /home/appuser --shell /usr/sbin/nologin appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appgroup /home/appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

RUN python -c "import nltk; nltk.download('cmudict', download_dir='${NLTK_DATA}')" && \
    chmod -R a+rX "${NLTK_DATA}"

COPY backend/ ./backend/
RUN touch /app/backend/__init__.py && \
    chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]