FROM python:3.9-slim

# Ajustes de entorno y rutas de caché (evita '/.cache')
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Asegura que las carpetas de caché existen y son escribibles
RUN mkdir -p /app/.cache/huggingface/transformers /app/.cache/huggingface/hub && \
    chmod -R 777 /app/.cache

EXPOSE 8000

# Un solo worker para no duplicar uso de RAM/descargas; timeout alto para primera carga
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300", "--keep-alive", "5"]
