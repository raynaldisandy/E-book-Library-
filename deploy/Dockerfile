# Dockerfile untuk Layanan REST API AI e-Library BPJS Kesehatan
FROM node:22-bookworm-slim AS dashboard
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src/ ./src/
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# Instalasi dependency sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements dan instal dependensi
COPY requirements.txt requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi, model, dan data
COPY api.py ./
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY --from=dashboard /frontend/dist/ ./frontend/dist/

# Port standar API
EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Pengecekan kesehatan container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Jalankan server
CMD ["python", "api.py"]
