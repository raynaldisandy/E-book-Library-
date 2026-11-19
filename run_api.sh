#!/usr/bin/env bash
# BPJS Kesehatan — AI e-Library REST API Service Runner
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ ! -f frontend/dist/index.html ]; then
    npm --prefix frontend ci
    npm --prefix frontend run build
fi

echo "========================================================================="
echo "  BPJS KESEHATAN — AI e-LIBRARY RECOMMENDATION REST API SERVICE"
echo "========================================================================="
echo "Menjalankan server pada http://0.0.0.0:8000 ..."
echo "Swagger UI Docs : http://localhost:8000/docs"
echo "ReDoc           : http://localhost:8000/redoc"
echo "Dashboard Web   : http://localhost:8000/"
echo "========================================================================="

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python api.py
else
    python3 api.py
fi
