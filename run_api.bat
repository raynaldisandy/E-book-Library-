@echo off
chcp 65001 > nul
echo =========================================================================
echo   BPJS KESEHATAN — AI e-LIBRARY RECOMMENDATION REST API SERVICE
echo =========================================================================
echo.
echo Menjalankan backend server pada http://localhost:8000 ...
echo Dokumentasi Swagger UI : http://localhost:8000/docs
echo Dokumentasi ReDoc      : http://localhost:8000/redoc
echo Dashboard Interaktif   : http://localhost:8000/
echo.
echo Tekan CTRL+C untuk menghentikan server.
echo =========================================================================
echo.

cd /d "%~dp0"
if not exist "frontend\dist\index.html" (
    call npm.cmd --prefix frontend ci
    if errorlevel 1 exit /b 1
    call npm.cmd --prefix frontend run build
    if errorlevel 1 exit /b 1
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" api.py
) else (
    python api.py
)

pause
