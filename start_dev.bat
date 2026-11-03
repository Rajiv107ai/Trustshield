@echo off
echo ===================================================
echo Starting TrustShield AI Platform (Backend + Frontend)
echo ===================================================

start "TrustShield FastAPI Backend" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

start "TrustShield Next.js Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Services launched!
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8000/docs
echo ===================================================

